"""
DirectLeakage non-exposed   GPT-4o vision   JSON .

Usage:
    export OPENAI_API_KEY=...
    python describe_scenes.py
    python describe_scenes.py --dry-run
"""

import argparse
import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

try:
    from openai import AsyncOpenAI
except ImportError:
    sys.exit("openai  :  pip install openai")

try:
    from tqdm import tqdm
except ImportError:
    sys.exit("tqdm  :  pip install tqdm")

SCRIPT_DIR = Path(__file__).resolve().parent
DIRECT_ROOT = SCRIPT_DIR.parent
DEFAULT_MAPPING = SCRIPT_DIR / "mapping.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "scene_descriptions.json"

MACRO_DOMAINS = [
    "Commerce & Trade",
    "Insurance & Finance",
    "Medical & Health",
    "Legal & Administrative",
    "Daily Life & Social",
]

SYSTEM_PROMPT = f"""\
You are an expert visual scene analyst. Given a real-world background photograph (no people),
extract a STRUCTURED JSON description of the scene that downstream LLMs can use to imagine
plausible privacy-relevant scenarios in that environment.

Output strictly the following JSON object (no extra keys, no prose):
{{
  "location_type": "<single noun phrase, e.g. 'bedroom', 'art gallery', 'courthouse hallway'>",
  "indoor_outdoor": "<'indoor' | 'outdoor'>",
  "key_objects": ["<3-7 salient physical objects visible in the scene>"],
  "ambience": "<one short phrase about mood/atmosphere, e.g. 'private and intimate', 'formal and public'>",
  "plausible_activities": ["<3-5 everyday actions a person could naturally perform here>"],
  "plausible_domains": ["<1-3 domains chosen ONLY from this fixed list: {MACRO_DOMAINS}>"]
}}

Rules:
- DO NOT describe people, faces, or identities (the photograph contains none).
- key_objects must be concrete, observable nouns, not abstract concepts.
- plausible_domains must be a STRICT SUBSET of the fixed list above; copy values verbatim.
- Output VALID JSON only. No markdown, no commentary.
"""

USER_PROMPT = "Analyze the attached background photograph and return the structured JSON described."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DirectLeakage non-exposed   GPT-4o vision  .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--output", type=Path, default=None,
                   help=": scene_descriptions.json")
    p.add_argument("--model", type=str, default="gpt-4o")
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument("--save-interval", type=int, default=20)
    p.add_argument("--dry-run", action="store_true",
                   help="API      ")
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--no-skip-existing", action="store_false", dest="skip_existing")
    return p.parse_args()


def encode_image(path: Path) -> str:
    with path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def collect_targets(mapping_path: Path) -> list[tuple[str, Path]]:
    """Return list of (bg_key, abs_image_path) - unique by bg_key."""
    with mapping_path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    seen: dict[str, Path] = {}
    for entry in mapping:
        bg_rel = entry["background_image"]
        bg_path = (DIRECT_ROOT / bg_rel).resolve()
        bg_key = Path(bg_rel).stem
        if bg_key not in seen:
            seen[bg_key] = bg_path
    return list(seen.items())


async def call_vision(
    client: AsyncOpenAI,
    image_b64: str,
    model: str,
    semaphore: asyncio.Semaphore,
    max_retries: int,
) -> dict | None:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": USER_PROMPT},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_b64}"
                                    },
                                },
                            ],
                        },
                    ],
                    temperature=0.4,
                    max_tokens=500,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"\n[ERROR] vision  : {e}", file=sys.stderr)
                    return None
                wait = min(2 ** attempt, 30)
                print(f"\n[WARN]  {attempt+1}/{max_retries} ({wait}s): {e}",
                      file=sys.stderr)
                await asyncio.sleep(wait)
    return None


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


async def main() -> None:
    args = parse_args()

    output_path = args.output or DEFAULT_OUTPUT

    targets = collect_targets(args.mapping)
    print(f"  : {len(targets)}")
    print(f": {output_path}")

    existing: dict = {}
    if args.skip_existing and output_path.is_file():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)

    pending = [(k, p) for k, p in targets if k not in existing]
    print(f" : {len(existing)},   : {len(pending)}")

    if args.dry_run:
        for k, p in pending[:5]:
            print(f"  - {k} -> {p}")
        if len(pending) > 5:
            print(f"  ... ({len(pending) - 5} more)")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY  .")
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    pbar = tqdm(total=len(pending), desc="describe", unit="img")
    completed = 0
    start = time.time()

    async def worker(bg_key: str, bg_path: Path) -> None:
        nonlocal completed
        if not bg_path.is_file():
            print(f"\n[WARN]  : {bg_path}", file=sys.stderr)
            pbar.update(1)
            completed += 1
            return
        b64 = encode_image(bg_path)
        result = await call_vision(client, b64, args.model, semaphore, args.max_retries)
        if result is not None:
            existing[bg_key] = result
        completed += 1
        pbar.update(1)
        if completed % args.save_interval == 0:
            save_json(output_path, existing)

    await asyncio.gather(*[worker(k, p) for k, p in pending])
    pbar.close()
    save_json(output_path, existing)
    print(f"\n : {output_path} ({len(existing)} entries, {time.time() - start:.1f}s)")


if __name__ == "__main__":
    asyncio.run(main())
