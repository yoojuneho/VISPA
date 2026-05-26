"""
Stage 1: Generate VLM-based compositing prompts for each background image.

For every background in outputs/class_filtered, calls Qwen3.5-27B (via vLLM on port 8000)
to produce TWO scenario JSON records per background, each with a distinct placement and
activity. Fields: scenario_index, scene_description, scene_category, placement_hint,
edit_prompt, negative_prompt.

Results are appended to outputs/compositing/prompts.jsonl.
Already-processed backgrounds are skipped automatically (resumable).
"""
import argparse
import datetime
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from lib.compositing_prompt import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    parse_stage1_scenarios,
)
from lib.compositing_utils import append_jsonl, read_jsonl
from lib.vlm_client_pool import VLMClientPool

DEFAULT_MAPPING = PROJECT_ROOT / "outputs" / "class_filtered" / "filename_mapping.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compositing"
DEFAULT_PORTS = [8000]
DEFAULT_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1: generate compositing prompts via VLM")
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING), help="Path to filename_mapping.json")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output root directory")
    parser.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS, help="vLLM server ports")
    parser.add_argument("--host", default="localhost", help="vLLM server host")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Concurrent VLM request threads")
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Process at most N backgrounds (for testing)")
    return parser.parse_args()


_append_lock = threading.Lock()


def process_one(
    entry: dict,
    pool: VLMClientPool,
    prompts_path: Path,
) -> list[dict]:
    bg_id = Path(entry["renamed_file_name"]).stem
    bg_path = Path(entry["output_image_path"])

    raw_response = pool.call(
        system_prompt=STAGE1_SYSTEM_PROMPT,
        user_prompt=STAGE1_USER_PROMPT,
        image_path=bg_path,
    )

    scenarios = parse_stage1_scenarios(raw_response)
    records = []
    for scenario in scenarios:
        record = {
            "bg_id": bg_id,
            "bg_filename": entry["renamed_file_name"],
            "bg_path": str(bg_path),
            "bg_class": entry["class_leaf_name"],
            "index": entry["index"],
            "scenario_index": scenario["scenario_index"],
            "scene_description": scenario.get("scene_description", ""),
            "scene_category": scenario.get("scene_category", "other"),
            "placement_hint": scenario.get("placement_hint", ""),
            "edit_prompt": scenario.get("edit_prompt", ""),
            "negative_prompt": scenario.get("negative_prompt", ""),
            "parse_error": scenario.get("parse_error", False),
            "raw_response": scenario.get("raw_response", raw_response),
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with _append_lock:
            append_jsonl(record, prompts_path)
        records.append(record)
    return records


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompts_path = output_dir / "prompts.jsonl"

    mapping: list[dict] = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if args.limit:
        mapping = mapping[: args.limit]

    # A bg_id is "done" only if BOTH scenario 0 and scenario 1 are stored without parse error
    existing_scenarios: dict[str, set] = {}
    for r in read_jsonl(prompts_path):
        if not r.get("parse_error", False) and r.get("edit_prompt"):
            existing_scenarios.setdefault(r["bg_id"], set()).add(r.get("scenario_index", 0))
    pending = [
        e for e in mapping
        if len(existing_scenarios.get(Path(e["renamed_file_name"]).stem, set())) < 2
    ]
    n_done = len(mapping) - len(pending)

    print(f"Backgrounds total={len(mapping)}, done={n_done}, pending={len(pending)}")
    if not pending:
        print("All backgrounds already processed.")
        return

    pool = VLMClientPool(
        ports=args.ports,
        host=args.host,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    failed_indices: list[int] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_one, entry, pool, prompts_path): entry
            for entry in pending
        }
        pbar = tqdm(total=len(pending), desc="Stage 1 - prompt generation")
        for future in as_completed(futures):
            entry = futures[future]
            try:
                results = future.result()
                if any(r.get("parse_error") for r in results):
                    failed_indices.append(entry["index"])
            except Exception as exc:
                failed_indices.append(entry["index"])
                tqdm.write(f"[ERROR] index={entry['index']} {entry['renamed_file_name']}: {exc}")
            pbar.update(1)
        pbar.close()

    n_success = len(pending) - len(failed_indices)
    print(f"\nDone. prompts.jsonl -> {prompts_path}")
    print(f"  success={n_success}, parse_error={len(failed_indices)}")
    if failed_indices:
        preview = failed_indices[:20]
        print(f"  failed indices (first 20): {preview}")


if __name__ == "__main__":
    main()
