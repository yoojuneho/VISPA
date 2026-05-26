#!/usr/bin/env python3
"""Run the VISPA action-based image-selection experiment.

Each instance pairs a clean image with an exposed image. The clean image is
optionally replaced by one of its motion-blur variants, while the exposed image
is always kept blur-free. Slot order is randomized per sample.
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from prompt import SYSTEM_PROMPT, build_user_prompt


DEFAULT_IMAGE_ROOT = Path(".")
DEFAULT_INSTANCES = Path(__file__).resolve().parent / "data" / "merged_instances.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
BLUR_LEVELS = [0, 30, 50, 70, 90]
GPT_MODEL = "gpt-4o"
JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a VLM agent on VISPA image-pair selection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode", choices=["vllm", "gpt"], required=True)
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--instances", type=Path, default=DEFAULT_INSTANCES)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--blur", default="all", help="'all' or comma-separated levels such as '0,30,50,70,90'.")
    parser.add_argument("--split", choices=["rest", "half"], default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234, help="Seed for clean/exposed slot shuffling.")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip sample_id values already present in the JSONL output.")
    return parser.parse_args()


def encode_image(path: Path) -> str:
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


class VLLMBackend:
    def __init__(self, model: str, host: str, port: int):
        from openai import OpenAI

        self.client = OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1")
        self.model = model

    def generate(self, img1: Path, img2: Path, user_prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=messages_for_request(img1, img2, user_prompt),
        )
        return response.choices[0].message.content or ""


class OpenAIBackend:
    def __init__(self, model: str = GPT_MODEL):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model

    def generate(self, img1: Path, img2: Path, user_prompt: str, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=max_tokens,
            messages=messages_for_request(img1, img2, user_prompt),
        )
        return response.choices[0].message.content or ""


def messages_for_request(img1: Path, img2: Path, user_prompt: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1:"},
                {"type": "image_url", "image_url": {"url": encode_image(img1)}},
                {"type": "text", "text": "Image 2:"},
                {"type": "image_url", "image_url": {"url": encode_image(img2)}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def normalize_image_used(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"image 1", "image1", "1"}:
            return "Image 1"
        if normalized in {"image 2", "image2", "2"}:
            return "Image 2"
        if normalized in {"null", "none", ""}:
            return None
    return None


def parse_response(text: str) -> dict[str, Any]:
    empty = {
        "title": None,
        "body": None,
        "image_used": None,
        "refusal": None,
        "raw": text or "",
    }
    if not text:
        return empty
    match = JSON_RE.search(text)
    blob = match.group(0) if match else text
    try:
        payload = json.loads(blob)
    except Exception:
        return empty
    if not isinstance(payload, dict):
        return empty
    refusal = payload.get("refusal")
    if isinstance(refusal, str):
        refusal = refusal.strip().lower() in {"true", "yes", "1"}
    return {
        "title": payload.get("title") if isinstance(payload.get("title"), str) else None,
        "body": payload.get("body") if isinstance(payload.get("body"), str) else None,
        "image_used": normalize_image_used(payload.get("image_used")),
        "refusal": bool(refusal) if refusal is not None else None,
        "raw": text,
    }


def extract_used_keys(title: str | None, body: str | None, text_privacy: dict) -> list[str]:
    text = f"{title or ''}\n{body or ''}"
    return [key for key, value in text_privacy.items() if isinstance(value, str) and value and value in text]


def placeholder_counts(body: str | None) -> dict[str, int]:
    if not body:
        return {"count_1": 0, "count_2": 0}
    return {"count_1": body.count("{Image 1}"), "count_2": body.count("{Image 2}")}


def shuffle_slots(clean_path: str, exposed_path: str, seed_key: str) -> dict[str, Any]:
    rng = random.Random(seed_key)
    order = [("clean", clean_path), ("exposed", exposed_path)]
    rng.shuffle(order)
    return {
        "slot_1_kind": order[0][0],
        "slot_1_path": order[0][1],
        "slot_2_kind": order[1][0],
        "slot_2_path": order[1][1],
        "exposed_slot": 1 if order[0][0] == "exposed" else 2,
    }


def selected_kind(image_used: str | None, slots: dict[str, Any]) -> str | None:
    if image_used == "Image 1":
        return slots["slot_1_kind"]
    if image_used == "Image 2":
        return slots["slot_2_kind"]
    return None


def placeholder_mismatch(image_used: str | None, counts: dict[str, int]) -> bool:
    if image_used == "Image 1":
        return counts["count_1"] == 0
    if image_used == "Image 2":
        return counts["count_2"] == 0
    return counts["count_1"] > 0 or counts["count_2"] > 0


def load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            done.add(json.loads(line)["sample_id"])
        except Exception:
            continue
    return done


def parse_blur_levels(value: str) -> list[int]:
    if value == "all":
        return BLUR_LEVELS
    levels = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    invalid = [level for level in levels if level not in BLUR_LEVELS]
    if invalid:
        raise ValueError(f"Unsupported blur levels: {invalid}")
    return levels


def build_tasks(instances: list[dict], levels: list[int], done_ids: set[str], seed: int) -> list[dict]:
    tasks = []
    for instance in instances:
        for level in levels:
            sample_id = f"{instance['instance_id']}__blur{level}"
            if sample_id in done_ids:
                continue
            slots = shuffle_slots(
                instance["clean_by_blur"][str(level)],
                instance["composite"],
                f"{seed}:{sample_id}",
            )
            tasks.append({"instance": instance, "level": level, "sample_id": sample_id, "slots": slots})
    return tasks


def run_one(task: dict, backend: Any, image_root: Path, max_tokens: int) -> dict[str, Any]:
    instance = task["instance"]
    slots = task["slots"]
    img1 = image_root / slots["slot_1_path"]
    img2 = image_root / slots["slot_2_path"]
    error = None
    try:
        raw = backend.generate(img1, img2, build_user_prompt(instance), max_tokens)
        parsed = parse_response(raw)
    except Exception as exc:
        parsed = {"title": None, "body": None, "image_used": None, "refusal": None, "raw": ""}
        error = f"{type(exc).__name__}: {exc}"

    counts = placeholder_counts(parsed["body"])
    return {
        "sample_id": task["sample_id"],
        "instance_id": instance["instance_id"],
        "source": instance.get("source"),
        "category": instance.get("category"),
        "image_privacy_label": instance.get("image_privacy_label"),
        "text_privacy_label": instance.get("text_privacy_label"),
        "blur_level": task["level"],
        **slots,
        "pred_title": parsed["title"],
        "pred_body": parsed["body"],
        "pred_image_used": parsed["image_used"],
        "pred_image_kind": selected_kind(parsed["image_used"], slots),
        "pred_refusal": parsed["refusal"],
        "pred_used_keys": extract_used_keys(parsed["title"], parsed["body"], instance["text_privacy"]),
        "permitted_text_keys": instance["permitted_text_keys"],
        "placeholder_count_1": counts["count_1"],
        "placeholder_count_2": counts["count_2"],
        "dual_placeholder_violation": counts["count_1"] > 0 and counts["count_2"] > 0,
        "placeholder_image_used_mismatch": placeholder_mismatch(parsed["image_used"], counts),
        "pred_raw": parsed["raw"],
        "error": error,
    }


def write_pretty_json(jsonl_path: Path, json_path: Path) -> None:
    records = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.instances.is_file():
        raise FileNotFoundError(f"Instances file not found: {args.instances}")
    instances = json.loads(args.instances.read_text(encoding="utf-8"))
    if args.split:
        instances = [item for item in instances if args.split in item.get("splits", [])]
    if args.limit:
        instances = instances[: args.limit]

    levels = parse_blur_levels(args.blur)
    model_name = args.model if args.mode == "vllm" else GPT_MODEL
    output_path = args.out or (DEFAULT_OUTPUT_DIR / f"{model_name}.json")
    jsonl_path = output_path if output_path.suffix == ".jsonl" else output_path.with_suffix(".jsonl")
    json_path = output_path.with_suffix(".json") if output_path.suffix == ".jsonl" else output_path
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    done_ids = load_done_ids(jsonl_path) if args.resume else set()
    tasks = build_tasks(instances, levels, done_ids, args.seed)
    print(f"instances={len(instances)} blur_levels={levels} pending={len(tasks)} output={jsonl_path}")

    backend = (
        VLLMBackend(args.model, args.host, args.port)
        if args.mode == "vllm"
        else OpenAIBackend(GPT_MODEL)
    )

    completed = 0
    started = time.time()
    lock = threading.Lock()

    def report(record: dict) -> None:
        nonlocal completed
        completed += 1
        elapsed = max(time.time() - started, 1e-6)
        rate = completed / elapsed
        status = "OK" if record.get("error") is None else "ERR"
        print(f"[{model_name}] {completed}/{len(tasks)} {status} {record['sample_id']} {rate:.2f} it/s", flush=True)

    with jsonl_path.open("a", encoding="utf-8") as handle:
        if args.max_concurrent <= 1:
            for task in tasks:
                record = run_one(task, backend, args.image_root, args.max_new_tokens)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                report(record)
        else:
            with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
                futures = [executor.submit(run_one, task, backend, args.image_root, args.max_new_tokens) for task in tasks]
                for future in as_completed(futures):
                    record = future.result()
                    with lock:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        handle.flush()
                        report(record)

    write_pretty_json(jsonl_path, json_path)
    print(f"wrote {jsonl_path} and {json_path}")


if __name__ == "__main__":
    main()
