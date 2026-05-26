"""
Stage 3: VLM-based quality validation of composited images.

Reads pairing_log.jsonl from Stage 2, calls Qwen3.5-27B on each composited image
to evaluate naturalness, then copies images to success/ or failed/ and writes
all decisions (with reasons) to validation_results.jsonl.

Outputs:
  outputs/compositing/success/{filename}
  outputs/compositing/failed/{filename}
  outputs/compositing/validation_results.jsonl
"""
import argparse
import datetime
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from lib.compositing_prompt import (
    STAGE3_SYSTEM_PROMPT,
    STAGE3_USER_PROMPT,
    parse_stage3_response,
)
from lib.compositing_utils import append_jsonl, load_done_ids, read_jsonl
from lib.vlm_client_pool import VLMClientPool

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compositing"
DEFAULT_PORTS = [8000, 5000]
DEFAULT_WORKERS = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 3: validate composited images via VLM")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--ports", nargs="+", type=int, default=DEFAULT_PORTS)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--model-name", default="Qwen3.5-27B")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


_append_lock = threading.Lock()


def validate_one(
    pairing: dict,
    pool: VLMClientPool,
    success_dir: Path,
    failed_dir: Path,
    results_path: Path,
) -> dict:
    output_path = Path(pairing["output_path"])
    if not output_path.exists():
        record = {
            "output_path": str(output_path),
            "bg_id": pairing.get("bg_id", ""),
            "face_id": pairing.get("face_id", ""),
            "decision": "reject",
            "reason": "Output file not found",
            "background_preserved": False,
            "person_detected": False,
            "natural_placement": False,
            "scene_appropriate": False,
            "not_posing": False,
            "parse_error": False,
            "raw_response": "",
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with _append_lock:
            append_jsonl(record, results_path)
        return record

    raw_response = pool.call(
        system_prompt=STAGE3_SYSTEM_PROMPT,
        user_prompt=STAGE3_USER_PROMPT,
        image_path=output_path,
    )

    parsed = parse_stage3_response(raw_response)
    decision = parsed["decision"]

    dest_dir = success_dir if decision == "keep" else failed_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / output_path.name
    shutil.copy2(str(output_path), str(dest_path))

    record = {
        "output_path": str(output_path),
        "dest_path": str(dest_path),
        "bg_id": pairing.get("bg_id", ""),
        "face_id": pairing.get("face_id", ""),
        "bg_class": pairing.get("bg_class", ""),
        "synthesis_seed": pairing.get("synthesis_seed"),
        "decision": decision,
        "reason": parsed.get("reason", ""),
        "background_preserved": parsed.get("background_preserved"),
        "person_detected": parsed.get("person_detected"),
        "natural_placement": parsed.get("natural_placement"),
        "scene_appropriate": parsed.get("scene_appropriate"),
        "not_posing": parsed.get("not_posing"),
        "parse_error": parsed.get("parse_error", False),
        "raw_response": parsed.get("raw_response", raw_response),
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with _append_lock:
        append_jsonl(record, results_path)
    return record


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    pairing_log_path = output_dir / "pairing_log.jsonl"
    results_path = output_dir / "validation_results.jsonl"
    success_dir = output_dir / "success"
    failed_dir = output_dir / "failed"

    pairing_records = read_jsonl(pairing_log_path)
    if not pairing_records:
        print("ERROR: pairing_log.jsonl is empty or missing. Run stage2 first.")
        sys.exit(1)

    done_output_paths = load_done_ids(results_path, "output_path")
    pending = [
        r for r in pairing_records
        if r.get("status") == "done" and r.get("output_path") not in done_output_paths
    ]
    if args.limit:
        pending = pending[: args.limit]

    print(
        f"Pairs total={len(pairing_records)}, "
        f"already validated={len(done_output_paths)}, "
        f"pending={len(pending)}"
    )
    if not pending:
        print("All images already validated.")
        _print_summary(results_path)
        return

    pool = VLMClientPool(
        ports=args.ports,
        host=args.host,
        model_name=args.model_name,
        max_tokens=args.max_tokens,
    )

    keep_count = 0
    reject_count = 0
    error_count = 0
    counter_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                validate_one, pairing, pool, success_dir, failed_dir, results_path
            ): pairing
            for pairing in pending
        }
        pbar = tqdm(total=len(pending), desc="Stage 3 - validation")
        for future in as_completed(futures):
            pairing = futures[future]
            try:
                result = future.result()
                with counter_lock:
                    if result["decision"] == "keep":
                        keep_count += 1
                    else:
                        reject_count += 1
                    if result.get("parse_error"):
                        error_count += 1
            except Exception as exc:
                with counter_lock:
                    reject_count += 1
                    error_count += 1
                tqdm.write(
                    f"[ERROR] {pairing.get('output_path', '->')}: {exc}"
                )
            pbar.update(1)
        pbar.close()

    print(f"\nDone.  results -> {results_path}")
    print(f"  keep={keep_count}, reject={reject_count}, parse_error={error_count}")
    print(f"  success/ -> {success_dir}")
    print(f"  failed/  -> {failed_dir}")


def _print_summary(results_path: Path) -> None:
    records = read_jsonl(results_path)
    keep = sum(1 for r in records if r.get("decision") == "keep")
    reject = sum(1 for r in records if r.get("decision") == "reject")
    print(f"Existing results: keep={keep}, reject={reject}, total={len(records)}")


if __name__ == "__main__":
    main()
