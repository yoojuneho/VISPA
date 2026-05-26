import argparse
import json
import os
import shutil
from pathlib import Path

from tqdm import tqdm

from lib.qwen_vlm_agent import QwenVLMClient, classify_candidates_stream, vlm_result_to_dict
from lib.sun397_filtering import (
    collect_candidates,
    dataclass_to_dict,
    parse_included_classes,
    write_jsonl,
)


DEFAULT_MIN_PIXEL_COUNT_EXCLUSIVE = 1024 * 768
DEFAULT_OUTPUT_DIR = "./outputs/sun397_vlm_filter"
DEFAULT_DATASET_DIR = "./datasets/SUN397"
DEFAULT_INCLUDED_CLASS_PATH = "./config/included_class.txt"
DEFAULT_EXCEPTION_CLASS_PATH = "./config/exception_class.txt"
DEFAULT_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
DEFAULT_MODEL_NAME = os.environ.get("VLLM_MODEL_NAME", "Qwen3.5-27B")
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")


def build_parser() -> argparse.ArgumentParser:
    #  help    CLI  .
    parser = argparse.ArgumentParser(
        description="SUN397          .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR, help="SUN397    ")
    parser.add_argument("--included-class-path", type=str, default=DEFAULT_INCLUDED_CLASS_PATH, help="Included SUN397 classes, one per line.")
    parser.add_argument("--exception-class-path", type=str, default=DEFAULT_EXCEPTION_CLASS_PATH, help="Excluded SUN397 classes, one per line.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=" JSONL, ,     ",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="all",
        help=" split. all   split   ",
    )
    parser.add_argument(
        "--min-pixel-count-exclusive",
        type=int,
        default=DEFAULT_MIN_PIXEL_COUNT_EXCLUSIVE,
        help="        ",
    )
    parser.add_argument("--limit", type=int, default=None, help="      ")
    parser.add_argument("--dry-run", action="store_true", help="VLM      ")
    parser.add_argument("--concurrency", type=int, default=4, help="  VLM  ")
    parser.add_argument("--retry-count", type=int, default=1, help="VLM     ")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help=" vLLM OpenAI  API ")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME, help="  ")
    parser.add_argument("--api-key", type=str, default=DEFAULT_API_KEY, help="OpenAI  API ")
    parser.add_argument("--timeout-seconds", type=float, default=180.0, help=" ()")
    parser.add_argument("--temperature", type=float, default=0.7, help="Qwen temperature")
    parser.add_argument("--top-p", type=float, default=0.8, help="Qwen top_p")
    parser.add_argument("--presence-penalty", type=float, default=1.5, help="Qwen presence_penalty")
    parser.add_argument("--top-k", type=int, default=20, help="Qwen extra_body.top_k")
    parser.add_argument("--min-p", type=float, default=0.0, help="Qwen extra_body.min_p")
    parser.add_argument("--max-tokens", type=int, default=256, help="VLM    ")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="      ",
    )
    return parser


def load_jsonl_records(jsonl_path: Path) -> list[dict]:
    # JSONL    dict  .
    if not jsonl_path.is_file():
        return []

    records: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_processed_image_paths(vlm_results_path: Path) -> set[str]:
    #            .
    if not vlm_results_path.is_file():
        return set()

    processed: set[str] = set()
    with vlm_results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            image_path = payload.get("image_path")
            if image_path:
                processed.add(str(image_path))
    return processed


def append_jsonl(records: list[dict], output_path: Path) -> None:
    #   JSONL   .
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def copy_included_images(vlm_result_dicts: list[dict], output_dir: Path) -> list[dict]:
    # keep         .
    included_root = output_dir / "included_images"
    mapping_records: list[dict] = []

    for result in vlm_result_dicts:
        if result.get("decision") != "keep":
            continue

        source_path = Path(result["image_path"])
        split_name = result["split"]
        class_name = result["class_name"].lstrip("/")
        destination_dir = included_root / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination_path = destination_dir / source_path.name
        shutil.copy2(source_path, destination_path)

        mapping_records.append(
            {
                "copied_image_path": str(destination_path),
                "source_image_path": str(source_path),
                "split": split_name,
                "class_name": result["class_name"],
                "decision": result["decision"],
                "reason": result["reason"],
            }
        )

    return mapping_records


def save_summary(output_dir: Path, payload: dict) -> None:
    #       JSON .
    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    #   :  -> dry-run  VLM  ->  
    args = build_parser().parse_args()

    dataset_dir = Path(args.dataset_dir).resolve()
    included_class_path = Path(args.included_class_path).resolve()
    exception_class_path = Path(args.exception_class_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefilter_path = output_dir / "prefilter_records.jsonl"
    candidate_path = output_dir / "eligible_candidates.jsonl"
    vlm_results_path = output_dir / "vlm_results.jsonl"
    included_mapping_path = output_dir / "included_mapping.json"

    if args.restart:
        # restart        .
        for path in [vlm_results_path, included_mapping_path]:
            if path.exists():
                path.unlink()

        included_root = output_dir / "included_images"
        if included_root.exists():
            shutil.rmtree(included_root)

    included_classes = parse_included_classes(included_class_path, exception_class_path)
    print(" .")
    candidates, prefilter_records, prefilter_summary = collect_candidates(
        dataset_dir=dataset_dir,
        requested_split=args.split,
        included_classes=included_classes,
        min_pixel_count_exclusive=args.min_pixel_count_exclusive,
        limit=args.limit,
        show_progress=True,
    )

    write_jsonl((dataclass_to_dict(record) for record in prefilter_records), prefilter_path)
    write_jsonl((dataclass_to_dict(candidate) for candidate in candidates), candidate_path)

    processed_image_paths: set[str] = set()
    if not args.restart:
        processed_image_paths = load_processed_image_paths(vlm_results_path)

    remaining_candidates = [candidate for candidate in candidates if candidate.image_path not in processed_image_paths]

    summary_payload = {
        "requested_split": args.split,
        "discovered_splits": prefilter_summary.discovered_splits,
        "included_class_count": prefilter_summary.included_class_count,
        "scanned_file_count": prefilter_summary.scanned_file_count,
        "eligible_count": prefilter_summary.eligible_count,
        "excluded_small_resolution_count": prefilter_summary.excluded_small_resolution_count,
        "image_read_error_count": prefilter_summary.image_read_error_count,
        "missing_class_directory_count": prefilter_summary.missing_class_directory_count,
        "already_processed_count": len(processed_image_paths),
        "remaining_candidate_count": len(remaining_candidates),
        "dry_run": args.dry_run,
        "concurrency": args.concurrency,
        "model_name": args.model_name,
    }
    save_summary(output_dir, summary_payload)

    if args.dry_run:
        print("[DRY-RUN]  ")
        print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
        return

    if not remaining_candidates:
        print("    .")
        return

    client = QwenVLMClient(
        base_url=args.base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        timeout_seconds=args.timeout_seconds,
        temperature=args.temperature,
        top_p=args.top_p,
        presence_penalty=args.presence_penalty,
        top_k=args.top_k,
        min_p=args.min_p,
        max_tokens=args.max_tokens,
    )

    print(f"VLM  .   : {len(remaining_candidates)}")
    print(f"  : {args.concurrency}")

    vlm_result_dicts: list[dict] = []
    vlm_progress = tqdm(total=len(remaining_candidates), desc="VLM  ", unit="image", mininterval=1.0)
    last_completed_count = 0

    def on_vlm_progress(stats: dict[str, int], result) -> None:
        #         JSONL .
        nonlocal last_completed_count
        completed_delta = stats["completed"] - last_completed_count
        if completed_delta > 0:
            vlm_progress.update(completed_delta)
            last_completed_count = stats["completed"]

        vlm_progress.set_postfix(
            keep=stats["keep"],
            reject=stats["reject"],
            failed=stats["failed"],
            in_flight=stats["in_flight"],
            submitted=stats["submitted"],
            refresh=False,
        )

        if result is not None:
            result_dict = vlm_result_to_dict(result)
            vlm_result_dicts.append(result_dict)
            append_jsonl([result_dict], vlm_results_path)

    classify_candidates_stream(
        client=client,
        candidates=remaining_candidates,
        concurrency=args.concurrency,
        retry_count=args.retry_count,
        progress_callback=on_vlm_progress,
    )
    vlm_progress.close()

    all_vlm_result_dicts = load_jsonl_records(vlm_results_path)
    mapping_records = copy_included_images(all_vlm_result_dicts, output_dir)
    included_mapping_path.write_text(json.dumps(mapping_records, ensure_ascii=False, indent=2), encoding="utf-8")

    keep_count = sum(1 for result in all_vlm_result_dicts if result["decision"] == "keep")
    reject_count = len(all_vlm_result_dicts) - keep_count
    summary_payload.update(
        {
            "processed_now_count": len(vlm_result_dicts),
            "processed_total_count": len(all_vlm_result_dicts),
            "keep_count": keep_count,
            "reject_count": reject_count,
            "included_mapping_count": len(mapping_records),
        }
    )
    save_summary(output_dir, summary_payload)

    print(" .")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
