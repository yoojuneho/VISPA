"""
Stage 2: Multi-GPU parallel image synthesis using Qwen-Image-Edit.

Reads prompts.jsonl from Stage 1, generates fixed face-background pairings
(seed=42), then synthesizes one composited image per pair across 2 pipeline
instances, each spanning 2 GPUs via device_map="balanced".

Outputs:
  outputs/compositing/raw/{bg_id}_face{face_id}.png
  outputs/compositing/pairing_log.jsonl
"""
import argparse
import datetime
import json
import sys
import threading
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from lib.compositing_utils import (
    append_jsonl,
    collect_face_paths,
    compute_output_size,
    generate_pairings,
    load_done_ids,
    read_jsonl,
    resize_if_needed,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "compositing"
DEFAULT_FACE_DIR = PROJECT_ROOT / "datasets" / "ours" / "face"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ImgEdit" / "Qwen-Image-Edit-2511"
DEFAULT_GPU_IDS = [0, 1, 2, 3]
DEFAULT_VRAM_PER_GPU = 44

_PIPELINES: dict[int, object] = {}         # pair_idx -> pipeline
_PIPELINE_PRIMARY_GPU: dict[int, int] = {} # pair_idx -> first GPU id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2: synthesize composited images")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--face-dir", default=str(DEFAULT_FACE_DIR))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=DEFAULT_GPU_IDS,
                        help="Synthesis GPUs; consecutive pairs share one pipeline")
    parser.add_argument("--vram-per-gpu", type=int, default=DEFAULT_VRAM_PER_GPU)
    parser.add_argument("--pairing-seed", type=int, default=42)
    parser.add_argument("--n-per-bg", type=int, default=2)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=None, help="Limit total pairs (for testing)")
    return parser.parse_args()


def preload_pipelines(
    gpu_ids: list[int],
    model_dir: str,
    vram_per_gpu_gb: int = DEFAULT_VRAM_PER_GPU,
) -> None:
    from diffusers import QwenImageEditPlusPipeline

    gpu_pairs = [gpu_ids[i : i + 2] for i in range(0, len(gpu_ids), 2)]
    n_visible = torch.cuda.device_count()
    load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}

    for pair_idx, pair in enumerate(gpu_pairs):
        print(f"  Loading pipeline {pair_idx} on GPUs {pair} ...")
        max_memory = {
            gid: (f"{vram_per_gpu_gb}GiB" if gid in pair else "0MiB")
            for gid in range(n_visible)
        }
        max_memory["cpu"] = "60GiB"
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            model_dir,
            device_map="balanced",
            max_memory=max_memory,
            **load_kwargs,
        )
        _PIPELINES[pair_idx] = pipe
        _PIPELINE_PRIMARY_GPU[pair_idx] = pair[0]
    print(f"All {len(gpu_pairs)} pipelines loaded (2 GPUs each).")


def synthesize_one(
    pairing: dict,
    prompt_record: dict,
    pair_idx: int,
    raw_dir: Path,
    pairing_log_path: Path,
    inf_steps: int,
    cfg_scale: float,
    log_lock: threading.Lock,
) -> dict:
    bg_path = Path(pairing["bg_path"])
    face_path = Path(pairing["face_path"])
    bg_id = pairing["bg_id"]
    face_id = pairing["face_id"]
    seed = pairing["synthesis_seed"]

    output_filename = f"{bg_id}_face{face_id}.png"
    output_path = raw_dir / output_filename

    image1 = Image.open(bg_path).convert("RGB")
    out_w, out_h = compute_output_size(image1)
    image1 = image1.resize((out_w, out_h), Image.LANCZOS)
    image2 = resize_if_needed(Image.open(face_path).convert("RGB"))

    prompt = prompt_record.get("edit_prompt", "")
    negative_prompt = prompt_record.get("negative_prompt", "")

    pipe = _PIPELINES[pair_idx]
    primary_gpu = _PIPELINE_PRIMARY_GPU[pair_idx]
    device = torch.device(f"cuda:{primary_gpu}")
    generator = torch.Generator(device=device).manual_seed(seed)

    output = pipe(
        image=[image1, image2],
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        true_cfg_scale=cfg_scale,
        num_inference_steps=inf_steps,
        num_images_per_prompt=1,
        height=out_h,
        width=out_w,
    )
    result_img = output.images[0]
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_img.save(str(output_path))

    record = {
        **pairing,
        "output_path": str(output_path),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "num_inference_steps": inf_steps,
        "cfg_scale": cfg_scale,
        "gpu_ids": [primary_gpu, primary_gpu + 1],
        "status": "done",
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with log_lock:
        append_jsonl(record, pairing_log_path)
    return record


def gpu_worker(
    pair_idx: int,
    tasks: list[dict],
    prompt_map: dict[str, dict],
    raw_dir: Path,
    pairing_log_path: Path,
    inf_steps: int,
    cfg_scale: float,
    log_lock: threading.Lock,
    pbar_lock: threading.Lock,
    pbar: tqdm,
    failed: list,
) -> None:
    for pairing in tasks:
        bg_id = pairing["bg_id"]
        prompt_key = f"{bg_id}_s{pairing['pair_index']}"
        prompt_record = prompt_map.get(prompt_key, {})
        try:
            synthesize_one(
                pairing=pairing,
                prompt_record=prompt_record,
                pair_idx=pair_idx,
                raw_dir=raw_dir,
                pairing_log_path=pairing_log_path,
                inf_steps=inf_steps,
                cfg_scale=cfg_scale,
                log_lock=log_lock,
            )
        except Exception as exc:
            key = f"{bg_id}_face{pairing['face_id']}"
            with pbar_lock:
                tqdm.write(f"[ERROR] pipe{pair_idx} {key}: {exc}")
            failed.append(key)
        with pbar_lock:
            pbar.update(1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    pairing_log_path = output_dir / "pairing_log.jsonl"
    prompts_path = output_dir / "prompts.jsonl"

    prompt_records = read_jsonl(prompts_path)
    if not prompt_records:
        print("ERROR: prompts.jsonl is empty or missing. Run stage1 first.")
        sys.exit(1)

    # Key by f"{bg_id}_s{scenario_index}" - last-wins for deduplicated lookup
    prompt_map: dict[str, dict] = {}
    for r in prompt_records:
        if not r.get("parse_error", False) and r.get("edit_prompt"):
            key = f"{r['bg_id']}_s{r.get('scenario_index', 0)}"
            prompt_map[key] = r
    valid_records = [r for r in prompt_records if not r.get("parse_error", False)]
    print(f"Prompt records: total={len(prompt_records)}, valid={len(valid_records)}, scenarios={len(prompt_map)}")

    # Build bg_entries from unique bg_ids that have at least one valid scenario
    seen_bg: dict[str, dict] = {}
    for r in valid_records:
        if r["bg_id"] not in seen_bg:
            seen_bg[r["bg_id"]] = r
    bg_entries = sorted(
        [
            {
                "index": r["index"],
                "renamed_file_name": r["bg_filename"],
                "output_image_path": r["bg_path"],
                "class_leaf_name": r["bg_class"],
            }
            for r in seen_bg.values()
        ],
        key=lambda e: e["index"],
    )

    face_paths = collect_face_paths(Path(args.face_dir))
    if not face_paths:
        print(f"ERROR: No face images found in {args.face_dir}")
        sys.exit(1)
    print(f"Face images: {len(face_paths)}")

    pairings = generate_pairings(
        bg_entries=bg_entries,
        face_paths=face_paths,
        n_per_bg=args.n_per_bg,
        seed=args.pairing_seed,
    )
    if args.limit:
        pairings = pairings[: args.limit]

    done_output_paths = load_done_ids(pairing_log_path, "output_path")
    pending = [
        p for p in pairings
        if str(raw_dir / f"{p['bg_id']}_face{p['face_id']}.png") not in done_output_paths
    ]
    print(
        f"Pairs: total={len(pairings)}, done={len(pairings) - len(pending)}, pending={len(pending)}"
    )
    if not pending:
        print("All pairs already synthesized.")
        return

    gpu_pairs = [args.gpu_ids[i : i + 2] for i in range(0, len(args.gpu_ids), 2)]
    print(f"\nPre-loading pipelines on GPU pairs {gpu_pairs} ...")
    preload_pipelines(args.gpu_ids, str(args.model_dir), args.vram_per_gpu)

    n_pairs = len(gpu_pairs)
    per_pair: dict[int, list] = {pi: [] for pi in range(n_pairs)}
    for i, p in enumerate(pending):
        per_pair[i % n_pairs].append(p)

    log_lock = threading.Lock()
    pbar_lock = threading.Lock()
    failed: list = []

    pbar = tqdm(total=len(pending), desc="Stage 2 - synthesis")
    threads = []
    for pair_idx in range(n_pairs):
        t = threading.Thread(
            target=gpu_worker,
            args=(
                pair_idx,
                per_pair[pair_idx],
                prompt_map,
                raw_dir,
                pairing_log_path,
                args.num_inference_steps,
                args.cfg_scale,
                log_lock,
                pbar_lock,
                pbar,
                failed,
            ),
            daemon=False,
        )
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    pbar.close()

    n_success = len(pending) - len(failed)
    print(f"\nDone.  raw/ -> {raw_dir}")
    print(f"  success={n_success}, failed={len(failed)}")
    if failed:
        print(f"  failed (first 20): {failed[:20]}")

    print(f"\nPairing log -> {pairing_log_path}")


if __name__ == "__main__":
    main()
