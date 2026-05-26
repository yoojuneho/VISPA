"""
run_image_pipeline.py - All-stages-overlap pipeline runner

Data flow:
  GPUs 0+1  | Stage 2: synthesis pipeline A  (2 GPUs each, no CPU offload)
  GPUs 2+3  | Stage 2: synthesis pipeline B
  GPUs 4-7  | Stage 1: VLM prompts --> synthesis_queue --> Stage 2
             | Stage 3: VLM valid.  <-- validation_queue <-- Stage 2

All three stages run concurrently. Each synthesis pipeline spans 2 GPUs
via device_map="balanced", eliminating CPU offloading for full VRAM speed.

Usage:
  python scripts/run_image_pipeline.py
  python scripts/run_image_pipeline.py --limit 10        # quick test
  python scripts/run_image_pipeline.py --skip-stage1     # reuse prompts.jsonl
  python scripts/run_image_pipeline.py --skip-stage1 --skip-stage3  # synth only
"""
import argparse
import datetime
import json
import shutil
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import torch
from PIL import Image
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent
sys.path.insert(0, str(BASE_DIR))

from lib.compositing_prompt import (
    STAGE1_SYSTEM_PROMPT,
    STAGE1_USER_PROMPT,
    STAGE3_SYSTEM_PROMPT,
    STAGE3_USER_PROMPT,
    parse_stage1_scenarios,
    parse_stage3_response,
)
from lib.compositing_utils import (
    append_jsonl,
    collect_face_paths,
    compute_output_size,
    generate_pairings,
    load_done_ids,
    read_jsonl,
    resize_if_needed,
)
from lib.vlm_client_pool import VLMClientPool

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pl"
DEFAULT_FACE_DIR = PROJECT_ROOT / "datasets" / "ours" / "face"
DEFAULT_MAPPING = PROJECT_ROOT / "outputs" / "class_filtered" / "filename_mapping.json"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "ImgEdit" / "Qwen-Image-Edit-2511"
DEFAULT_GPU_IDS = [0, 1, 2, 3]
DEFAULT_VLM_PORTS = [8000]
DEFAULT_VRAM_PER_GPU = 44
_SENTINEL = object()

_PIPELINES: dict[int, object] = {}       # pair_idx -> pipeline
_PIPELINE_PRIMARY_GPU: dict[int, int] = {}  # pair_idx -> first GPU id (for generator)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parallel 3-stage compositing pipeline")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--face-dir", default=str(DEFAULT_FACE_DIR))
    p.add_argument("--mapping", default=str(DEFAULT_MAPPING))
    p.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--gpu-ids", nargs="+", type=int, default=DEFAULT_GPU_IDS,
                   help="Synthesis GPU IDs. Consecutive pairs share one pipeline: [0,1,2,3] -> 2 pipelines")
    p.add_argument("--vram-per-gpu", type=int, default=DEFAULT_VRAM_PER_GPU,
                   help="VRAM budget per GPU in GiB (used for device_map max_memory)")
    p.add_argument("--vlm-ports", nargs="+", type=int, default=DEFAULT_VLM_PORTS)
    p.add_argument("--vlm-host", default="localhost")
    p.add_argument("--vlm-model-name", default="Qwen3.5-27B")
    p.add_argument("--vlm-max-tokens", type=int, default=2048,
                   help="max_tokens for Stage 1 VLM calls (2-scenario JSON needs ~620 tokens)")
    p.add_argument("--vlm-workers", type=int, default=16,
                   help="Concurrent VLM threads (shared by Stage 1 and Stage 3)")
    p.add_argument("--pairing-seed", type=int, default=42)
    p.add_argument("--n-per-bg", type=int, default=2)
    p.add_argument("--num-inference-steps", type=int, default=30)
    p.add_argument("--cfg-scale", type=float, default=4.0)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-stage1", action="store_true",
                   help="Skip VLM prompt generation, reuse existing prompts.jsonl")
    p.add_argument("--skip-stage2", action="store_true",
                   help="Skip synthesis, send existing pairing_log entries to Stage 3")
    p.add_argument("--skip-stage3", action="store_true",
                   help="Skip VLM validation")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Pipeline loader
# -----------------------------------------------------------------------------
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
        print(f"  [init] Loading pipeline {pair_idx} on GPUs {pair} ...")
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

    print(f"  [init] {len(gpu_pairs)} synthesis pipelines ready (2 GPUs each).\n")


# -----------------------------------------------------------------------------
# Stage 1 worker - VLM prompt generation
# -----------------------------------------------------------------------------
def _stage1_worker(
    bg_task_q: Queue,
    synthesis_q: Queue,
    pool: VLMClientPool,
    prompts_path: Path,
    pairings_by_bg: dict[str, list[dict]],
    s1_pending_keys: set[str],
    locks: dict,
    pbars: dict,
) -> None:
    while True:
        try:
            bg_entry = bg_task_q.get(timeout=1)
        except Empty:
            continue
        if bg_entry is _SENTINEL:
            break

        bg_id = Path(bg_entry["renamed_file_name"]).stem
        bg_path = Path(bg_entry["output_image_path"])

        try:
            raw = pool.call(
                system_prompt=STAGE1_SYSTEM_PROMPT,
                user_prompt=STAGE1_USER_PROMPT,
                image_path=bg_path,
            )
            scenarios = parse_stage1_scenarios(raw)
        except Exception as exc:
            scenarios = [
                {
                    "scenario_index": i,
                    "scene_description": "", "scene_category": "other",
                    "placement_hint": "", "edit_prompt": "", "negative_prompt": "",
                    "parse_error": True, "raw_response": str(exc),
                }
                for i in range(2)
            ]

        for scenario in scenarios:
            si = scenario["scenario_index"]
            record = {
                "bg_id": bg_id,
                "bg_filename": bg_entry["renamed_file_name"],
                "bg_path": str(bg_path),
                "bg_class": bg_entry["class_leaf_name"],
                "index": bg_entry["index"],
                "scenario_index": si,
                **{k: scenario.get(k, "") for k in (
                    "scene_description", "scene_category", "placement_hint",
                    "edit_prompt", "negative_prompt",
                )},
                "parse_error": scenario.get("parse_error", False),
                "raw_response": scenario.get("raw_response", ""),
                "timestamp": datetime.datetime.now().isoformat(),
            }
            with locks["prompts"]:
                append_jsonl(record, prompts_path)

            if not scenario.get("parse_error", False) and scenario.get("edit_prompt"):
                for pairing in pairings_by_bg.get(bg_id, []):
                    if pairing["pair_index"] == si:
                        pair_key = f"{bg_id}_face{pairing['face_id']}"
                        if pair_key in s1_pending_keys:
                            synthesis_q.put({
                                "pairing": pairing,
                                "edit_prompt": scenario.get("edit_prompt", ""),
                                "negative_prompt": scenario.get("negative_prompt", ""),
                            })

        with locks["cnt"]:
            locks["cnt_obj"]["s1_done"] += 1
        with locks["pbar"]:
            pbars["s1"].update(1)


# -----------------------------------------------------------------------------
# Stage 2 worker - image synthesis (one per pipeline pair)
# -----------------------------------------------------------------------------
def _stage2_worker(
    pair_idx: int,
    synthesis_q: Queue,
    validation_q: Queue,
    pairing_log_path: Path,
    raw_dir: Path,
    inf_steps: int,
    cfg_scale: float,
    do_stage3: bool,
    locks: dict,
    pbars: dict,
) -> None:
    while True:
        try:
            item = synthesis_q.get(timeout=2)
        except Empty:
            continue
        if item is _SENTINEL:
            break

        pairing = item["pairing"]
        edit_prompt = item["edit_prompt"]
        neg_prompt = item["negative_prompt"]
        bg_id = pairing["bg_id"]
        face_id = pairing["face_id"]
        seed = pairing["synthesis_seed"]
        output_path = raw_dir / f"{bg_id}_face{face_id}.png"

        status = "error"
        try:
            bg_img = Image.open(pairing["bg_path"]).convert("RGB")
            out_w, out_h = compute_output_size(bg_img)
            image1 = bg_img.resize((out_w, out_h), Image.LANCZOS)
            image2 = resize_if_needed(Image.open(pairing["face_path"]).convert("RGB"))

            pipe = _PIPELINES[pair_idx]
            primary_gpu = _PIPELINE_PRIMARY_GPU[pair_idx]
            device = torch.device(f"cuda:{primary_gpu}")
            gen = torch.Generator(device=device).manual_seed(seed)

            out = pipe(
                image=[image1, image2],
                prompt=edit_prompt,
                negative_prompt=neg_prompt,
                generator=gen,
                true_cfg_scale=cfg_scale,
                num_inference_steps=inf_steps,
                num_images_per_prompt=1,
                height=out_h,
                width=out_w,
            )
            raw_dir.mkdir(parents=True, exist_ok=True)
            out.images[0].save(str(output_path))
            status = "done"
        except Exception as exc:
            tqdm.write(f"[Stage2 pipe{pair_idx}] ERROR {bg_id}_face{face_id}: {exc}")

        record = {
            **pairing,
            "output_path": str(output_path),
            "prompt": edit_prompt,
            "negative_prompt": neg_prompt,
            "num_inference_steps": inf_steps,
            "cfg_scale": cfg_scale,
            "gpu_ids": list(range(
                _PIPELINE_PRIMARY_GPU[pair_idx],
                _PIPELINE_PRIMARY_GPU[pair_idx] + 2,
            )),
            "status": status,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        with locks["pairing_log"]:
            append_jsonl(record, pairing_log_path)

        if status == "done" and do_stage3:
            validation_q.put(record)

        with locks["cnt"]:
            locks["cnt_obj"]["s2_done"] += 1
            if status == "done":
                locks["cnt_obj"]["s2_ok"] += 1
        with locks["pbar"]:
            pbars["s2"].update(1)


# -----------------------------------------------------------------------------
# Stage 3 worker - VLM validation
# -----------------------------------------------------------------------------
def _stage3_worker(
    validation_q: Queue,
    pool: VLMClientPool,
    success_dir: Path,
    failed_dir: Path,
    results_path: Path,
    locks: dict,
    pbars: dict,
) -> None:
    while True:
        try:
            record = validation_q.get(timeout=2)
        except Empty:
            continue
        if record is _SENTINEL:
            break

        output_path = Path(record["output_path"])
        decision, reason, parsed = "reject", "output file missing", {}

        if output_path.exists():
            try:
                raw = pool.call(
                    system_prompt=STAGE3_SYSTEM_PROMPT,
                    user_prompt=STAGE3_USER_PROMPT,
                    image_path=output_path,
                )
                parsed = parse_stage3_response(raw)
                decision = parsed["decision"]
                reason = parsed.get("reason", "")
            except Exception as exc:
                decision, reason = "reject", str(exc)

        dest_dir = success_dir if decision == "keep" else failed_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(str(output_path), str(dest_dir / output_path.name))
        except Exception:
            pass

        result = {
            **record,
            "dest_path": str(dest_dir / output_path.name),
            "decision": decision,
            "reason": reason,
            "background_preserved": parsed.get("background_preserved"),
            "person_detected": parsed.get("person_detected"),
            "natural_placement": parsed.get("natural_placement"),
            "scene_appropriate": parsed.get("scene_appropriate"),
            "not_posing": parsed.get("not_posing"),
            "parse_error": parsed.get("parse_error", False),
            "raw_response": parsed.get("raw_response", ""),
            "val_timestamp": datetime.datetime.now().isoformat(),
        }
        with locks["results"]:
            append_jsonl(result, results_path)

        with locks["cnt"]:
            locks["cnt_obj"]["s3_done"] += 1
            if decision == "keep":
                locks["cnt_obj"]["s3_keep"] += 1
        with locks["pbar"]:
            pbars["s3"].update(1)
            pbars["s3"].total = (pbars["s3"].total or 0) + 0  # keep display fresh
            pbars["s3"].refresh()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    prompts_path = output_dir / "prompts.jsonl"
    pairing_log_path = output_dir / "pairing_log.jsonl"
    results_path = output_dir / "validation_results.jsonl"
    success_dir = output_dir / "success"
    failed_dir = output_dir / "failed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Load mapping & generate pairings ---------------------------------
    mapping: list[dict] = json.loads(Path(args.mapping).read_text(encoding="utf-8"))
    if args.limit:
        mapping = mapping[: args.limit]
    bg_entries = sorted(mapping, key=lambda e: e["index"])

    face_paths = collect_face_paths(Path(args.face_dir))
    if not face_paths:
        print(f"ERROR: no face images in {args.face_dir}")
        sys.exit(1)

    all_pairings = generate_pairings(bg_entries, face_paths, args.n_per_bg, args.pairing_seed)
    pairings_by_bg: dict[str, list[dict]] = {}
    for p in all_pairings:
        pairings_by_bg.setdefault(p["bg_id"], []).append(p)

    # -- Resumability -----------------------------------------------------
    # Key prompts by f"{bg_id}_s{scenario_index}" - last-wins handles duplicate runs
    existing_prompts: dict[str, dict] = {}
    for _r in read_jsonl(prompts_path):
        if not _r.get("parse_error", False) and _r.get("edit_prompt"):
            _key = f"{_r['bg_id']}_s{_r.get('scenario_index', 0)}"
            existing_prompts[_key] = _r

    done_output_paths = load_done_ids(pairing_log_path, "output_path")
    done_val_paths = load_done_ids(results_path, "output_path")

    pending_pairings = [
        p for p in all_pairings
        if str(raw_dir / f"{p['bg_id']}_face{p['face_id']}.png") not in done_output_paths
    ]

    # Stage 3 backlog: synthesized but not yet validated
    pending_val = [
        r for r in read_jsonl(pairing_log_path)
        if r.get("status") == "done" and r.get("output_path") not in done_val_paths
    ]

    # group_a: pairings whose per-scenario prompt is already stored
    group_a = [
        p for p in pending_pairings
        if f"{p['bg_id']}_s{p['pair_index']}" in existing_prompts
    ]
    group_a_pair_keys = {f"{p['bg_id']}_face{p['face_id']}" for p in group_a}
    # group_b: bg_ids where at least one pairing still needs a prompt
    group_b_bg_ids = {
        p["bg_id"] for p in pending_pairings
        if f"{p['bg_id']}_s{p['pair_index']}" not in existing_prompts
    }
    pending_bg = [e for e in bg_entries
                  if Path(e["renamed_file_name"]).stem in group_b_bg_ids]
    # Stage-1 workers must NOT re-push pairings already covered by group_a
    s1_pending_keys = {
        f"{p['bg_id']}_face{p['face_id']}" for p in pending_pairings
    } - group_a_pair_keys

    n_s1 = len(pending_bg)
    n_s2 = len(pending_pairings)
    n_s3_pre = len(pending_val)

    print(f"\n{'=' * 60}")
    print(f"  Parallel Face Compositing Pipeline")
    print(f"{'=' * 60}")
    print(f"  Backgrounds   : {len(mapping)} total  |  {n_s1} need Stage 1")
    print(f"  Pairs (synth) : {len(all_pairings)} total  |  {n_s2} pending")
    print(f"  Backlog val.  : {n_s3_pre} from prior run")
    gpu_pairs = [args.gpu_ids[i : i + 2] for i in range(0, len(args.gpu_ids), 2)]
    print(f"  GPU pairs     : {gpu_pairs}  |  vLLM ports: {args.vlm_ports}")
    print(f"  inf-steps     : {args.num_inference_steps}  |  n-per-bg: {args.n_per_bg}")
    if args.limit:
        print(f"  limit         : {args.limit}")
    print(f"{'=' * 60}\n")

    if n_s1 == 0 and n_s2 == 0 and n_s3_pre == 0:
        print("Nothing to do - all items already processed.")
        return

    # -- Pre-load synthesis pipelines -------------------------------------
    do_stage2 = not args.skip_stage2 and n_s2 > 0
    do_stage3 = not args.skip_stage3
    do_stage1 = not args.skip_stage1 and n_s1 > 0

    if do_stage2:
        print("Pre-loading synthesis pipelines ...")
        preload_pipelines(args.gpu_ids, args.model_dir, args.vram_per_gpu)

    vlm_pool = VLMClientPool(
        ports=args.vlm_ports,
        host=args.vlm_host,
        model_name=args.vlm_model_name,
        max_tokens=args.vlm_max_tokens,
    )

    # -- Queues & shared state ---------------------------------------------
    synthesis_q: Queue = Queue(maxsize=1024)
    validation_q: Queue = Queue(maxsize=1024)

    cnt_obj = {
        "s1_done": 0,
        "s2_done": 0, "s2_ok": 0,
        "s3_done": 0, "s3_keep": 0,
    }
    pbar_s1 = tqdm(total=n_s1 if do_stage1 else 0,
                   desc="Stage1 prompts  ", position=0, leave=True)
    pbar_s2 = tqdm(total=n_s2 if do_stage2 else 0,
                   desc="Stage2 synthesis", position=1, leave=True)
    pbar_s3 = tqdm(total=None,
                   desc="Stage3 validate ", position=2, leave=True)

    locks = {
        "cnt": threading.Lock(),
        "cnt_obj": cnt_obj,
        "prompts": threading.Lock(),
        "pairing_log": threading.Lock(),
        "results": threading.Lock(),
        "pbar": threading.Lock(),
    }
    pbars = {"s1": pbar_s1, "s2": pbar_s2, "s3": pbar_s3}

    # -- Start Stage 3 workers ---------------------------------------------
    s3_threads = []
    if do_stage3:
        for _ in range(args.vlm_workers):
            t = threading.Thread(
                target=_stage3_worker,
                args=(validation_q, vlm_pool, success_dir, failed_dir,
                      results_path, locks, pbars),
                daemon=False,
            )
            t.start()
            s3_threads.append(t)
        # Pre-seed Stage 3 with the backlog from prior runs
        for rec in pending_val:
            validation_q.put(rec)

    # -- Start Stage 2 workers ---------------------------------------------
    s2_threads = []
    if do_stage2:
        for pair_idx in range(len(gpu_pairs)):
            t = threading.Thread(
                target=_stage2_worker,
                args=(pair_idx, synthesis_q, validation_q,
                      pairing_log_path, raw_dir,
                      args.num_inference_steps, args.cfg_scale,
                      do_stage3, locks, pbars),
                daemon=False,
            )
            t.start()
            s2_threads.append(t)
        # Pre-seed synthesis_q with pairings that already have a prompt (group A)
        for pairing in group_a:
            pr = existing_prompts[f"{pairing['bg_id']}_s{pairing['pair_index']}"]
            synthesis_q.put({
                "pairing": pairing,
                "edit_prompt": pr.get("edit_prompt", ""),
                "negative_prompt": pr.get("negative_prompt", ""),
            })

    # -- Start Stage 1 workers ---------------------------------------------
    s1_threads = []
    if do_stage1:
        bg_task_q: Queue = Queue()
        for entry in pending_bg:
            bg_task_q.put(entry)
        for _ in range(args.vlm_workers):
            bg_task_q.put(_SENTINEL)

        for _ in range(args.vlm_workers):
            t = threading.Thread(
                target=_stage1_worker,
                args=(bg_task_q, synthesis_q, vlm_pool, prompts_path,
                      pairings_by_bg, s1_pending_keys, locks, pbars),
                daemon=False,
            )
            t.start()
            s1_threads.append(t)

    # -- Coordinate termination ---------------------------------------------
    # Wait Stage 1 -> poison Stage 2
    for t in s1_threads:
        t.join()
    pbar_s1.close()

    if do_stage2:
        for _ in s2_threads:
            synthesis_q.put(_SENTINEL)
        for t in s2_threads:
            t.join()
    pbar_s2.close()

    # Wait Stage 2 -> poison Stage 3
    if do_stage3:
        for _ in s3_threads:
            validation_q.put(_SENTINEL)
        for t in s3_threads:
            t.join()
    pbar_s3.close()

    # -- Summary -----------------------------------------------------------
    c = cnt_obj
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete.")
    print(f"  Stage 1  prompts   : {c['s1_done']} processed")
    print(f"  Stage 2  synthesis : {c['s2_ok']}/{c['s2_done']} ok")
    print(f"  Stage 3  validate  : keep={c['s3_keep']}, "
          f"reject={c['s3_done'] - c['s3_keep']}, total={c['s3_done']}")
    print(f"  Output -> {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
