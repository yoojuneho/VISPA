import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_MAX_PIXELS = 1024 * 768 * 2        # 1,572,864 - VLM input cap
_MAX_OUTPUT_PIXELS = 1152 * 1536    # 1,769,472 - synthesis output cap


def resize_if_needed(img: Image.Image, max_pixels: int = _MAX_PIXELS) -> Image.Image:
    """Halve image dimensions repeatedly until total pixel count <= max_pixels."""
    while img.width * img.height > max_pixels:
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    return img


def compute_output_size(
    bg_img: Image.Image,
    max_pixels: int = _MAX_OUTPUT_PIXELS,
    multiple_of: int = 16,
) -> tuple[int, int]:
    """Return (width, height) preserving bg_img aspect ratio, capped by repeated halving
    to max_pixels, then rounded down to the nearest multiple_of."""
    w, h = bg_img.size
    while w * h > max_pixels:
        w //= 2
        h //= 2
    w = max(multiple_of, (w // multiple_of) * multiple_of)
    h = max(multiple_of, (h // multiple_of) * multiple_of)
    return w, h


def generate_pairings(
    bg_entries: list[dict],
    face_paths: list[Path],
    n_per_bg: int = 2,
    seed: int = 42,
) -> list[dict]:
    """
    For each background entry randomly sample n_per_bg distinct faces.

    bg_entries must each have keys: index, renamed_file_name, output_image_path, class_leaf_name.
    Returns list of pairing dicts ready for logging and synthesis.
    """
    rng = random.Random(seed)
    face_paths_sorted = sorted(face_paths)
    pairings: list[dict] = []
    for bg_entry in bg_entries:
        chosen = rng.sample(face_paths_sorted, min(n_per_bg, len(face_paths_sorted)))
        for pair_idx, face_path in enumerate(chosen):
            bg_index = bg_entry["index"]
            face_stem = Path(face_path).stem        # e.g. "0123"
            syn_seed = (bg_index * 31337 + pair_idx * 7919) % (2 ** 31)  # unique seed per scenario
            bg_id = Path(bg_entry["renamed_file_name"]).stem
            pairings.append(
                {
                    "bg_index": bg_index,
                    "bg_id": bg_id,
                    "bg_path": bg_entry["output_image_path"],
                    "bg_class": bg_entry["class_leaf_name"],
                    "face_id": face_stem,
                    "face_path": str(face_path),
                    "pair_index": pair_idx,
                    "synthesis_seed": syn_seed,
                }
            )
    return pairings


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    """Overwrite output_path with all records as JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, output_path: Path) -> None:
    """Append a single record to a JSONL file (thread-safe via OS append semantics)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file. Returns empty list if file absent."""
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_done_ids(jsonl_path: Path, id_field: str) -> set:
    """Return set of values for id_field found in existing JSONL (for resumability)."""
    if not jsonl_path.exists():
        return set()
    done: set = set()
    for rec in read_jsonl(jsonl_path):
        val = rec.get(id_field)
        if val is not None:
            done.add(val)
    return done


def collect_face_paths(face_dir: Path) -> list[Path]:
    """Return sorted list of image paths under face_dir."""
    return sorted(
        p for p in face_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
