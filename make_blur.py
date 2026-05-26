#!/usr/bin/env python3
"""Create controlled motion-blur variants for VISPA clean images.

The implementation follows the motion-blur procedure described in the paper:
multi-sample spatial shifts are averaged along a slightly randomized camera
shake trajectory. Blur levels scale a base strength and jitter linearly.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from PIL import Image, ImageChops


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_LEVELS = [30, 50, 70, 90]
DEFAULT_BASE_STRENGTH = 100.0
DEFAULT_BASE_JITTER = 20.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply controlled motion blur to one image or to seed JSON files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=None, help="Input image path for single-image mode.")
    parser.add_argument("--output", type=Path, default=None, help="Output path for single-image mode.")
    parser.add_argument("--mode", choices=["linear", "multi"], default="multi")
    parser.add_argument("--strength", type=float, default=DEFAULT_BASE_STRENGTH)
    parser.add_argument("--jitter", type=float, default=DEFAULT_BASE_JITTER)
    parser.add_argument("--angle", type=float, default=0.0, help="Motion direction in degrees.")
    parser.add_argument("--samples", type=int, default=17, help="Number of shifted images to average.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by multi mode.")
    parser.add_argument(
        "--shake-level",
        type=int,
        default=None,
        help="Scale strength and jitter by this percentage for single-image mode.",
    )
    parser.add_argument(
        "--batch-from-seed-jsons",
        action="store_true",
        help="Blur each seed's background_image and update Attributes.Image_Privacy paths.",
    )
    parser.add_argument("--seed-jsons", type=Path, nargs="+", default=[])
    parser.add_argument("--levels", type=int, nargs="+", default=DEFAULT_LEVELS)
    parser.add_argument(
        "--motion-blur-dir",
        type=Path,
        default=Path("MotionBlur"),
        help="Directory, relative to each seed JSON parent unless absolute, for blurred images.",
    )
    return parser.parse_args()


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def shifted_copy(image: Image.Image, dx: float, dy: float) -> Image.Image:
    return ImageChops.offset(image, int(round(dx)), int(round(dy)))


def running_average(images: list[Image.Image]) -> Image.Image:
    if not images:
        raise ValueError("No images to average.")
    averaged = images[0]
    for index, image in enumerate(images[1:], start=2):
        averaged = Image.blend(averaged, image, 1.0 / index)
    return averaged


def build_linear_offsets(strength: float, angle_degrees: float, samples: int) -> list[tuple[float, float]]:
    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if samples == 1 or strength == 0:
        return [(0.0, 0.0)]

    angle = math.radians(angle_degrees)
    fx, fy = math.cos(angle), math.sin(angle)
    half = strength / 2.0
    return [(fx * (-half + i * strength / (samples - 1)), fy * (-half + i * strength / (samples - 1))) for i in range(samples)]


def build_multi_offsets(
    strength: float,
    angle_degrees: float,
    samples: int,
    jitter: float,
    seed: int,
) -> list[tuple[float, float]]:
    if samples < 1:
        raise ValueError("samples must be >= 1.")
    if samples == 1 or strength == 0:
        return [(0.0, 0.0)]

    rng = random.Random(seed)
    angle = math.radians(angle_degrees)
    fx, fy = math.cos(angle), math.sin(angle)
    px, py = -fy, fx
    half = strength / 2.0
    offsets: list[tuple[float, float]] = []

    for index in range(samples):
        progress = index / (samples - 1)
        centered = -1.0 + 2.0 * progress
        eased = math.copysign(abs(centered) ** 0.85, centered)
        local_scale = math.sin(progress * math.pi)
        forward = eased * half
        lateral = rng.gauss(0.0, jitter * local_scale)
        forward_noise = rng.gauss(0.0, max(strength * 0.03, 0.0) * local_scale)
        offsets.append((fx * (forward + forward_noise) + px * lateral, fy * (forward + forward_noise) + py * lateral))

    offsets[0] = (-fx * half, -fy * half)
    offsets[-1] = (fx * half, fy * half)
    return offsets


def apply_motion_blur(
    image: Image.Image,
    *,
    mode: str,
    strength: float,
    angle: float,
    samples: int,
    jitter: float,
    seed: int,
) -> Image.Image:
    offsets = (
        build_linear_offsets(strength, angle, samples)
        if mode == "linear"
        else build_multi_offsets(strength, angle, samples, jitter, seed)
    )
    return running_average([shifted_copy(image, dx, dy) for dx, dy in offsets])


def scaled_parameters(strength: float, jitter: float, level: int | None) -> tuple[float, float]:
    if level is None:
        return strength, jitter
    if level <= 0:
        raise ValueError("shake level must be positive.")
    scale = level / 100.0
    return strength * scale, jitter * scale


def normalize_levels(levels: list[int]) -> list[int]:
    unique = []
    seen = set()
    for level in levels:
        if level <= 0:
            raise ValueError("All blur levels must be positive.")
        if level not in seen:
            seen.add(level)
            unique.append(level)
    if not unique:
        raise ValueError("At least one blur level is required.")
    return unique


def image_privacy_mapping(background_image_rel: str, levels: list[int]) -> dict[str, str]:
    source = Path(background_image_rel)
    return {str(level): (Path("MotionBlur") / str(level) / source).as_posix() for level in levels}


def process_single_image(args: argparse.Namespace) -> Path:
    if args.input is None:
        raise SystemExit("--input is required unless --batch-from-seed-jsons is set.")
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")

    strength, jitter = scaled_parameters(args.strength, args.jitter, args.shake_level)
    image = ensure_rgb(Image.open(args.input))
    blurred = apply_motion_blur(
        image,
        mode=args.mode,
        strength=strength,
        angle=args.angle,
        samples=args.samples,
        jitter=jitter,
        seed=args.seed,
    )

    output = args.output
    if output is None:
        suffix = f"blur{args.shake_level}" if args.shake_level is not None else "blur"
        output = PROJECT_ROOT / "outputs" / f"{args.input.stem}_{suffix}{args.input.suffix}"
    output.parent.mkdir(parents=True, exist_ok=True)
    blurred.save(output)
    return output


def resolve_background_path(seed_json: Path, background_image: str) -> Path:
    candidates = [
        seed_json.parent / background_image,
        seed_json.parent.parent / background_image,
        PROJECT_ROOT / background_image,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"background_image not found for {seed_json}: {background_image}")


def process_seed_json(args: argparse.Namespace, seed_json: Path, levels: list[int]) -> None:
    if not seed_json.is_file():
        raise FileNotFoundError(f"Seed JSON not found: {seed_json}")
    data = json.loads(seed_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Seed JSON must contain a list: {seed_json}")

    blur_root = args.motion_blur_dir if args.motion_blur_dir.is_absolute() else seed_json.parent / args.motion_blur_dir
    cache: dict[tuple[str, int], str] = {}

    for item in data:
        background_rel = item.get("background_image")
        if not isinstance(background_rel, str):
            continue
        source_path = resolve_background_path(seed_json, background_rel)
        source_image = ensure_rgb(Image.open(source_path))

        for level in levels:
            key = (background_rel, level)
            if key not in cache:
                strength, jitter = scaled_parameters(args.strength, args.jitter, level)
                blurred = apply_motion_blur(
                    source_image,
                    mode=args.mode,
                    strength=strength,
                    angle=args.angle,
                    samples=args.samples,
                    jitter=jitter,
                    seed=args.seed,
                )
                rel_output = Path("MotionBlur") / str(level) / Path(background_rel)
                output_path = blur_root.parent / rel_output
                output_path.parent.mkdir(parents=True, exist_ok=True)
                blurred.save(output_path)
                cache[key] = rel_output.as_posix()

        attrs = item.setdefault("ci_context", {}).setdefault("Attributes", {})
        attrs["Image_Privacy"] = image_privacy_mapping(background_rel, levels)

    seed_json.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise ValueError("--samples must be >= 1.")
    if args.strength < 0 or args.jitter < 0:
        raise ValueError("--strength and --jitter must be non-negative.")

    if args.batch_from_seed_jsons:
        levels = normalize_levels(args.levels)
        if not args.seed_jsons:
            raise SystemExit("--seed-jsons is required in batch mode.")
        for seed_json in args.seed_jsons:
            process_seed_json(args, seed_json, levels)
            print(f"updated {seed_json}")
    else:
        output = process_single_image(args)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
