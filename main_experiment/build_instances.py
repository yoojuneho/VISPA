#!/usr/bin/env python3
"""Build VISPA evaluation instances from CI Seed JSON files.

The output schema is consumed by ``run.py``. Paths are kept relative to
``--image-root`` so the benchmark can be relocated without code changes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


BLUR_LEVELS = ("30", "50", "70", "90")


class MalformedSeedEntry(Exception):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert VISPA CI Seed JSON files into main-experiment instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--direct-seed", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data" / "merged_instances.json")
    parser.add_argument(
        "--direct-prefix",
        default="Image/DirectLeakage/",
        help="Relative prefix prepended to clean/exposed image paths from the direct seed.",
    )
    parser.add_argument("--allow-missing", action="store_true", help="Keep instances even if image files are absent.")
    return parser.parse_args()


def build_instance(raw: dict, source: str, root_prefix: str) -> dict:
    context = raw.get("ci_context") or {}
    attrs = context.get("Attributes") or {}
    for field in ("Text_Privacy", "Permitted_Text_Keys", "Image_Privacy"):
        if field not in attrs:
            raise MalformedSeedEntry(f"missing Attributes.{field}")
    if "User_Instruction" not in context:
        raise MalformedSeedEntry("missing ci_context.User_Instruction")

    image_privacy = attrs["Image_Privacy"]
    clean_by_blur = {"0": root_prefix + raw["background_image"]}
    if isinstance(image_privacy, dict):
        for level in BLUR_LEVELS:
            if level not in image_privacy:
                raise MalformedSeedEntry(f"Image_Privacy[{level}] missing")
            clean_by_blur[level] = root_prefix + image_privacy[level]
    else:
        raise MalformedSeedEntry("Attributes.Image_Privacy must be a dict with blur levels")

    return {
        "instance_id": f"{source}_{raw['instance_id']}",
        "source": source,
        "category": raw["metadata"]["category"],
        "bg_id": raw["metadata"]["bg_id"],
        "face_id": raw["metadata"]["face_id"],
        "image_privacy_label": raw["privacy_status"]["image_privacy"],
        "text_privacy_label": raw["privacy_status"]["text_privacy"],
        "composite": root_prefix + raw["combined_image"],
        "clean_by_blur": clean_by_blur,
        "text_privacy": attrs["Text_Privacy"],
        "permitted_text_keys": attrs["Permitted_Text_Keys"],
        "user_instruction": context["User_Instruction"],
        "instruction_revision": raw.get("instruction_revision"),
    }


def missing_paths(instance: dict, image_root: Path) -> list[str]:
    paths = [instance["composite"], *instance["clean_by_blur"].values()]
    return [path for path in paths if not (image_root / path).exists()]


def load_seed(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Seed JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Seed JSON must contain a list: {path}")
    return payload


def main() -> None:
    args = parse_args()
    instances = []
    malformed = []
    missing = []

    for raw in load_seed(args.direct_seed):
        try:
            instance = build_instance(raw, "direct", args.direct_prefix)
        except MalformedSeedEntry as exc:
            malformed.append((raw.get("instance_id", "->"), str(exc)))
            continue
        absent = missing_paths(instance, args.image_root)
        if absent and not args.allow_missing:
            missing.append((instance["instance_id"], absent))
            continue
        instances.append(instance)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(instances, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"wrote {len(instances)} instances to {args.output}")
    print(f"source counts: {dict(Counter(item['source'] for item in instances))}")
    if malformed:
        print(f"malformed entries skipped: {len(malformed)}")
    if missing:
        print(f"entries skipped for missing image paths: {len(missing)}")


if __name__ == "__main__":
    main()
