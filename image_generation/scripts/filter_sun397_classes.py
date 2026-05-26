'''
Usage:
  python scripts/filter_sun397_classes.py
  python scripts/filter_sun397_classes.py --input-dir ./outputs/sun397_vlm_filter/included_images --exception-class-path ./config/exception_class.txt --output-dir ./outputs/class_filtered --restart

Description:
  - Read kept SUN397 images from split/alphabet/class_path structure.
  - Exclude every image whose class path matches exception_class.txt.
  - Copy remaining images into one flat output directory.
  - Rename each copied image as 0000_classname.ext while preserving the original extension.
  - Write summary.json and filename_mapping.json into the output directory.
'''

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_INPUT_DIR = "./outputs/sun397_vlm_filter/included_images"
DEFAULT_EXCEPTION_CLASS_PATH = "./config/exception_class.txt"
DEFAULT_OUTPUT_DIR = "./outputs/class_filtered"


@dataclass
class ImageRecord:
    source_image_path: str
    split: str
    class_path: str
    class_leaf_name: str
    source_file_name: str
    source_extension: str


@dataclass
class ClassDirectoryRecord:
    split: str
    class_path: str
    class_leaf_name: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SUN397 keep         .",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, default=DEFAULT_INPUT_DIR, help=" keep   ")
    parser.add_argument(
        "--exception-class-path",
        type=str,
        default=DEFAULT_EXCEPTION_CLASS_PATH,
        help="Class names to exclude, one semantic SUN397 class path per line.",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="   ")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="   output-dir     ",
    )
    return parser


def normalize_config_class_name(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    if ":" in normalized:
        normalized = normalized.split(":", 1)[1].strip()
    normalized = normalized.strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return "/".join(parts)


def load_exception_classes(exception_class_path: Path) -> list[str]:
    if not exception_class_path.is_file():
        raise FileNotFoundError(f"   : {exception_class_path}")

    classes: list[str] = []
    with exception_class_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            normalized = normalize_config_class_name(line)
            if normalized:
                classes.append(normalized)
    return classes


def iter_split_dirs(input_dir: Path) -> list[Path]:
    split_dirs = sorted(path for path in input_dir.iterdir() if path.is_dir())
    if not split_dirs:
        raise FileNotFoundError(f"   split   : {input_dir}")
    return split_dirs


def collect_class_directories(input_dir: Path) -> tuple[list[ClassDirectoryRecord], dict[str, set[str]], set[str], set[str]]:
    class_directory_records: list[ClassDirectoryRecord] = []
    class_paths_by_leaf: dict[str, set[str]] = {}
    discovered_splits: set[str] = set()
    discovered_class_paths: set[str] = set()

    for split_dir in iter_split_dirs(input_dir):
        split_name = split_dir.name
        discovered_splits.add(split_name)

        for alphabet_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for class_dir in sorted(path for path in alphabet_dir.rglob("*") if path.is_dir()):
                class_parts = class_dir.relative_to(alphabet_dir).parts
                if not class_parts:
                    continue

                class_path = "/".join(class_parts)
                class_leaf_name = class_parts[-1]
                discovered_class_paths.add(class_path)
                class_paths_by_leaf.setdefault(class_leaf_name, set()).add(class_path)
                class_directory_records.append(
                    ClassDirectoryRecord(
                        split=split_name,
                        class_path=class_path,
                        class_leaf_name=class_leaf_name,
                    )
                )

    if not class_directory_records:
        raise FileNotFoundError(f"     : {input_dir}")

    return class_directory_records, class_paths_by_leaf, discovered_splits, discovered_class_paths


def collect_image_records(input_dir: Path) -> list[ImageRecord]:
    image_records: list[ImageRecord] = []

    for split_dir in iter_split_dirs(input_dir):
        split_name = split_dir.name

        for image_path in sorted(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            relative_parts = image_path.relative_to(split_dir).parts
            if len(relative_parts) < 3:
                raise ValueError(
                    "    split/alphabet/class_path/file  : "
                    f"{image_path}"
                )

            class_parts = relative_parts[1:-1]
            class_path = "/".join(class_parts)
            class_leaf_name = class_parts[-1]

            image_records.append(
                ImageRecord(
                    source_image_path=str(image_path),
                    split=split_name,
                    class_path=class_path,
                    class_leaf_name=class_leaf_name,
                    source_file_name=image_path.name,
                    source_extension=image_path.suffix.lower(),
                )
            )

    if not image_records:
        raise FileNotFoundError(f"     : {input_dir}")

    return image_records


def validate_exception_classes(exception_classes: list[str], discovered_class_paths: set[str]) -> set[str]:
    return {
        class_name
        for class_name in exception_classes
        if any(class_path == class_name or class_path.startswith(f"{class_name}/") for class_path in discovered_class_paths)
    }


def should_exclude_image(record: ImageRecord, excluded_leaf_names: set[str]) -> bool:
    return any(record.class_path == item or record.class_path.startswith(f"{item}/") for item in excluded_leaf_names)


def prepare_output_dir(output_dir: Path, restart: bool) -> None:
    if output_dir.exists() and restart:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def sanitize_class_name(class_name: str) -> str:
    sanitized = class_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
    sanitized = "_".join(part for part in sanitized.split("_") if part)
    if not sanitized:
        raise ValueError(f"  : {class_name}")
    return sanitized


def copy_filtered_images(image_records: list[ImageRecord], excluded_leaf_names: set[str], output_dir: Path) -> list[dict]:
    kept_records = [record for record in image_records if not should_exclude_image(record, excluded_leaf_names)]
    kept_records.sort(key=lambda record: (record.class_leaf_name, record.class_path, record.source_image_path))

    mapping_records: list[dict] = []
    for index, record in enumerate(kept_records):
        sanitized_class_name = sanitize_class_name(record.class_leaf_name)
        file_name = f"{index:04d}_{sanitized_class_name}{record.source_extension}"
        destination_path = output_dir / file_name
        shutil.copy2(record.source_image_path, destination_path)
        mapping_records.append(
            {
                "index": index,
                "renamed_file_name": file_name,
                "output_image_path": str(destination_path),
                "source_image_path": record.source_image_path,
                "split": record.split,
                "class_path": record.class_path,
                "class_leaf_name": record.class_leaf_name,
                "source_file_name": record.source_file_name,
                "source_extension": record.source_extension,
            }
        )

    return mapping_records


def build_summary(
    image_records: list[ImageRecord],
    excluded_leaf_names: set[str],
    discovered_splits: set[str],
    discovered_class_paths: set[str],
    mapping_records: list[dict],
) -> dict:
    excluded_image_count = sum(1 for record in image_records if should_exclude_image(record, excluded_leaf_names))
    excluded_class_paths = sorted(
        class_path
        for class_path in discovered_class_paths
        if any(class_path == item or class_path.startswith(f"{item}/") for item in excluded_leaf_names)
    )

    return {
        "input_split_count": len(discovered_splits),
        "input_class_count": len(discovered_class_paths),
        "input_image_count": len(image_records),
        "exception_class_count": len(excluded_leaf_names),
        "matched_exception_class_count": len(excluded_leaf_names),
        "excluded_class_count": len(excluded_class_paths),
        "excluded_image_count": excluded_image_count,
        "output_image_count": len(mapping_records),
        "renamed_image_count": len(mapping_records),
        "excluded_class_names": sorted(excluded_leaf_names),
        "excluded_class_paths": excluded_class_paths,
    }


def write_json(output_path: Path, payload) -> None:
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir).resolve()
    exception_class_path = Path(args.exception_class_path).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"  : {input_dir}")

    exception_classes = load_exception_classes(exception_class_path)
    _, class_paths_by_leaf, discovered_splits, discovered_class_paths = collect_class_directories(input_dir)
    image_records = collect_image_records(input_dir)
    excluded_leaf_names = validate_exception_classes(exception_classes, discovered_class_paths)

    prepare_output_dir(output_dir, args.restart)
    mapping_records = copy_filtered_images(image_records, excluded_leaf_names, output_dir)
    summary_payload = build_summary(
        image_records=image_records,
        excluded_leaf_names=excluded_leaf_names,
        discovered_splits=discovered_splits,
        discovered_class_paths=discovered_class_paths,
        mapping_records=mapping_records,
    )

    write_json(output_dir / "filename_mapping.json", mapping_records)
    write_json(output_dir / "summary.json", summary_payload)

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
