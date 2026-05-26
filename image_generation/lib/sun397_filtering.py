import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class ImageCandidate:
    # VLM       
    image_path: str
    split: str
    class_name: str
    width: int
    height: int
    pixel_count: int


@dataclass
class PrefilterRecord:
    #         
    image_path: str
    split: str
    class_name: str
    width: int | None
    height: int | None
    pixel_count: int | None
    eligible: bool
    reason: str


@dataclass
class PrefilterSummary:
    #      
    requested_split: str
    discovered_splits: list[str]
    included_class_count: int
    scanned_file_count: int
    eligible_count: int
    excluded_small_resolution_count: int
    image_read_error_count: int
    missing_class_directory_count: int


def parse_included_classes(included_class_path: Path, exception_class_path: Path | None = None) -> list[str]:
    entries = read_class_lines(included_class_path)
    excluded = parse_excluded_classes(exception_class_path) if exception_class_path else set()
    normalized_classes: list[str] = []

    for entry in entries:
        normalized = normalize_class_entry(entry)
        if normalized and not is_excluded_class(normalized, excluded):
            normalized_classes.append(normalized)

    return normalized_classes


def parse_excluded_classes(exception_class_path: Path) -> set[str]:
    entries = read_class_lines(exception_class_path)
    return {normalize_semantic_class_name(entry) for entry in entries if normalize_semantic_class_name(entry)}


def read_class_lines(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Class list not found: {path}")
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        normalized = line.strip()
        if normalized and not normalized.startswith("#"):
            lines.append(normalized)
    return lines


def normalize_class_entry(entry: str) -> str:
    # Accept "airplane_cabin", "/a/airplane_cabin", or "1: /a/airplane_cabin".
    match = re.match(r"^\s*\d+\s*:\s*(/.*)$", entry.strip())
    normalized = match.group(1) if match else entry.strip()
    normalized = normalized.replace("\\", "/").strip()
    semantic = normalize_semantic_class_name_from_raw(normalized)

    if semantic:
        return f"/{semantic[0]}/{semantic}".rstrip("/")

    if not normalized.startswith("/"):
        normalized = "/" + normalized.lstrip("/")

    return normalized.rstrip("/")


def normalize_semantic_class_name(entry: str) -> str:
    normalized = normalize_semantic_class_name_from_raw(entry)
    if normalized:
        return normalized
    normalized = entry.strip().replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return "/".join(parts)


def normalize_semantic_class_name_from_raw(entry: str) -> str:
    normalized = re.sub(r"^\s*\d+\s*:\s*", "", entry.strip())
    normalized = normalized.replace("\\", "/").strip().strip("/")
    parts = [part for part in normalized.split("/") if part]
    if len(parts) > 1 and len(parts[0]) == 1:
        parts = parts[1:]
    return "/".join(parts)


def is_excluded_class(class_entry: str, excluded: set[str]) -> bool:
    semantic = normalize_semantic_class_name(class_entry)
    return any(semantic == item or semantic.startswith(f"{item}/") for item in excluded)


def resolve_splits(dataset_dir: Path, requested_split: str) -> list[str]:
    # all    split  .
    if requested_split == "all":
        discovered = sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())
        if not discovered:
            raise FileNotFoundError(f"  split   : {dataset_dir}")
        return discovered

    split_dir = dataset_dir / requested_split
    if not split_dir.is_dir():
        raise FileNotFoundError(f" split  : {split_dir}")
    return [requested_split]


def iter_image_files(class_dir: Path) -> Iterable[Path]:
    #       .
    for path in class_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def inspect_image(image_path: Path) -> tuple[int, int, int]:
    #    width, height, pixel_count .
    with Image.open(image_path) as image:
        width, height = image.size
    return width, height, width * height


def collect_candidates(
    dataset_dir: Path,
    requested_split: str,
    included_classes: list[str],
    min_pixel_count_exclusive: int,
    limit: int | None = None,
    show_progress: bool = False,
) -> tuple[list[ImageCandidate], list[PrefilterRecord], PrefilterSummary]:
    #      VLM    .
    splits = resolve_splits(dataset_dir, requested_split)

    candidates: list[ImageCandidate] = []
    records: list[PrefilterRecord] = []

    scanned_file_count = 0
    excluded_small_resolution_count = 0
    image_read_error_count = 0
    missing_class_directory_count = 0

    class_progress = None
    if show_progress:
        class_progress = tqdm(
            total=len(splits) * len(included_classes),
            desc="  ",
            unit="class",
        )

    try:
        for split_name in splits:
            split_dir = dataset_dir / split_name

            for class_name in included_classes:
                class_dir = split_dir / class_name.lstrip("/")
                if not class_dir.is_dir():
                    missing_class_directory_count += 1
                    if class_progress is not None:
                        class_progress.update(1)
                        class_progress.set_postfix(
                            scanned=scanned_file_count,
                            eligible=len(candidates),
                            small=excluded_small_resolution_count,
                            missing=missing_class_directory_count,
                        )
                    continue

                for image_path in iter_image_files(class_dir):
                    scanned_file_count += 1

                    try:
                        width, height, pixel_count = inspect_image(image_path)
                    except (OSError, UnidentifiedImageError, ValueError):
                        image_read_error_count += 1
                        records.append(
                            PrefilterRecord(
                                image_path=str(image_path),
                                split=split_name,
                                class_name=class_name,
                                width=None,
                                height=None,
                                pixel_count=None,
                                eligible=False,
                                reason="image_read_error",
                            )
                        )
                        continue

                    if pixel_count <= min_pixel_count_exclusive:
                        excluded_small_resolution_count += 1
                        records.append(
                            PrefilterRecord(
                                image_path=str(image_path),
                                split=split_name,
                                class_name=class_name,
                                width=width,
                                height=height,
                                pixel_count=pixel_count,
                                eligible=False,
                                reason="resolution_too_small",
                            )
                        )
                        continue

                    candidate = ImageCandidate(
                        image_path=str(image_path),
                        split=split_name,
                        class_name=class_name,
                        width=width,
                        height=height,
                        pixel_count=pixel_count,
                    )
                    candidates.append(candidate)
                    records.append(
                        PrefilterRecord(
                            image_path=str(image_path),
                            split=split_name,
                            class_name=class_name,
                            width=width,
                            height=height,
                            pixel_count=pixel_count,
                            eligible=True,
                            reason="eligible",
                        )
                    )

                    if limit is not None and len(candidates) >= limit:
                        summary = PrefilterSummary(
                            requested_split=requested_split,
                            discovered_splits=splits,
                            included_class_count=len(included_classes),
                            scanned_file_count=scanned_file_count,
                            eligible_count=len(candidates),
                            excluded_small_resolution_count=excluded_small_resolution_count,
                            image_read_error_count=image_read_error_count,
                            missing_class_directory_count=missing_class_directory_count,
                        )
                        return candidates, records, summary

                if class_progress is not None:
                    class_progress.update(1)
                    class_progress.set_postfix(
                        scanned=scanned_file_count,
                        eligible=len(candidates),
                        small=excluded_small_resolution_count,
                        missing=missing_class_directory_count,
                    )
    finally:
        if class_progress is not None:
            class_progress.close()

    summary = PrefilterSummary(
        requested_split=requested_split,
        discovered_splits=splits,
        included_class_count=len(included_classes),
        scanned_file_count=scanned_file_count,
        eligible_count=len(candidates),
        excluded_small_resolution_count=excluded_small_resolution_count,
        image_read_error_count=image_read_error_count,
        missing_class_directory_count=missing_class_directory_count,
    )
    return candidates, records, summary


def write_jsonl(records: Iterable[dict], output_path: Path) -> None:
    # dict iterable JSONL  .
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dataclass_to_dict(instance) -> dict:
    # dataclass JSON   dict .
    return asdict(instance)
