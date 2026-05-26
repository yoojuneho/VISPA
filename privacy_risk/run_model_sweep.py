#!/usr/bin/env python3
"""Run privacy-risk scoring for every model listed in models/test_model_list.txt."""

from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Run Code/privacy_risk/run_privacy_risk.py for a list of served VLMs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-list", type=Path, default=root / "privacy_risk" / "models" / "test_model_list.txt")
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "privacy_risk_sweep" / dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_models(path: Path) -> list[str]:
    models: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if line and not line.startswith("#"):
                models.append(line)
    return models


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def main() -> int:
    args = parse_args()
    models = read_models(args.model_list)
    script = project_root() / "privacy_risk" / "run_privacy_risk.py"
    if args.dry_run:
        print(f"Would evaluate {len(models)} models on {args.image_dir}")
        for model in models:
            print(model)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for model in models:
        model_output = args.output_dir / safe_name(model)
        command = [
            sys.executable,
            str(script),
            "--image-dir",
            str(args.image_dir.resolve()),
            "--base-url",
            args.base_url,
            "--model",
            model,
            "--output-dir",
            str(model_output),
            "--workers",
            str(args.workers),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--timeout",
            str(args.timeout),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])
        if args.overwrite:
            command.append("--overwrite")
        print("Running " + " ".join(command))
        completed = subprocess.run(command, cwd=project_root())
        if completed.returncode != 0:
            failures.append(model)

    if failures:
        print("Failed models:")
        for model in failures:
            print(f"- {model}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
