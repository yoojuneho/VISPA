#!/usr/bin/env python3
"""Summarize VISPA privacy-risk JSONL outputs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create CSV summaries from privacy_risk_results.jsonl.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("results_jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("model", ""))].append(row)

    summary: list[dict[str, Any]] = []
    for model, model_rows in sorted(grouped.items()):
        parsed = [
            row
            for row in model_rows
            if row.get("parse_ok") is True and isinstance(row.get("risk_score"), int)
        ]
        count = len(parsed)
        mean_score = sum(row["risk_score"] for row in parsed) / count if count else None
        summary.append(
            {
                "model": model,
                "total": len(model_rows),
                "parsed": count,
                "mean_risk_score": round(mean_score, 4) if mean_score is not None else "",
            }
        )
    return summary


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_jsonl.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.results_jsonl)
    summary = summarize(rows)

    out_path = output_dir / "privacy_risk_summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "total",
                "parsed",
                "mean_risk_score",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
