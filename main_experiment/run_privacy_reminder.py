#!/usr/bin/env python3
"""Run the VISPA action experiment with the generic privacy reminder prompt."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import prompt  # noqa: E402
import run as base_run  # noqa: E402


base_run.SYSTEM_PROMPT = prompt.SYSTEM_PROMPT_PRIVACY_REMINDER
base_run.DEFAULT_OUTPUT_DIR = HERE / "output" / "privacy_reminder"


if __name__ == "__main__":
    base_run.main()
