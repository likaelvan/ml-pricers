#!/usr/bin/env python3
"""Compatibility wrapper for the asset-neutral TabM quarter runner."""
from pathlib import Path
import runpy


TABULAR_RUNNER = Path(__file__).resolve().parents[1] / "tabular" / "quarter_train_test.py"


if __name__ == "__main__":
    runpy.run_path(str(TABULAR_RUNNER), run_name="__main__")
