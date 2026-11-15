"""Ensure tests import the repo-local src/ package (not a stale installed
copy, e.g. the reviewer-run snapshot installed elsewhere)."""

import pathlib
import sys

SRC = str(pathlib.Path(__file__).resolve().parents[1] / "src")
if SRC in sys.path:
    sys.path.remove(SRC)
sys.path.insert(0, SRC)
