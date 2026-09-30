"""Resolve the local ``code/`` package root for e4_* provenance runners.

Server originals lived at the production-repo root. After the ALLOW pull they
live under ``code/experiments/``; callers must treat ``code/`` as REPO so that
``experiments.*`` and ``dmhi.method.*`` imports resolve.
"""
from __future__ import annotations

from pathlib import Path


def code_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    if here.is_file():
        here = here.parent
    for p in (here, *here.parents):
        if (p / "src" / "dmhi").is_dir() and (p / "experiments").is_dir():
            return p
    raise RuntimeError(
        f"Cannot locate code/ root (need src/dmhi + experiments) from {here}"
    )
