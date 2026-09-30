#!/usr/bin/env python3
"""Audit third-party Python imports against requirements.txt (stdlib only).

Usage:
    python scripts/audit_third_party_imports.py
    python scripts/audit_third_party_imports.py --requirements requirements.txt
    python scripts/audit_third_party_imports.py --warn-only
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {".git", "__pycache__", ".venv", ".venv_fig", "node_modules"}

# import top-level name -> pip distribution name (when they differ)
IMPORT_TO_PIP: dict[str, str] = {
    "yaml": "PyYAML",
    "sklearn": "scikit-learn",
    "mpl_toolkits": "matplotlib",
}

# Local / in-repo modules (not pip packages)
LOCAL_PREFIXES = (
    "src",
    "method",
    "experiments",
    "baselines",
    "common",
    "common_server",
    "lec_data",
    "lec_pipeline",
    "lec_stage1",
    "lec_stage2",
    "lec_stage3",
    "ablation_geom",
    "csdi_physio_dataset",
    "main_model",
    "model",
    "utils",
    "make_sample",
    "temp_script",
    "generate_ai_workspace",
    "update_plan_status",
    "validate_topic_tree",
    "gen_compare_rate_tables",
)

# Documented optional / server-only (not in default requirements)
DEFAULT_ALLOWLIST = frozenset(
    {
        "pypots",
        "h5py",
        "pandas",
        "rich",
        "vitaldb",
        "nlb_tools",
        "breizhcrops",
        "einops",
    }
)


def top_name(module: str) -> str:
    return module.split(".")[0]


def is_stdlib(name: str) -> bool:
    if name in {"__future__"}:
        return True
    if name in sys.stdlib_module_names:
        return True
    return False


def is_local(name: str) -> bool:
    return name in LOCAL_PREFIXES or name.startswith("_")


def parse_requirements(path: Path) -> set[str]:
    packages: set[str] = set()
    if not path.is_file():
        return packages
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-r"):
            continue
        # strip version specifiers
        for sep in ("==", ">=", "<=", "~=", "!=", "<", ">"):
            if sep in line:
                line = line.split(sep, 1)[0].strip()
                break
        if line:
            packages.add(line)
    return packages


def collect_imports(
    roots: list[Path],
    *,
    exclude_scratchpad: bool,
) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            if exclude_scratchpad and "scratchpad" in path.parts:
                continue
            try:
                tree = ast.parse(
                    path.read_text(encoding="utf-8", errors="replace"),
                    filename=str(path),
                )
            except SyntaxError:
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            mods: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mods.add(top_name(alias.name))
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        mods.add(top_name(node.module))
            for m in mods:
                if is_stdlib(m) or is_local(m):
                    continue
                found.setdefault(m, set()).add(rel)
    return found


def import_to_pip(name: str) -> str:
    return IMPORT_TO_PIP.get(name, name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit imports vs requirements.txt")
    parser.add_argument(
        "--requirements",
        type=Path,
        default=REPO_ROOT / "requirements.txt",
        help="Path to requirements file",
    )
    parser.add_argument(
        "--exclude-scratchpad",
        action="store_true",
        help="Skip cursor-agent-team/ai_workspace/scratchpad",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Print gaps but exit 0",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        help="Extra pip package names to allow (repeatable)",
    )
    args = parser.parse_args(argv)

    req_path = args.requirements.resolve()
    declared = parse_requirements(req_path)
    allow = DEFAULT_ALLOWLIST | frozenset(args.allow)

    scan_roots = [
        REPO_ROOT / "code",
        REPO_ROOT / "paper",
        REPO_ROOT / "release_tooling",
        REPO_ROOT / "scripts",
        REPO_ROOT / "cursor-agent-team",
    ]
    imports = collect_imports(scan_roots, exclude_scratchpad=args.exclude_scratchpad)

    gaps: list[tuple[str, str, set[str]]] = []
    for imp, files in sorted(imports.items()):
        pip_name = import_to_pip(imp)
        if pip_name in declared or pip_name in allow:
            continue
        gaps.append((imp, pip_name, files))

    print(f"requirements: {req_path.relative_to(REPO_ROOT)}")
    print(f"declared packages: {len(declared)}")
    print(f"third-party import roots scanned: {len(imports)}")

    if gaps:
        print(f"\nGAPS ({len(gaps)}):")
        for imp, pip_name, files in gaps:
            sample = sorted(files)[0]
            print(f"  {imp} -> pip:{pip_name}  (e.g. {sample})")
        if args.warn_only:
            print("\nAUDIT: WARN (gaps listed)")
            return 0
        print("\nAUDIT: FAIL")
        return 1

    print("\nAUDIT: PASS (no undeclared third-party imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
