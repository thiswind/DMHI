#!/usr/bin/env python3
"""Replace {REPO} placeholders in the reply material with the real GitHub URL.

Preconditions enforced before any write:
  1. code repo tag v0.3.0-r1 exists locally
  2. every {REPO} target path (blob/tree) exists in that tag's tree

Usage (repo root of riemannian-imputation-paper):
    python code/scripts/replace_repo_placeholders.py           # check + apply
    python code/scripts/replace_repo_placeholders.py --check   # check only
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MATERIAL = ROOT / "paper/revision/回复素材-胡矿.md"
TAG = "v0.3.0-r1"
URL = "https://github.com/thiswind/DMHI"

PAT = re.compile(r"\{REPO\}/(blob|tree)/" + TAG + r"/([^\s)\]]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    # 1. tag exists
    try:
        subprocess.run(["git", "-C", ROOT / "code", "rev-parse", "--verify",
                        f"refs/tags/{TAG}"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        sys.exit(f"FAIL: tag {TAG} not found in code repo — release first")

    # 2. every path exists in tag tree
    text = MATERIAL.read_text()
    targets = PAT.findall(text)
    if not targets and "{REPO}" in text:
        sys.exit("FAIL: {REPO} present but no recognizable pattern — inspect manually")
    tree = subprocess.run(["git", "-C", ROOT / "code", "ls-tree", "-r", "--name-only",
                           TAG], capture_output=True, text=True, check=True).stdout.split()
    treeset = set(tree)
    missing = [p for _, p in targets if p not in treeset]
    if missing:
        sys.exit("FAIL: paths missing from tag:\n  " + "\n  ".join(sorted(set(missing))))

    n = text.count("{REPO}")
    print(f"check OK: {n} placeholders, {len(set(targets))} unique paths, all in {TAG}")

    if args.check:
        return
    if URL not in text:
        new = text.replace("{REPO}", URL)
        MATERIAL.write_text(new)
        print(f"applied: {n} -> {URL}")
    else:
        print("already applied — no change")


if __name__ == "__main__":
    main()
