#!/bin/bash
# v0.3.0-r1 release rehearsal/executor for the code repo (IoTJ-R1).
#
# Usage:
#   bash scripts/release_v030_r1.sh --dry-run   # rehearsal: build squash commit
#                                           # on temp branch, show summary, delete
#   bash scripts/release_v030_r1.sh --release   # REAL: squash -> move main ->
#                                           # tag v0.3.0-r1 (still NO push; push
#                                           # is a separate explicit step)
#
# Red lines honored: revision/iotj-r1 never pushes; main only moves on
# --release; nothing is deleted except the temp rehearsal branch.
set -euo pipefail

MODE="${1:--dry-run}"
cd "$(dirname "$0")/.."
SRC_BRANCH="revision/iotj-r1"
TMP_BRANCH="tmp/v0.3.0-r1-squash"
TAG="v0.3.0-r1"

echo "== preflight =="
git rev-parse --verify "$SRC_BRANCH" >/dev/null || { echo "missing $SRC_BRANCH"; exit 1; }
[ -z "$(git status --porcelain)" ] || { echo "worktree dirty; commit first"; exit 1; }
CUR="$(git branch --show-current)"
if [ "$CUR" != "$SRC_BRANCH" ]; then
  echo "not on $SRC_BRANCH (on ${CUR:-detached})"; exit 1
fi
SRC_HEAD="$(git rev-parse --short HEAD)"
N_COMMITS="$(git rev-list --count main..HEAD 2>/dev/null || echo '?')"
echo "source: $SRC_BRANCH @ $SRC_HEAD ($N_COMMITS commits ahead of main)"

if git rev-parse --verify "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "tag $TAG already exists -> refuse"; exit 1
fi

echo "== build squash on $TMP_BRANCH =="
git checkout --orphan "$TMP_BRANCH"
git add -A
git commit -m "v0.3.0-r1: revision release

Squashed from $SRC_BRANCH @ $SRC_HEAD ($N_COMMITS commits).
Contents: e7 natural-missingness + e9 incremental + results/revision-r1
archive + e8/e10 gap-filler scripts + verification tooling."
SQ="$(git rev-parse --short HEAD)"
echo "squash commit: $SQ"
git checkout "$SRC_BRANCH"
git branch -D "$TMP_BRANCH"

if [ "$MODE" = "--dry-run" ]; then
  echo "== DRY RUN complete =="
  echo "rehearsal OK; real release would:"
  echo "  1. recreate squash commit, reset main to it, tag $TAG"
  echo "  2. push would remain MANUAL (git push origin main $TAG)"
  exit 0
fi

if [ "$MODE" != "--release" ]; then echo "unknown mode: $MODE"; exit 1; fi

echo "== REAL RELEASE =="
git checkout --orphan "$TMP_BRANCH"
git add -A
git commit -m "v0.3.0-r1: revision release

Squashed from $SRC_BRANCH @ $SRC_HEAD ($N_COMMITS commits).
Contents: e7 natural-missingness + e9 incremental + results/revision-r1
archive + e8/e10 gap-filler scripts + verification tooling."
git branch -f main "$TMP_BRANCH"
git checkout main
git branch -D "$TMP_BRANCH"
git tag -a "$TAG" -m "IoTJ-R1 revision release (official rerun numbers)"
echo "== done: main @ $(git rev-parse --short HEAD), tag $TAG =="
echo "push remains manual: git push origin main $TAG"
