#!/bin/bash
# v0.3.0-r1 release rehearsal/executor for the code repo (IoTJ-R1).
#
# Usage:
#   bash scripts/release_v030_r1.sh --dry-run   # rehearsal: build the squash
#                                           # commit object on top of main,
#                                           # show summary, keep nothing
#   bash scripts/release_v030_r1.sh --release   # REAL: squash tree -> new
#                                           # commit with parent main -> move
#                                           # main -> tag v0.3.0-r1 (still NO
#                                           # push; push is a separate step)
#
# The squash commit is created with `git commit-tree` as a CHILD of the
# current main (not an orphan), so `git push origin main` stays a
# fast-forward and the main history remains connected. The work tree and
# the current branch are never switched.
#
# Red lines honored: revision/iotj-r1 never pushes; main only moves on
# --release; nothing is deleted.
set -euo pipefail

MODE="${1:--dry-run}"
cd "$(dirname "$0")/.."
SRC_BRANCH="revision/iotj-r1"
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
MAIN_HEAD="$(git rev-parse --short main)"
echo "source: $SRC_BRANCH @ $SRC_HEAD ($N_COMMITS commits ahead of main @ $MAIN_HEAD)"

if git rev-parse --verify "refs/tags/$TAG" >/dev/null 2>&1; then
  echo "tag $TAG already exists -> refuse"; exit 1
fi

MSG="v0.3.0-r1: revision release

Squashed from $SRC_BRANCH @ $SRC_HEAD ($N_COMMITS commits).
Contents: training-free surgery (learned SPD metric net, learned decoder
and every torch import removed from src/dmhi; fit() replaced by build())
+ e7 natural-missingness + e9 incremental + results/revision-r1 archive
(e1 metric ablation, regression and checkpoint-rebuild evidence)
+ e8/e10 gap-filler scripts + verification tooling."

echo "== build squash commit on top of main =="
SQ="$(git commit-tree "HEAD^{tree}" -p main -m "$MSG")"
echo "squash commit: $(git rev-parse --short "$SQ")  parent: $MAIN_HEAD  tree: $(git rev-parse --short "$SQ^{tree}")"
[ "$(git rev-parse "$SQ^{tree}")" = "$(git rev-parse "HEAD^{tree}")" ] \
  || { echo "tree mismatch"; exit 1; }

if [ "$MODE" = "--dry-run" ]; then
  echo "== DRY RUN complete =="
  echo "rehearsal OK; real release would:"
  echo "  1. create the same squash commit, move main to it, tag $TAG"
  echo "  2. push would remain MANUAL (git push origin main $TAG, fast-forward)"
  exit 0
fi

if [ "$MODE" != "--release" ]; then echo "unknown mode: $MODE"; exit 1; fi

echo "== REAL RELEASE =="
git branch -f main "$SQ"
git tag -a "$TAG" -m "IoTJ-R1 revision release (training-free code, rebuilt checkpoints)" "$SQ"
echo "== done: main @ $(git rev-parse --short main), tag $TAG =="
echo "push remains manual: git push origin main $TAG"
