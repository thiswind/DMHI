#!/bin/bash
# GPU-window watcher for e8/e10: poll every 10 min, auto-start the gap-filler
# on whichever GPU has >= 6 GiB free (pick the freer card). Exits when the
# launcher completes (any rc except the guard-abort 3). Read-only polling;
# never touches other users' processes.
set -u
cd "$(dirname "$0")/.."
LOG=runs/e8_e10.log

echo "[watch] start $(date '+%F %T')" >> "$LOG"
while true; do
    pick=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$pick" | head -1 | tr -d ' ')
    if [ "$free" -ge 6000 ]; then
        echo "[watch] $(date '+%F %T') GPU $pick free ${free} MiB — launching" >> "$LOG"
        bash scripts/run_e8_e10.sh "$pick" >> "$LOG" 2>&1
        rc=$?
        if [ "$rc" -ne 3 ]; then
            echo "[watch] launcher rc=$rc, done $(date '+%F %T')" >> "$LOG"
            break
        fi
        echo "[watch] guard aborted mid-way (rc=3), keep waiting" >> "$LOG"
    else
        echo "[watch] $(date '+%F %T') best GPU $pick free ${free} MiB — waiting" >> "$LOG"
    fi
    sleep 600
done
