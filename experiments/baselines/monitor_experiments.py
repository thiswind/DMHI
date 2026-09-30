#!/usr/bin/env python3
"""
Background monitor for baseline experiments on VIA-02.

Polls .done sentinel files every --interval seconds.
Writes one status line per poll to --report_path.
When all methods complete, writes a final summary and exits.

Run via nohup:
  nohup python experiments/baselines/monitor_experiments.py \\
    --log_dir experiments/logs/baselines \\
    --methods classical csdi saits brits imputeformer timemixer psw_i \\
    --interval 600 \\
    --report_path experiments/logs/baselines/run_report.md \\
    > experiments/logs/baselines/monitor.log 2>&1 &
"""

import argparse
import datetime
import pathlib
import subprocess
import sys
import time


def check_status(log_dir: pathlib.Path, methods: list) -> dict:
    statuses = {}
    for method in methods:
        done_file = log_dir / f"{method}.done"
        log_file = log_dir / f"{method}.log"

        if done_file.exists():
            if log_file.exists():
                try:
                    tail = log_file.read_text(errors="replace").splitlines()
                    last_lines = " ".join(tail[-5:]).lower()
                    if "traceback" in last_lines or "error" in last_lines:
                        statuses[method] = "done_with_errors"
                    else:
                        statuses[method] = "done"
                except Exception:
                    statuses[method] = "done"
            else:
                statuses[method] = "done"
        else:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", f"run_{method}.py"],
                    capture_output=True, text=True,
                )
                if result.returncode == 0:
                    statuses[method] = "running"
                else:
                    if log_file.exists():
                        statuses[method] = "not_started_or_failed"
                    else:
                        statuses[method] = "not_started"
            except Exception:
                statuses[method] = "unknown"
    return statuses


def extract_latest_mae(log_file: pathlib.Path) -> str:
    if not log_file.exists():
        return "N/A"
    try:
        lines = log_file.read_text(errors="replace").splitlines()
        for line in reversed(lines):
            line_lower = line.lower()
            if "mae=" in line_lower or "mae :" in line_lower:
                return line.strip()[-60:]
        return "N/A"
    except Exception:
        return "N/A"


def write_report(report_path: pathlib.Path, statuses: dict,
                 log_dir: pathlib.Path, elapsed_h: float):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        f"# Baseline Experiment Monitor Report",
        f"",
        f"**Updated**: {now}  |  **Elapsed**: {elapsed_h:.1f}h",
        f"",
        f"| Method | Status | Latest MAE line |",
        f"|--------|--------|-----------------|",
    ]
    for method, status in statuses.items():
        mae_line = extract_latest_mae(log_dir / f"{method}.log")
        lines.append(f"| {method} | {status} | {mae_line} |")
    lines.append("")
    report_path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", default="experiments/logs/baselines")
    parser.add_argument("--methods", nargs="+",
                        default=["classical", "csdi", "saits", "brits",
                                 "imputeformer", "timemixer", "psw_i"])
    parser.add_argument("--interval", type=int, default=600,
                        help="Poll interval in seconds")
    parser.add_argument("--report_path",
                        default="experiments/logs/baselines/run_report.md")
    parser.add_argument("--max_wait_h", type=float, default=120.0,
                        help="Maximum hours to wait before giving up")
    args = parser.parse_args()

    log_dir = pathlib.Path(args.log_dir)
    report_path = pathlib.Path(args.report_path)
    log_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print(f"[monitor] Starting. Watching {len(args.methods)} methods every "
          f"{args.interval}s. Report: {report_path}", flush=True)

    while True:
        elapsed_s = time.time() - t_start
        elapsed_h = elapsed_s / 3600

        statuses = check_status(log_dir, args.methods)
        write_report(report_path, statuses, log_dir, elapsed_h)

        done_count = sum(1 for s in statuses.values()
                         if s in ("done", "done_with_errors"))
        total = len(args.methods)

        status_str = " | ".join(f"{m}:{s}" for m, s in statuses.items())
        print(f"[monitor] {datetime.datetime.now().strftime('%H:%M:%S')} "
              f"done={done_count}/{total} | {status_str}", flush=True)

        if done_count == total:
            print(f"[monitor] All {total} methods complete. "
                  f"Elapsed: {elapsed_h:.1f}h. Writing final report.", flush=True)
            write_report(report_path, statuses, log_dir, elapsed_h)

            with open(report_path, "a") as f:
                f.write(f"\n**ALL DONE** at {datetime.datetime.now().isoformat()}\n")
            break

        if elapsed_h >= args.max_wait_h:
            print(f"[monitor] Max wait {args.max_wait_h}h exceeded. Exiting.", flush=True)
            write_report(report_path, statuses, log_dir, elapsed_h)
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
