#!/usr/bin/env python3
"""Run M1..M4 and build day1_measurements.{json,md}.

  source /opt/ros/noetic/setup.bash
  python3 research/analysis/run_day1.py                                   # all ~/bags/lc_gt_*.bag
  python3 research/analysis/run_day1.py --bags ~/bags/lc_gt_2026-09-28-*.bag --out /tmp/x
Extra arguments (--bags/--cache/--no-cache/--lanes/--out/--jobs) are passed to every script.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    for s in ("m1_lc_duration.py", "m2_abort_return.py", "m3_npc_longitudinal.py",
              "m4_mode_detection_delay.py"):
        print("=" * 20, s, flush=True)
        subprocess.check_call([sys.executable, os.path.join(HERE, s)] + args)
    out = []
    if "--out" in args:
        out = ["--out", args[args.index("--out") + 1]]
    subprocess.check_call([sys.executable, os.path.join(HERE, "day1_report.py")] + out)


if __name__ == "__main__":
    main()
