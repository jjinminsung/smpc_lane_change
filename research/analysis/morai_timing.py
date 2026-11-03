#!/usr/bin/env python3
"""MORAI 트리 계산 시간: /smpc/tree_debug 의 treeSelect 'ms=' 와 실행 중 재계획(tree_monitor) 메시지 간격.

    python3 morai_timing.py "v3_B_*tree_nogate*.bag"      (~/bags/research 기준 glob)
간격은 같은 차선 변경 안(1 s 미만)만 센다.
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import rosbag

BAGS = Path.home() / "bags" / "research"


def main():
    ms, gaps = [], []
    files = sorted(glob.glob(str(BAGS / (sys.argv[1] if len(sys.argv) > 1 else "v3_B_*tree_nogate*.bag"))))
    for f in files:
        mt = []
        with rosbag.Bag(f) as b:
            for _, m, t in b.read_messages(topics=["/smpc/tree_debug"]):
                if m.data.startswith("tree_monitor"):
                    mt.append(t.to_sec())
                else:
                    x = re.search(r"ms=([0-9.]+)", m.data)
                    if x:
                        ms.append(float(x.group(1)))
        d = np.diff(mt)
        gaps += list(d[d < 1.0])
    ms, gaps = np.array(ms), np.array(gaps)
    print(f"bags={len(files)}  treeSelect ms: n={len(ms)} p50={np.median(ms):.1f} p90={np.percentile(ms, 90):.1f} "
          f"p99={np.percentile(ms, 99):.1f} max={ms.max():.1f}")
    print(f"monitor period s: n={len(gaps)} p50={np.median(gaps):.3f} p90={np.percentile(gaps, 90):.3f} "
          f"p99={np.percentile(gaps, 99):.3f}")


if __name__ == "__main__":
    main()
