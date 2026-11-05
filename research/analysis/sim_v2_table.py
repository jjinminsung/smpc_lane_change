#!/usr/bin/env python3
"""plan_v2 단계 4: 시뮬 비교표. 같은 시드(2000–)의 짝비교.

    python3 sim_v2_table.py [--arms baseline tree_gt_nogate rx rx_single rx_common] [--n 1000] [--suffix _v8]

합류 시간 차이는 실패 = 120 s 로 두고 짝마다 뺀 평균의 부트스트랩 95 % CI. 충돌 분류는 sim_collisions.py.
결과 파일 = results/final/sim_{arm}_n{n}{suffix}.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results" / "final"
FAIL_T = 120.0
LABEL = {"baseline": "기존 SMPC", "tree_gt_nogate": "현재 D", "rx": "새 구조", "rx_single": "새 구조-가지 없음",
         "rx_common": "새 구조-G6–G8 유지", "rx_m5": "새 구조-모니터 5ε"}


def load(arm, n=200, suffix=""):
    d = json.loads((RES / f"sim_{arm}_n{n}{suffix}.json").read_text())
    return {e["seed"]: e for e in d["episodes"]}, d["summary"]


def t_of(e):
    return e["t_merge"] if e["success"] else FAIL_T


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    m = np.random.default_rng(seed).choice(x, (n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["baseline", "tree_gt_nogate", "rx", "rx_single", "rx_common"])
    ap.add_argument("--ref", default="tree_gt_nogate")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    data = {}
    for arm in a.arms:
        try:
            data[arm] = load(arm, a.n, a.suffix)
        except FileNotFoundError:
            print("없음:", arm)
    print("| 방법 | 성공 % | 합류 시간 중앙값 (s) | 충돌 % | 복귀/에피소드 | 승인/에피소드 | 차선 끝 정지/에피소드 | 최소 간격 5 % (m) |")
    print("|---|---|---|---|---|---|---|---|")
    for arm, (eps, s) in data.items():
        print(f"| {LABEL.get(arm, arm)} | {s['success_%']:.1f} | {s['merge_time_median_s']:.1f} | {s['collision_%']:.1f} | "
              f"{s['aborts_per_ep']:.2f} | {s['accepts_per_ep']:.2f} | {s['lane_end_stops_per_ep']:.2f} | "
              f"{s.get('min_sep_p05_m', float('nan')):.2f} |")
    if a.ref in data:
        print(f"\n| 비교 (− {LABEL.get(a.ref, a.ref)}) | 합류 시간 차이 평균 [95 % CI] (s) | 빠른 짝 | 성공 차이 | 충돌 차이 |")
        print("|---|---|---|---|---|")
        ref = data[a.ref][0]
        for arm, (eps, _) in data.items():
            if arm == a.ref:
                continue
            seeds = sorted(set(eps) & set(ref))
            d = [t_of(eps[s]) - t_of(ref[s]) for s in seeds]
            m, lo, hi = boot(d)
            print(f"| {LABEL.get(arm, arm)} | {m:+.1f} [{lo:+.1f}, {hi:+.1f}] | {sum(x < 0 for x in d)}/{len(d)} | "
                  f"{sum(eps[s]['success'] for s in seeds) - sum(ref[s]['success'] for s in seeds):+d} | "
                  f"{sum(eps[s]['collision'] for s in seeds) - sum(ref[s]['collision'] for s in seeds):+d} |")


if __name__ == "__main__":
    main()
