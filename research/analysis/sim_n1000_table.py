#!/usr/bin/env python3
"""시뮬레이터 1,000 시드 최종표(시드 2000–2999, 옛 시뮬 = switch_delay 0, GT 인지): results/final/sim_*_n1000.json.

실패 분류: 충돌 / 이탈(차선 끝을 넘음, t_end < 120 s) / 시간 초과(120 s).
짝 차이: 같은 시드끼리, 실패 = 120 s 검열값, bootstrap 95 % CI(10 000 회). 기준 = 새 구조(rx_mt).
결과: results/final/table_n1000.json
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results" / "final"
FAIL_T = 120.0
REF = "rx_mt"
ARMS = [
    ("baseline", "기존 SMPC (dd1cf54)"),
    ("tree_gt_nogate", "D 트리 + 옛 모니터 + 게이트 정리"),
    ("rx_mt", "새 구조: 반응형 트리, 모니터 (가) 2ε"),
    ("rx_mt_m5", "새 구조, 모니터 (가) 5ε"),
    ("rx_mo", "새 구조, 옛 모니터 규칙 2ε"),
    ("rx_single", "새 구조 - 가지 없음"),
]


def load(name):
    eps = json.load(open(RES / f"sim_{name}_n1000.json"))["episodes"]
    return {e["seed"]: e for e in eps}


def fail_kind(e):
    if e["success"]:
        return "ok"
    if e["collision"]:
        return "collision"
    return "timeout" if e["t_end"] >= FAIL_T - 0.2 else "offroad"


def main():
    rng = np.random.default_rng(0)
    data = {a: load(a) for a, _ in ARMS}
    ref = data[REF]
    out = {}
    print(f"{'방법':34s} {'성공':>7s} {'충돌':>6s} {'이탈':>6s} {'시간초과':>8s} {'합류 중앙값(성공)':>16s} {'차선 끝 정지/ep':>14s}  새 구조 대비 짝 평균 [95 % CI]")
    for a, label in ARMS:
        E = data[a]
        k = [fail_kind(e) for e in E.values()]
        ok = [e["t_merge"] for e in E.values() if e["success"]]
        seeds = sorted(set(E) & set(ref))
        d = np.array([(E[s]["t_merge"] if E[s]["success"] else FAIL_T) - (ref[s]["t_merge"] if ref[s]["success"] else FAIL_T)
                      for s in seeds])
        bs = np.array([rng.choice(d, d.size).mean() for _ in range(10000)])
        row = dict(n=len(E), success_pct=100 * k.count("ok") / len(E), collision_pct=100 * k.count("collision") / len(E),
                   offroad_pct=100 * k.count("offroad") / len(E), timeout_pct=100 * k.count("timeout") / len(E),
                   merge_median_success_s=float(np.median(ok)),
                   lane_end_stops_per_ep=float(np.mean([e["lane_end_stops"] for e in E.values()])),
                   paired_mean_vs_ref_s=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
        out[a] = dict(label=label, **row)
        ci = "—" if a == REF else f"{row['paired_mean_vs_ref_s']:+.1f} s [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]"
        print(f"{label:34s} {row['success_pct']:6.1f}% {row['collision_pct']:5.1f}% {row['offroad_pct']:5.1f}% {row['timeout_pct']:7.1f}%"
              f" {row['merge_median_success_s']:14.1f} s {row['lane_end_stops_per_ep']:14.2f}  {ci}")
    (RES / "table_n1000.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
