#!/usr/bin/env python3
"""시뮬레이터 최종 결과표: results/final/sim_*_n200.json → markdown 표 + 짝 bootstrap CI.

    python3 sim_final_table.py            # 표 출력, results/final/table.json 저장

합류 시간은 실패(충돌·이탈·시간 초과)를 120 s 로 둔 검열값(censored)으로 짝 평균 차이를 낸다.
같은 시드끼리 짝을 짓는다(시드 2000–2199). CI = 짝 차이의 bootstrap 95 %(10 000 회).
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results" / "final"
FAIL_T = 120.0

ROWS = [
    # (파일 이름, 판단, 실행, 표, 게이트, 비고)
    ("baseline", "기존", "rear_abort", "yaml", "유지", "dd1cf54 그대로"),
    ("baseline_gt", "기존", "rear_abort", "GT", "유지", ""),
    ("baseline_gt_nogate", "기존", "rear_abort", "GT", "정리", "2×2 (A)"),
    ("tree_gt_nogate_bdec", "기존", "트리 모니터", "GT", "정리", "2×2 (B)"),
    ("tree_gt_nogate_nomon", "트리", "rear_abort", "GT", "정리", "2×2 (C)"),
    ("tree_gt_nogate", "트리", "트리 모니터", "GT", "정리", "2×2 (D), 제안"),
    ("tree_gt", "트리", "트리 모니터", "GT", "유지", ""),
    ("tree_gt_nogate_full", "트리", "트리 모니터", "GT", "정리", "실행 안 되는 대응 포함"),
    ("tree_gt_nogate_linret", "트리", "트리 모니터", "GT", "정리", "선형 복귀 모델"),
]
PAIRS = [
    ("baseline_gt_nogate", "baseline", "게이트만 정리 (기존 판단)"),
    ("tree_gt_nogate", "baseline", "제안 vs 기존"),
    ("tree_gt_nogate", "baseline_gt", "제안 vs 기존(GT 표)"),
    ("tree_gt_nogate_bdec", "baseline_gt_nogate", "모니터 효과 (기존 판단)"),
    ("tree_gt_nogate", "tree_gt_nogate_nomon", "모니터 효과 (트리 판단)"),
    ("tree_gt_nogate_nomon", "baseline_gt_nogate", "판단 효과 (rear_abort)"),
    ("tree_gt_nogate", "tree_gt_nogate_bdec", "판단 효과 (트리 모니터)"),
    ("tree_gt_nogate_full", "tree_gt_nogate", "실행 안 되는 대응을 허용하면"),
    ("tree_gt_nogate_linret", "tree_gt_nogate", "선형 복귀 모델이면"),
]


def load(name):
    p = RES / f"sim_{name}_n200.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    eps = sorted(d["episodes"], key=lambda r: r["seed"])
    return d["summary"], eps


def censored(eps):
    return np.array([r["t_merge"] if r["success"] else FAIL_T for r in eps])


def boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.choice(diff, size=(n, len(diff)), replace=True).mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    data = {r[0]: load(r[0]) for r in ROWS}
    out = {"rows": [], "pairs": []}
    print("| 방법 | 판단 | 실행 | 확률표 | 게이트 | 성공 | 충돌 | 이탈 | 합류 중앙값 (s) | abort/ep | 차선 끝 정지/ep | 2-D 분리 p05 (m) | 분리 <0.5 m 에피소드 | 비고 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for name, dec, exe, tab, gate, note in ROWS:
        if data[name] is None:
            print(f"| {name} | {dec} | {exe} | {tab} | {gate} | (실행 중) |||||||||")
            continue
        s, eps = data[name]
        row = {"name": name, **s}
        out["rows"].append(row)
        print(f"| {name} | {dec} | {exe} | {tab} | {gate} | {s['success_%']:.1f} % | {s['collision_%']:.1f} % | "
              f"{s['offroad_%']:.1f} % | {s['merge_time_median_s']:.1f} | {s['aborts_per_ep']:.2f} | "
              f"{s['lane_end_stops_per_ep']:.2f} | {s['min_sep_p05_m']:.2f} | {s['eps_sep_below_0.5m']} | {note} |")
    print("\n| 비교 | A − B 합류 시간 (s, 실패 = 120 s) [95 % CI] | 성공률 차이 (%p) | 충돌 차이 (%p) |")
    print("|---|---|---|---|")
    for a, b, label in PAIRS:
        if data[a] is None or data[b] is None:
            continue
        ea, eb = data[a][1], data[b][1]
        assert [r["seed"] for r in ea] == [r["seed"] for r in eb]
        m, lo, hi = boot_ci(censored(ea) - censored(eb))
        ds = 100.0 * (np.mean([r["success"] for r in ea]) - np.mean([r["success"] for r in eb]))
        dc = 100.0 * (np.mean([r["collision"] for r in ea]) - np.mean([r["collision"] for r in eb]))
        out["pairs"].append({"a": a, "b": b, "label": label, "mean": m, "lo": lo, "hi": hi, "d_success": ds,
                             "d_collision": dc})
        print(f"| {label}: {a} − {b} | {m:+.1f} [{lo:+.1f}, {hi:+.1f}] | {ds:+.1f} | {dc:+.1f} |")
    (RES / "table.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
