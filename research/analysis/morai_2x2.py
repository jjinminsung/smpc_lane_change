#!/usr/bin/env python3
"""MORAI 2×2 (단계 E): S1–S4 × {기존, A, B, C, D} × 3, 같은 세션에서 번갈아 돌린 주행.

    python3 morai_2x2.py [--prefix v4_]     → markdown 표 + 짝비교, results/morai_2x2.json

  baseline        기존 SMPC (게이트 유지, rear_abort)
  baseline_nogate A. 기존 판단 + 게이트 정리 + rear_abort
  bdec            B. 기존 판단 + 게이트 정리 + 트리 모니터
  tree_nomon      C. 트리 판단 + 게이트 정리 + rear_abort
  tree_nogate     D. 트리 판단 + 게이트 정리 + 트리 모니터 (제안)
짝비교는 같은 (시나리오, 반복) 끼리, 실패 = 180 s. 접촉 분류는 morai_final.py 와 같다.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np

from morai_final import FAIL_T, LOG, RES, contacts_with_speed, refine_kind

METHODS = ["baseline", "baseline_nogate", "bdec", "tree_nomon", "tree_nogate"]
LABEL = {"baseline": "기존 SMPC", "baseline_nogate": "A. 기존 판단 + rear_abort", "bdec": "B. 기존 판단 + 모니터",
         "tree_nomon": "C. 트리 판단 + rear_abort", "tree_nogate": "D. 제안 (트리 + 모니터)"}
EFFECTS = [("게이트 정리 (A − 기존)", "baseline_nogate", "baseline"),
           ("모니터 효과, 기존 판단 (B − A)", "bdec", "baseline_nogate"),
           ("모니터 효과, 트리 판단 (D − C)", "tree_nogate", "tree_nomon"),
           ("판단 효과, rear_abort (C − A)", "tree_nomon", "baseline_nogate"),
           ("판단 효과, 모니터 (D − B)", "tree_nogate", "bdec"),
           ("전체 (D − 기존)", "tree_nogate", "baseline")]


def t_of(r):
    x = r["metrics"]
    return x["t_merge"] if x.get("success") else FAIL_T


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="v4_")
    a = ap.parse_args()
    rows = {}
    for line in LOG.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r["tag"].startswith(f"{a.prefix}E_") and r.get("metrics") and "error" not in r["metrics"]:
                rows[r["tag"]] = r
    for r in rows.values():
        r["contacts"] = contacts_with_speed(r)
        for c in r["contacts"]:
            c["kind"] = refine_kind(r, c)
    by = {(r["scenario"].replace("research_", ""), r["method"], r["rep"]): r for r in rows.values()}

    print("| 방법 | n | 성공 | 합류 시간 중앙값 (s) | 복귀/주행 | 차선 끝 정지/주행 | 접촉: 자차 차선 변경 중 / 그 밖 |")
    print("|---|---|---|---|---|---|---|")
    summary = []
    for m in METHODS:
        rs = [r for (s, mm, k), r in by.items() if mm == m]
        if not rs:
            continue
        ok = [r["metrics"]["t_merge"] for r in rs if r["metrics"].get("success")]
        k = collections.Counter(c["kind"] for r in rs for c in r["contacts"])
        row = {"method": m, "label": LABEL[m], "n": len(rs), "success": len(ok),
               "t_merge_median": statistics.median(ok) if ok else None,
               "aborts_per_run": statistics.mean([len(r["metrics"]["aborts"]) / 2 for r in rs]),
               "lane_end_stops_per_run": statistics.mean([r["metrics"]["lane_end_stops"] for r in rs]),
               "contacts_lc": k["during_ego_lane_change"],
               "contacts_other": sum(v for kk, v in k.items() if kk != "during_ego_lane_change"),
               "contacts": dict(k)}
        summary.append(row)
        tm = f"{row['t_merge_median']:.1f}" if ok else "–"
        print(f"| {LABEL[m]} | {len(rs)} | {len(ok)}/{len(rs)} | {tm} | {row['aborts_per_run']:.2f} | "
              f"{row['lane_end_stops_per_run']:.2f} | {row['contacts_lc']} / {row['contacts_other']} |")

    print("\n| 비교 | 짝 | 합류 시간 차이 평균 [95 % CI] (s) | 중앙값 (s) | 앞쪽이 빠른 짝 | 성공 차이 |")
    print("|---|---|---|---|---|---|")
    effects = []
    for name, x, y in EFFECTS:
        keys = sorted({(s, k) for (s, mm, k) in by if mm == x} & {(s, k) for (s, mm, k) in by if mm == y})
        if not keys:
            continue
        d = [t_of(by[(s, x, k)]) - t_of(by[(s, y, k)]) for s, k in keys]
        mean, lo, hi = boot(d)
        ds = sum(bool(by[(s, x, k)]["metrics"].get("success")) - bool(by[(s, y, k)]["metrics"].get("success"))
                 for s, k in keys)
        e = {"name": name, "a": x, "b": y, "n": len(keys), "mean": mean, "lo": lo, "hi": hi,
             "median": statistics.median(d), "faster": sum(v < 0 for v in d), "success_diff": ds,
             "pairs": [{"scenario": s, "rep": k, "diff": v} for (s, k), v in zip(keys, d)]}
        effects.append(e)
        print(f"| {name} | {len(keys)} | {mean:+.1f} [{lo:+.1f}, {hi:+.1f}] | {e['median']:+.1f} | "
              f"{e['faster']}/{len(keys)} | {ds:+d} |")

    per_run = [{"scenario": s, "method": m, "rep": k, "success": bool(r["metrics"].get("success")),
                "t_merge": r["metrics"].get("t_merge"), "aborts": len(r["metrics"]["aborts"]) // 2,
                "lane_end_stops": r["metrics"]["lane_end_stops"], "stop_time": r["metrics"]["stop_time"],
                "contacts": [c["kind"] for c in r["contacts"]]} for (s, m, k), r in sorted(by.items())]
    (RES / "morai_2x2.json").write_text(json.dumps({"summary": summary, "effects": effects, "runs": per_run},
                                                   indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
