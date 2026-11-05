#!/usr/bin/env python3
"""MORAI v11(예측 수정 두 가지 + 모니터 (가)) vs v9(수정 전, 옛 모니터) 짝 비교.

짝: v9 V ↔ v11 P (H 12 시나리오 × 1), v9 W ↔ v11 Q (S1–S4 × 3). 방법: rx ↔ rx_fix, rx_single ↔ rx_single_fix.
같은 시나리오·반복끼리 짝. 실패한 주행의 합류 시간 = FAIL_T.
"""
import json
import statistics as st
from pathlib import Path

import numpy as np

LOG = Path.home() / "github/smpc_lane_change/research/results/morai_runs.jsonl"
FAIL_T = 180.0
PAIRS = [("V", "P"), ("W", "Q")]
METH = [("rx", "rx_fix"), ("rx_single", "rx_single_fix")]


def load():
    R = {}
    for l in LOG.open():
        r = json.loads(l)
        t = r["tag"]
        if not (t.startswith("v9_") or t.startswith("v11_")):
            continue
        R[(t.split("_")[0], r["phase"], r["scenario"], r["method"], r["rep"])] = r   # 같은 태그 재실행은 마지막
    return R


def row(r):
    m = (r or {}).get("metrics") or {}
    if "success" not in m:
        return None
    return dict(succ=m["success"], col=m["collision"] or m["morai_collision"],
                t=m["t_merge"] if m["success"] else FAIL_T, ab=len(m["aborts"]), st=m["lane_end_stops"],
                stop_t=m["stop_time"], gap=m["min_gap_m"], acc=m["accepts"], contacts=m.get("morai_contacts", []))


def main():
    R = load()
    for m9, m11 in METH:
        print(f"\n=== {m9} (v9) vs {m11} (v11) ===")
        allp = []
        for p9, p11 in PAIRS:
            keys = sorted({(k[2], k[4]) for k in R if k[0] == "v9" and k[1] == p9 and k[3] == m9})
            ps = []
            for sc, rep in keys:
                a, b = row(R.get(("v9", p9, sc, m9, rep))), row(R.get(("v11", p11, sc, m11, rep)))
                if a and b:
                    ps.append((sc, rep, a, b))
            allp += ps
            if not ps:
                print(f"  [{p9}↔{p11}] 아직 짝 없음")
                continue
            print(f"  [{p9}↔{p11}] 짝 {len(ps)}")
            for sc, rep, a, b in ps:
                flag = "  <<" if (a["succ"] != b["succ"] or a["col"] != b["col"] or abs(a["t"] - b["t"]) > 5) else ""
                print(f"    {sc.replace('research_', ''):14s} r{rep}: v9 {'OK ' if a['succ'] else 'FAIL'} {a['t']:6.1f}s ab{a['ab']} st{a['st']} "
                      f"gap{a['gap']:5.1f} col{int(a['col'])} | v11 {'OK ' if b['succ'] else 'FAIL'} {b['t']:6.1f}s ab{b['ab']} "
                      f"st{b['st']} gap{b['gap']:5.1f} col{int(b['col'])}{flag}")
        if not allp:
            continue
        A = [a for _, _, a, _ in allp]
        B = [b for _, _, _, b in allp]
        d = np.array([b["t"] - a["t"] for _, _, a, b in allp])
        print(f"  합계 짝 {len(allp)}: 성공 v9 {sum(x['succ'] for x in A)} / v11 {sum(x['succ'] for x in B)}, "
              f"충돌·접촉 v9 {sum(x['col'] for x in A)} / v11 {sum(x['col'] for x in B)}")
        print(f"  합류 시간 중앙값 v9 {st.median(x['t'] for x in A):.1f} / v11 {st.median(x['t'] for x in B):.1f} s, "
              f"짝 차이 중앙값 {np.median(d):+.1f} s, 평균 {d.mean():+.1f} s (v11 빠름 {np.sum(d < -1)} / 느림 {np.sum(d > 1)} / ±1 s {np.sum(abs(d) <= 1)})")
        print(f"  복귀(abort) v9 {sum(x['ab'] for x in A)} / v11 {sum(x['ab'] for x in B)}, 차선 끝 정지 v9 {sum(x['st'] for x in A)} / "
              f"v11 {sum(x['st'] for x in B)}, 최소 간격 p10 v9 {np.percentile([x['gap'] for x in A], 10):.1f} / "
              f"v11 {np.percentile([x['gap'] for x in B], 10):.1f} m")


if __name__ == "__main__":
    main()
