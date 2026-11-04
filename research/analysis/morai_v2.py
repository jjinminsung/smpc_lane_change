#!/usr/bin/env python3
"""plan_v2 단계 7: MORAI 에서 새 구조(rx)와 가지 없음(rx_single)을 기존 SMPC·현재 D 와 비교한다.

    python3 morai_v2.py        → markdown 표 + results/morai_v2.json

H 격자(12 개, 1 회): 기존·A·D = 단계 H(v4_H_*), rx·rx_single = 단계 V. 같은 날 다른 시각.
S1–S4(3 회): 기존·D = 단계 B(v2_B_*, D 는 v3_B_* 우선), rx·rx_single = 단계 W. 세션이 달라 교통이 같지 않다.
rx·rx_single 은 v6_ = 승인–모니터 불일치 수정 뒤(자차의 목표 차선 투영, 선택기가 비운 목표 차선 앞차), v5_ = 수정 전.
짝 = 같은 (시나리오, 반복). 실패 = 180 s. 접촉 분류는 morai_final.py 와 같다.
"""
import json
import statistics

import numpy as np

from morai_final import FAIL_T, LOG, RES, contacts_with_speed, refine_kind

ARMS = {
    "H": {"기존 SMPC": ("v4_H_", "baseline"), "A. 게이트 정리": ("v4_H_", "baseline_nogate"),
          "현재 D": ("v4_H_", "tree_nogate"), "새 구조(수정 전)": ("v5_V_", "rx"), "가지 없음(수정 전)": ("v5_V_", "rx_single"),
          "새 구조": ("v6_V_", "rx"), "새 구조-가지 없음": ("v6_V_", "rx_single")},
    "S": {"기존 SMPC": ("v2_B_", "baseline"), "현재 D": ("v3_B_", "tree_nogate"),
          "새 구조(수정 전)": ("v5_W_", "rx"), "가지 없음(수정 전)": ("v5_W_", "rx_single"),
          "새 구조": ("v6_W_", "rx"), "새 구조-가지 없음": ("v6_W_", "rx_single")},
}


def t_of(r):
    m = r["metrics"]
    return m["t_merge"] if m.get("success") else FAIL_T


def boot(d, n=10000, seed=0):
    d = np.asarray(d, float)
    m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    rows = {}
    for line in LOG.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("metrics") and "error" not in r["metrics"]:
                rows[r["tag"]] = r
    out = {}
    for grid, arms in ARMS.items():
        by = {}
        for name, (prefix, method) in arms.items():
            for r in rows.values():
                if r["tag"].startswith(prefix) and r["method"] == method and \
                        ((grid == "H") == r["scenario"].startswith("research_H_")) and \
                        (grid == "H" or r["scenario"].startswith("research_S")):
                    key = (r["scenario"].replace("research_", ""), r["rep"])
                    by.setdefault(name, {})[key] = r
        if not by:
            continue
        for r in [r for d in by.values() for r in d.values()]:
            r["contacts"] = contacts_with_speed(r)
            for c in r["contacts"]:
                c["kind"] = refine_kind(r, c)
        print(f"\n### {'H 격자' if grid == 'H' else 'S1–S4'}\n")
        print("| 방법 | n | 성공 | 합류 시간 중앙값 (s) | 복귀/주행 | 차선 끝 정지/주행 | 접촉: 자차 차선 변경 중 / 그 밖 |")
        print("|---|---|---|---|---|---|---|")
        summ = []
        for name, d in by.items():
            rs = list(d.values())
            ok = [r["metrics"]["t_merge"] for r in rs if r["metrics"].get("success")]
            lc = sum(c["kind"] == "during_ego_lane_change" for r in rs for c in r["contacts"])
            oth = sum(c["kind"] != "during_ego_lane_change" for r in rs for c in r["contacts"])
            s = {"arm": name, "n": len(rs), "success": len(ok), "t_merge_median": statistics.median(ok) if ok else None,
                 "aborts_per_run": statistics.mean([len(r["metrics"]["aborts"]) / 2 for r in rs]),
                 "lane_end_stops_per_run": statistics.mean([r["metrics"]["lane_end_stops"] for r in rs]),
                 "contacts_lc": lc, "contacts_other": oth}
            summ.append(s)
            tm = f"{s['t_merge_median']:.1f}" if ok else "–"
            print(f"| {name} | {len(rs)} | {len(ok)}/{len(rs)} | {tm} | {s['aborts_per_run']:.2f} | "
                  f"{s['lane_end_stops_per_run']:.2f} | {lc} / {oth} |")
        ref_names = [n for n in by if n != "새 구조"]
        effs = []
        if "새 구조" in by:
            print("\n| 비교 (새 구조 − ·) | 짝 | 합류 시간 차이 평균 [95 % CI] (s) | 새 구조가 빠른 짝 | 성공 차이 |")
            print("|---|---|---|---|---|")
            for nm in ref_names:
                keys = sorted(set(by["새 구조"]) & set(by[nm]))
                if not keys:
                    continue
                d = [t_of(by["새 구조"][k]) - t_of(by[nm][k]) for k in keys]
                m, lo, hi = boot(d)
                ds = sum(bool(by["새 구조"][k]["metrics"].get("success")) - bool(by[nm][k]["metrics"].get("success")) for k in keys)
                effs.append({"vs": nm, "n": len(keys), "mean": m, "lo": lo, "hi": hi, "faster": sum(x < 0 for x in d),
                             "success_diff": ds, "pairs": [{"key": list(k), "diff": x} for k, x in zip(keys, d)]})
                print(f"| − {nm} | {len(keys)} | {m:+.1f} [{lo:+.1f}, {hi:+.1f}] | {sum(x < 0 for x in d)}/{len(keys)} | {ds:+d} |")
        out[grid] = {"summary": summ, "effects": effs,
                     "runs": {nm: {f"{k[0]}_r{k[1]}": {"success": bool(r["metrics"].get("success")), "t_merge": r["metrics"].get("t_merge"),
                                                      "aborts": len(r["metrics"]["aborts"]) // 2, "contacts": [c["kind"] for c in r["contacts"]]}
                                   for k, r in sorted(d.items())} for nm, d in by.items()}}
    (RES / "morai_v2.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
