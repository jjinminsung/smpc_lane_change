#!/usr/bin/env python3
"""research/results/morai_runs.jsonl → 시나리오 × 방법 요약표(markdown)와 짝비교.

    python3 morai_summary.py [--phase B] [--since 2026-11-04T01:10]

합류 시간은 자차가 움직이기 시작한 때부터 2→3 차선 변경 완료까지. 실패는 별도로 센다.
접촉은 MORAI /CollisionData 기준이며 'npc_cut_in_after_merge'(합류 완료 뒤 NPC 가 끼어듦)는 따로 표시한다.
"""
import argparse
import collections
import json
import statistics
from pathlib import Path

LOG = Path(__file__).resolve().parents[1] / "results" / "morai_runs.jsonl"


def load(phases, valid_from):
    rows = []
    for line in LOG.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if phases and r.get("phase") not in phases:
            continue
        if valid_from and r["tag"] < valid_from:
            continue
        rows.append(r)
    return rows


def fmt(xs, p=1):
    return f"{statistics.median(xs):.{p}f}" if xs else "–"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", nargs="*", default=[])
    ap.add_argument("--prefix", default="", help="이 문자열로 시작하는 tag 만 (예: 재실험 접두사)")
    a = ap.parse_args()
    rows = [r for r in load(a.phase, None) if r["tag"].startswith(a.prefix)]
    cell = collections.defaultdict(list)
    for r in rows:
        cell[(r["scenario"].replace("research_", ""), r["method"])].append(r)
    methods = sorted({m for _, m in cell})
    print("| 시나리오 | 방법 | n | 성공 | 합류 시간 중앙값 (s) | 차선 끝 정지 | 정지 시간 (s) | abort | 최소 간격 (m) | 접촉 (자차 원인 / 합류 후 끼어듦) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (sc, m) in sorted(cell):
        rs = cell[(sc, m)]
        ms = [r["metrics"] for r in rs if r.get("metrics") and "error" not in r["metrics"]]
        ok = [x for x in ms if x.get("success")]
        tm = [x["t_merge"] for x in ok]
        own = sum(1 for x in ms for c in x.get("morai_contacts", []) if c["kind"] != "npc_cut_in_after_merge")
        cut = sum(1 for x in ms for c in x.get("morai_contacts", []) if c["kind"] == "npc_cut_in_after_merge")
        gaps = [x["min_gap_m"] for x in ms if x.get("min_gap_m") is not None]
        print(f"| {sc} | {m} | {len(rs)} | {len(ok)}/{len(ms)} | {fmt(tm)} | {fmt([x['lane_end_stops'] for x in ms], 1)} | "
              f"{fmt([x['stop_time'] for x in ms])} | {fmt([len(x['aborts']) for x in ms], 1)} | "
              f"{min(gaps):.2f} | {own} / {cut} |" if gaps else f"| {sc} | {m} | {len(rs)} | – |")
    # 짝비교: 같은 (시나리오, 반복) 의 기존 vs 트리
    if {"baseline", "tree_nogate"} <= set(methods):
        diffs = []
        for (sc, m), rs in cell.items():
            if m != "baseline":
                continue
            for r in rs:
                mate = [x for x in cell.get((sc, "tree_nogate"), []) if x["rep"] == r["rep"]]
                if not mate:
                    continue
                a_, b_ = r["metrics"], mate[0]["metrics"]
                ta = a_["t_merge"] if a_.get("success") else 180.0
                tb = b_["t_merge"] if b_.get("success") else 180.0
                diffs.append(tb - ta)
        if diffs:
            print(f"\n짝비교 (트리+게이트 정리 − 기존, 실패=180 s): n={len(diffs)}, 중앙값 {statistics.median(diffs):+.1f} s, "
                  f"평균 {statistics.mean(diffs):+.1f} s, 트리가 빠른 짝 {sum(d < 0 for d in diffs)}/{len(diffs)}")


if __name__ == "__main__":
    main()
