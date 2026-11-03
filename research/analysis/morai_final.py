#!/usr/bin/env python3
"""MORAI 최종 결과표. 최종 바이너리(2026-11-04 02:28) 기준으로 주행을 고른다.

    python3 morai_final.py            # markdown 표 + 짝비교, results/morai_final.json 저장

  기존(baseline): v2_B_*, v3_C_*  (트리 변경은 baseline 경로에 영향이 없다: tree_planner_enabled=false)
  트리+게이트 정리(tree_nogate): v3_B_*, v3_C_*
  트리(게이트 유지, tree): v3_D_*
접촉 분류(MORAI /CollisionData, 자차와 0.3 m 이내):
  during_ego_lane_change      자차 차선 변경 중 (판단 로직 책임)
  ego_lane_keeping            자차가 움직이며 차선 유지 중
  npc_contact_ego_stationary  자차 정지(차선 끝 대기) 중 NPC 가 피하다 접촉
  npc_cut_in_after_merge      합류 완료 뒤 NPC(차선 끝 강제 합류)가 끼어듦
"""
import collections
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parents[0] / "results"
LOG = RES / "morai_runs.jsonl"
BAGS = Path.home() / "bags" / "research"
FAIL_T = 180.0


def pick(r):
    tag, m = r["tag"], r["method"]
    if m == "baseline" and (tag.startswith("v2_B_") or tag.startswith("v3_C_")):
        return True
    if m == "tree_nogate" and (tag.startswith("v3_B_") or tag.startswith("v3_C_")):
        return True
    return m == "tree" and tag.startswith("v3_D_")


def contacts_with_speed(r):
    """예전 기록은 접촉 때 자차 속도가 없다 → 그 bag 만 다시 계산한다."""
    cs = (r.get("metrics") or {}).get("morai_contacts", [])
    if not cs or all("ego_v" in c for c in cs):
        return cs
    out = subprocess.run([sys.executable, str(HERE / "morai_metrics.py"), str(BAGS / f"{r['tag']}.bag")],
                         capture_output=True, text=True, timeout=600)
    try:
        return json.loads(out.stdout.strip().splitlines()[-1])["morai_contacts"]
    except Exception:
        return cs


def main():
    rows = {}
    for line in LOG.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if pick(r) and r.get("metrics") and "error" not in r["metrics"]:
                rows[r["tag"]] = r            # 같은 tag 가 여러 번이면 마지막
    for r in rows.values():
        r["contacts"] = contacts_with_speed(r)
    cell = collections.defaultdict(list)
    for r in rows.values():
        cell[(r["phase"] if r["phase"] != "C" else "C", r["scenario"].replace("research_", ""), r["method"])].append(r)

    def kinds(rs):
        k = collections.Counter(c["kind"] for r in rs for c in r["contacts"])
        return k

    print("| 단계 | 시나리오 | 방법 | n | 성공 | 합류 시간 중앙값 (s) | 차선 끝 정지 | 정지 시간 (s) | abort | 최소 간격 (m) | 접촉: 차선 변경 중 / 주행 중 / 정지 중 / 합류 후 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    table = []
    for key in sorted(cell):
        ph, sc, m = key
        rs = cell[key]
        ms = [r["metrics"] for r in rs]
        ok = [x for x in ms if x.get("success")]
        k = kinds(rs)
        gaps = [x["min_gap_m"] for x in ms if x.get("min_gap_m") is not None]
        row = {"phase": ph, "scenario": sc, "method": m, "n": len(rs), "success": len(ok),
               "t_merge_median": statistics.median([x["t_merge"] for x in ok]) if ok else None,
               "lane_end_stops_median": statistics.median([x["lane_end_stops"] for x in ms]),
               "stop_time_median": statistics.median([x["stop_time"] for x in ms]),
               "aborts_mean": statistics.mean([len(x["aborts"]) / 2 for x in ms]),
               "min_gap": min(gaps) if gaps else None, "contacts": dict(k)}
        table.append(row)
        tm = f"{row['t_merge_median']:.1f}" if ok else "–"
        print(f"| {ph} | {sc} | {m} | {len(rs)} | {len(ok)}/{len(rs)} | {tm} | {row['lane_end_stops_median']:.0f} | "
              f"{row['stop_time_median']:.1f} | {row['aborts_mean']:.1f} | {row['min_gap']:.2f} | "
              f"{k['during_ego_lane_change']} / {k['ego_lane_keeping']} / {k['npc_contact_ego_stationary']} / "
              f"{k['npc_cut_in_after_merge']} |")

    # 짝비교: 같은 (단계, 시나리오, 반복) 의 기존 vs 트리+게이트 정리
    def t_of(r):
        x = r["metrics"]
        return x["t_merge"] if x.get("success") else FAIL_T
    pairs = []
    for (ph, sc, m), rs in cell.items():
        if m != "baseline":
            continue
        for r in rs:
            mate = [x for x in cell.get((ph, sc, "tree_nogate"), []) if x["rep"] == r["rep"]]
            if mate:
                pairs.append({"phase": ph, "scenario": sc, "rep": r["rep"], "baseline": t_of(r), "tree": t_of(mate[0]),
                              "baseline_ok": r["metrics"].get("success"), "tree_ok": mate[0]["metrics"].get("success")})
    summary = {}
    for ph in sorted({p["phase"] for p in pairs}) + ["all"]:
        ps = [p for p in pairs if ph == "all" or p["phase"] == ph]
        if not ps:
            continue
        d = [p["tree"] - p["baseline"] for p in ps]
        summary[ph] = {"n": len(ps), "median_diff_s": statistics.median(d), "mean_diff_s": statistics.mean(d),
                       "tree_faster": sum(x < 0 for x in d), "baseline_success": sum(bool(p["baseline_ok"]) for p in ps),
                       "tree_success": sum(bool(p["tree_ok"]) for p in ps)}
        s = summary[ph]
        print(f"\n짝비교 {ph} (트리+게이트 정리 − 기존, 실패 = {FAIL_T:.0f} s): n={s['n']}, 중앙값 {s['median_diff_s']:+.1f} s, "
              f"평균 {s['mean_diff_s']:+.1f} s, 트리가 빠른 짝 {s['tree_faster']}/{s['n']}, "
              f"성공 기존 {s['baseline_success']} / 트리 {s['tree_success']}")
    by_m = collections.defaultdict(collections.Counter)
    for r in rows.values():
        by_m[r["method"]].update(c["kind"] for c in r["contacts"])
        by_m[r["method"]]["runs"] += 1
    print("\n접촉 합계:", {m: dict(c) for m, c in by_m.items()})
    (RES / "morai_final.json").write_text(json.dumps({"table": table, "pairs": pairs, "paired_summary": summary,
                                                      "contacts_by_method": {m: dict(c) for m, c in by_m.items()}},
                                                     indent=1))


if __name__ == "__main__":
    main()
