#!/usr/bin/env python3
"""plan_v2 단계 7 MORAI 비교표: 기존 SMPC / D(트리 + 옛 모니터 + 게이트 정리) / 새 구조(반응형 트리) / 새 구조-가지 없음.

같은 시나리오 묶음, 다른 시각의 주행(교통이 매번 다름):
  H 격자 12 개 × 1 회 : 기존·D = v4 단계 H, 새 구조 = v11 단계 P (예측 수정 켬, 모니터 (가) 2ε)
  S1–S4 × 3 회       : 기존 = v2 단계 B, D = v3 단계 B, 새 구조 = v11 단계 Q
참고: v9(예측 수정 전, 옛 모니터 규칙) V·W.
실패한 주행의 합류 시간 = 180 s. 접촉 = MORAI 충돌 이벤트(ego_lc_active = 자차 차선 변경 중).
결과: results/morai_step7.json
"""
import json
import statistics as st
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
FAIL_T = 180.0
ARMS = {
    "H": [("기존 SMPC", "v4", "H", "baseline"), ("D 트리 + 옛 모니터", "v4", "H", "tree_nogate"),
          ("새 구조 (반응형 트리)", "v11", "P", "rx_fix"), ("새 구조 - 가지 없음", "v11", "P", "rx_single_fix"),
          ("(참고) v9 새 구조, 수정 전", "v9", "V", "rx"), ("(참고) v9 가지 없음, 수정 전", "v9", "V", "rx_single")],
    "S": [("기존 SMPC", "v2", "B", "baseline"), ("D 트리 + 옛 모니터", "v3", "B", "tree_nogate"),
          ("새 구조 (반응형 트리)", "v11", "Q", "rx_fix"), ("새 구조 - 가지 없음", "v11", "Q", "rx_single_fix"),
          ("(참고) v9 새 구조, 수정 전", "v9", "W", "rx"), ("(참고) v9 가지 없음, 수정 전", "v9", "W", "rx_single")],
}


def load():
    R = {}
    for line in (RES / "morai_runs.jsonl").open():
        r = json.loads(line)
        R[(r["tag"].split("_")[0], r["phase"], r["scenario"], r["method"], r["rep"])] = r   # 재실행은 마지막
    return R


def summarize(runs):
    ms = [r["metrics"] for r in runs if "success" in (r.get("metrics") or {})]
    t = [m["t_merge"] if m["success"] else FAIL_T for m in ms]
    contacts = [c for m in ms for c in (m.get("morai_contacts") or [])]
    slow = [m for r, m in zip(runs, ms) if r["scenario"].endswith(("_d40", "_d60"))]
    return dict(
        n=len(ms), success=sum(m["success"] for m in ms), merge_median_s=round(st.median(t), 1),
        morai_contacts=len(contacts), contacts_during_ego_lc=sum(bool(c.get("ego_lc_active")) for c in contacts),
        contact_kinds=sorted({c.get("kind", "?") for c in contacts}),
        aborts=sum(len(m["aborts"]) for m in ms), lane_end_stops=sum(m["lane_end_stops"] for m in ms),
        slow_target_n=len(slow), slow_target_success=sum(m["success"] for m in slow),
        slow_target_median_s=round(st.median([m["t_merge"] if m["success"] else FAIL_T for m in slow]), 1) if slow else None,
    )


def main():
    R = load()
    out = {}
    for bundle, arms in ARMS.items():
        print(f"\n=== {'H 격자: NPC 생성 {30,50,70} → 목표 {40,60,80,100} km/h × 1' if bundle == 'H' else 'S1–S4 × 3'} ===")
        print(f"{'방법':28s} {'성공':>6s} {'합류 중앙값':>10s} {'접촉(변경 중)':>12s} {'복귀':>5s} {'차선 끝 정지':>10s}"
              + ("  NPC 목표 40·60 km/h 성공 / 중앙값" if bundle == "H" else ""))
        for name, pre, ph, meth in arms:
            runs = [r for k, r in R.items() if k[0] == pre and k[1] == ph and k[3] == meth]
            s = summarize(runs)
            out[f"{bundle}/{name}"] = dict(tag=pre, phase=ph, method=meth, **s)
            extra = f"  {s['slow_target_success']}/{s['slow_target_n']} / {s['slow_target_median_s']} s" if bundle == "H" else ""
            print(f"{name:28s} {s['success']:>3d}/{s['n']:<2d} {s['merge_median_s']:>8.1f} s {s['morai_contacts']:>6d} ({s['contacts_during_ego_lc']})"
                  f" {s['aborts']:>9d} {s['lane_end_stops']:>10d}{extra}")
    (RES / "morai_step7.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
