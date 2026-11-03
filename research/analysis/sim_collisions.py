#!/usr/bin/env python3
"""시뮬레이터 충돌 에피소드 분류: results/final/sim_<planner>_n200.json 의 충돌 시드를 다시 돌려(결정적)
처음 겹친 순간의 자차·NPC 상태로 나눈다.

    python3 sim_collisions.py tree_gt_nogate baseline_gt_nogate ...   → 표 + results/final/collisions.json

분류
  npc_merge_cut_in   NPC 가 차선 끝 합류 중(merge_to 있음)이고 자차 앞(종 겹침 전)에 들어옴
  npc_merge_alongside NPC 가 합류 중이고 자차와 나란히 겹침(반쯤 걸친 정지 차량 옆을 지나감 등)
  ego_lane_change    자차 차선 변경(또는 복귀) 중, NPC 는 합류 중이 아님
  ego_rear_end       자차 차선 유지 중 앞차를 받음
  npc_rear_end       자차 차선 유지 중 뒤차가 받음
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent / "sim"
RES = HERE.parent / "results" / "final"
sys.path.insert(0, str(SIM))

import smpc_sim.world as W  # noqa: E402
from run_batch import planner_factory  # noqa: E402

STATE = {}
_orig = W.obb_overlap


def _obb(e, n, track):
    hit = _orig(e, n, track)
    if hit and "npc" not in STATE:
        w = STATE["world"]
        STATE["npc"] = {"uid": n.uid, "lane": n.lane, "s": round(n.s, 2), "y": round(n.y, 2), "v": round(n.v, 2),
                        "merging": getattr(n, "merge_to", None) is not None}
        STATE["ego"] = {"s": round(e.s, 2), "y": round(e.y, 2), "v": round(e.v, 2), "lane": e.lane,
                        "lc_active": bool(w.sup.active)}
        STATE["rel"] = "ahead" if n.s - 0.5 * n.length >= e.s + e.front - 0.6 else (
            "behind" if n.s + 0.5 * n.length <= e.s - e.rear + 0.6 else "alongside")
    return hit


W.obb_overlap = _obb


def classify(planner, seed):
    STATE.clear()
    w = W.World(planner_factory(planner), W.EpisodeConfig(seed=seed))
    STATE["world"] = w
    res = w.run()
    n, e, rel = STATE.get("npc"), STATE.get("ego"), STATE.get("rel")
    if not res.collision or n is None:
        return {"seed": seed, "kind": "not_reproduced"}
    if n["merging"]:
        kind = "npc_merge_alongside" if rel == "alongside" else "npc_merge_cut_in"
    elif e["lc_active"]:
        kind = "ego_lane_change"
    else:
        kind = "npc_rear_end" if rel == "behind" else "ego_rear_end"
    return {"seed": seed, "kind": kind, "rel": rel, "npc": n, "ego": e}


def main():
    out = {}
    for planner in sys.argv[1:]:
        p = RES / f"sim_{planner}_n200.json"
        if not p.exists():
            continue
        eps = json.loads(p.read_text())["episodes"]
        seeds = [r["seed"] for r in eps if r["collision"]]
        rows = [classify(planner, s) for s in seeds]
        out[planner] = rows
        kinds = {}
        for r in rows:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(f"{planner}: {len(rows)} collisions {kinds}")
        for r in rows:
            print("   ", r["seed"], r["kind"], r.get("rel"), "npc", r.get("npc"), "ego", r.get("ego"))
    prev = json.loads((RES / "collisions.json").read_text()) if (RES / "collisions.json").exists() else {}
    prev.update(out)
    (RES / "collisions.json").write_text(json.dumps(prev, indent=1))


if __name__ == "__main__":
    main()
