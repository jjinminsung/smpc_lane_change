"""여러 에피소드를 병렬로 돌려 지표를 모은다.

    python3 research/sim/run_batch.py --planner baseline --n 200 --out research/results/sim_baseline.json
"""
import argparse
import json
import math
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smpc_sim.world import World, EpisodeConfig  # noqa: E402


def planner_factory(name):
    if name == "baseline":
        from smpc_sim.baseline import BaselineSmpc
        return BaselineSmpc
    if name == "baseline_nogate":
        from smpc_sim.baseline import BaselineSmpc

        def factory_ng(P, track):
            b = BaselineSmpc(P, track)
            b.keep_gates, b.name = False, "baseline_nogate"
            return b
        return factory_ng
    if name in ("baseline_gt", "baseline_gt_nogate"):
        from smpc_sim.baseline import BaselineSmpc
        from smpc_sim.tree import GT_REAR_TABLE

        def factory(P, track):
            P.rear_accel_probability_by_accel = list(GT_REAR_TABLE)
            P.min_rear_accel_probability = 0.0
            b = BaselineSmpc(P, track)
            b.keep_gates = name == "baseline_gt"
            b.name = name
            return b
        return factory
    if name.startswith("tree"):
        from smpc_sim.tree import make_tree_planner
        return make_tree_planner(name)
    raise ValueError(name)


def run_one(args):
    name, seed, cfg_over = args
    cfg = EpisodeConfig(seed=seed, **cfg_over)
    r = World(planner_factory(name), cfg).run()
    d = asdict(r)
    d.pop("trace")
    return d


def summarize(rows):
    n = len(rows)
    ok = [r for r in rows if r["success"]]
    def pct(x):
        return 100.0 * x / max(1, n)
    merge = np.array([r["t_merge"] for r in ok]) if ok else np.array([math.nan])
    return {
        "n": n,
        "success_%": pct(len(ok)),
        "collision_%": pct(sum(r["collision"] for r in rows)),
        "merge_time_median_s": float(np.median(merge)),
        "merge_time_p90_s": float(np.percentile(merge, 90)),
        "lane_end_stops_per_ep": float(np.mean([r["lane_end_stops"] for r in rows])),
        "stop_time_mean_s": float(np.mean([r["stop_time"] for r in rows])),
        "aborts_per_ep": float(np.mean([r["aborts"] for r in rows])),
        "accepts_per_ep": float(np.mean([r["accepts"] for r in rows])),
        "min_gap_p05_m": float(np.percentile([r["min_gap"] for r in rows], 5)),
        "min_ttc_p05_s": float(np.percentile([min(r["min_ttc"], 99.0) for r in rows], 5)),
        "min_sep_p05_m": float(np.percentile([min(r.get("min_sep", 99.0), 99.0) for r in rows], 5)),
        "eps_sep_below_0.5m": int(sum(r.get("min_sep", 99.0) < 0.5 for r in rows)),
        "eps_side_by_side_below_0.5m": int(sum(r.get("near_lat_ticks", 0) > 0 for r in rows)),
        "offroad_%": pct(sum((not r["success"]) and (not r["collision"]) and r["t_end"] < 119.0 for r in rows)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="baseline")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="")
    ap.add_argument("--cfg", default="{}", help="EpisodeConfig override JSON, 예: '{\"perception\": \"lidar_like\"}'")
    a = ap.parse_args()
    over = json.loads(a.cfg)
    t0 = time.time()
    with mp.Pool(a.workers) as pool:
        rows = pool.map(run_one, [(a.planner, a.seed0 + i, over) for i in range(a.n)])
    s = summarize(rows)
    s["wall_s"] = round(time.time() - t0, 1)
    print(json.dumps(s, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"summary": s, "planner": a.planner, "cfg": over, "episodes": rows}, indent=0))


if __name__ == "__main__":
    main()
