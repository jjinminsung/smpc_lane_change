"""C++ treeEvaluate/treeBranches 가 파이썬 기준 구현(tree.py)과 같은지 비교한다.

    python3 -m unittest research/sim/tests/test_tree_vs_cpp.py -v
필요: HARNESS=tree_harness research/cpp_check/build.sh
"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import test_risk_vs_cpp as base  # noqa: E402
from test_risk_vs_cpp import make_scenario, LANE_END, ros_env  # noqa: E402
from smpc_sim.params import load_params, REPO_ROOT  # noqa: E402
from smpc_sim.risk import EgoTrajectory  # noqa: E402
from smpc_sim.rollouts import ActionSpec, CHANGE_NOW, KEEP  # noqa: E402
from smpc_sim.track import Track  # noqa: E402
from smpc_sim.tree import TreePlanner  # noqa: E402
from smpc_sim.vehicles import TargetSet, TargetVehicle  # noqa: E402

HARNESS = REPO_ROOT / "research/cpp_check/bin/tree_harness"
NODE = "/smpc_tree_harness"


def roots_for(ego, rng, P):
    """같은 출발 상태에서 차선 변경 루트 3 개 + KEEP 1 개."""
    K, dt = P.horizon_steps, P.prediction_dt_sec
    t = dt * np.arange(1, K + 1)
    out = []
    for start, dur in ((0.6, 4.2), (1.0 + rng.uniform(0, 1), 4.4), (rng.uniform(0.2, 2.0), rng.uniform(3.0, 4.5))):
        u = np.clip((t - start) / dur, 0.0, 1.0)
        out.append((True, rng.uniform(0.5, 3.0), ego["x"], P.lane_width_m * u, ego["v"]))
    out.append((False, 0.0, ego["x"], np.zeros(K), ego["v"]))
    return t, out


class TreeVsCpp(base.RiskVsCpp):
    @classmethod
    def setUpClass(cls):
        if not HARNESS.exists():
            raise unittest.SkipTest("HARNESS=tree_harness research/cpp_check/build.sh")
        super().setUpClass()
        subprocess.run(["rosparam", "load", str(REPO_ROOT / "config/smpc_decision.yaml"), NODE],
                       env=ros_env(), check=True)
        subprocess.run(["rosparam", "set", f"{NODE}/waypoint_directory", cls.tmp.name], env=ros_env(), check=True)

    def run_tree_cpp(self, text):
        path = Path(self.tmp.name) / "tree_scenario.txt"
        path.write_text(text)
        out = subprocess.run([str(HARNESS), str(path), f"__name:={NODE[1:]}"], env=ros_env(),
                             capture_output=True, text=True, timeout=60)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        self.assertTrue(line, out.stdout + out.stderr)
        return json.loads(line[-1])

    def test_random_trees(self):
        rng = np.random.default_rng(42)
        P = load_params()
        track = Track(csv_end=dict(LANE_END), csv_start={k: 0.0 for k in LANE_END})
        mission = 25.0
        n_feas = n_cmp = n_ret = 0
        worst = 0.0
        for i in range(25):
            ego, veh = make_scenario(rng, P)
            t_grid, roots = roots_for(ego, rng, P)
            rear = next((v["uid"] for v in veh if v["role"] == "target_rear"), -1)
            d_t = float(rng.choice([-3.5, -3.0, -2.4, -2.0]))     # 차선 변경 중 여러 진행 단계
            lines = [f"ego_speed {ego['v0']:.17g}", f"mission {mission}", f"targets 0 1 {d_t}", f"rear {rear}",
                     f"ego_xy {ego['s0']:.17g} {P.lane_width_m + d_t:.17g}",
                     f"vehicles {len(veh)}"]
            for v in veh:
                lines.append(" ".join(str(v[c]) if isinstance(v[c], (int, str)) else f"{v[c]:.17g}"
                                      for c in ("uid", "role", "lane", "s", "d", "delta_s", "yaw", "yaw_rate",
                                                "v_long", "length", "width", "half_s", "front_delta_s",
                                                "rear_delta_s", "accel_obs")))
            lines.append(f"roots {len(roots)}")
            for change, cost, x, y, v in roots:
                lines.append(f"root {int(change)} {cost:.17g} {len(t_grid)}")
                lines += [f"{t_grid[k]:.17g} {x[k]:.17g} {y[k]:.17g} 0 {v[k]:.17g}" for k in range(len(t_grid))]
            cpp = self.run_tree_cpp("\n".join(lines) + "\n")

            planner = TreePlanner(load_params(), track)
            planner.mission = mission
            tvs = []
            for v in veh:
                tv = TargetVehicle(valid=True, unique_id=v["uid"], role=v["role"], lane_id=v["lane"], s=v["s"],
                                   d=v["d"], delta_s=v["delta_s"], yaw=v["yaw"], yaw_rate=v["yaw_rate"],
                                   v_long=v["v_long"], length=v["length"], width=v["width"],
                                   x=v["s"], y=v["lane"] * P.lane_width_m + v["d"])
                h = planner.hist.histories.setdefault(tv.unique_id, None)
                from smpc_sim.vehicles import VehicleHistory
                planner.hist.histories[tv.unique_id] = VehicleHistory(initialized=True, v_long=tv.v_long,
                                                                      filtered_accel=v["accel_obs"])
                tvs.append(tv)
            ts = TargetSet(current_lane_id=0, target_lane_id=1, ego_d_target=-3.5)
            ts_mon = TargetSet(current_lane_id=0, target_lane_id=1, ego_d_target=d_t, ego_s_current=ego["s0"])
            for tv in tvs:
                if tv.role in ("target_front", "target_rear", "current_front"):
                    setattr(ts, tv.role, tv)
            R = next((tv for tv in tvs if tv.unique_id == rear), None)
            for tv in tvs:
                if tv.role in ("target_front", "target_rear", "current_front"):
                    setattr(ts_mon, tv.role, tv)
            traj = EgoTrajectory(t=t_grid, x=np.vstack([r[2] for r in roots]), y=np.vstack([r[3] for r in roots]),
                                 yaw=np.zeros((len(roots), len(t_grid))), v=np.vstack([r[4] for r in roots]),
                                 valid=np.ones(len(roots), dtype=bool))
            specs = [ActionSpec(CHANGE_NOW, change_left=True) if r[0] else ActionSpec(KEEP) for r in roots]
            feas, cost, combo, peak, br = planner.evaluate_tree(traj, specs, ts, ego["v0"], R, tvs,
                                                                np.array([r[1] for r in roots]))
            with self.subTest(scenario=i):
                self.assertEqual([b[0] for b in br], [b[0] for b in cpp["branches"]])
                np.testing.assert_allclose([b[1] for b in br], [b[1] for b in cpp["branches"]], atol=1e-12)
                for j, rc in enumerate(cpp["roots"]):
                    n_cmp += 1
                    self.assertEqual(bool(feas[j]), rc["feasible"], f"root {j}")
                    if feas[j]:
                        n_feas += 1
                        self.assertAlmostEqual(cost[j], rc["cost"], places=6)
                    self.assertAlmostEqual(peak[j], rc["peak"], places=6)
                    worst = max(worst, abs(peak[j] - rc["peak"]))
            planner.last_acc_target = None
            mon = planner.monitor(0.0, ts_mon, ego["v0"], mission) == "return"
            with self.subTest(scenario=i, part="monitor"):
                self.assertEqual(mon, cpp["monitor_return"], cpp["monitor_detail"])
            n_ret += mon
        print(f"\nmonitor returns: {n_ret}/25")
        print(f"\n{n_cmp} roots compared, {n_feas} feasible, max |peak diff| = {worst:.2e}")
        self.assertGreater(n_feas, 5)
        self.assertLess(n_feas, n_cmp)

    def test_random_scenarios(self):   # 위험도 비교는 test_risk_vs_cpp 에서 한다
        pass


if __name__ == "__main__":
    unittest.main()
