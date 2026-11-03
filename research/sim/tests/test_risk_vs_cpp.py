"""파이썬 위험도 이식이 원본 C++ evaluateTrajectory 와 같은 값을 내는지 비교한다.

필요: research/cpp_check/build.sh 로 만든 risk_harness, ROS noetic.
격리 마스터(포트 11411)를 직접 띄우므로 MORAI 쪽 마스터(11311)와 충돌하지 않는다.

    python3 -m unittest research/sim/tests/test_risk_vs_cpp.py -v
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from smpc_sim.params import load_params, CONFIG_DIR, REPO_ROOT  # noqa: E402
from smpc_sim.prediction import StraightLanes, modes_for_vehicle  # noqa: E402
from smpc_sim.risk import EgoTrajectory, evaluate_trajectory  # noqa: E402
from smpc_sim.vehicles import TargetVehicle  # noqa: E402

HARNESS = REPO_ROOT / "research/cpp_check/bin/risk_harness"
MASTER = "http://localhost:11411"
NODE = "/smpc_risk_harness"
LANE_END = {0: 400.0, 1: 600.0, 2: 800.0, 3: 2000.0}


def ros_env():
    env = dict(os.environ)
    env["ROS_MASTER_URI"] = MASTER
    return env


def write_lanes(directory, P):
    for lane, end in LANE_END.items():
        with open(Path(directory) / f"{lane}.csv", "w") as f:
            for i, x in enumerate(np.arange(0.0, end + 1e-9, 0.5)):
                f.write(f"{i},{x:.3f},{lane * P.lane_width_m:.3f},0,0,0\n")


def make_scenario(rng, P):
    """자차 궤적 1개 + TV 3대(current_front, target_front, target_rear)."""
    K, dt = P.horizon_steps, P.prediction_dt_sec
    t = dt * np.arange(1, K + 1)
    v0 = rng.uniform(5.0, 25.0)
    a = rng.choice([0.0, -0.8, 1.0])
    v = np.clip(v0 + a * t, 0.0, None)
    s0 = rng.uniform(150.0, 330.0)
    x = s0 + np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)]) + v0 * dt
    change = rng.random() < 0.7
    y = np.zeros(K)
    if change:
        start, dur = rng.uniform(0.2, 2.0), rng.uniform(3.0, 4.5)
        u = np.clip((t - start) / dur, 0.0, 1.0)
        y = P.lane_width_m * u * u * (3.0 - 2.0 * u)
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    ego = dict(t=t, x=x, y=y, yaw=yaw, v=v, v0=v0, s0=s0)

    vehicles = []
    specs = [("current_front", 0, rng.uniform(8.0, 60.0)),
             ("target_front", 1, rng.uniform(-3.0, 60.0)),
             ("target_rear", 1, -rng.uniform(5.0, 80.0))]
    for uid, (role, lane, rel) in enumerate(specs, start=10):
        if rng.random() < 0.15:
            continue
        s = s0 + rel
        if role == "target_front" and rng.random() < 0.3:
            s = LANE_END[1] - rng.uniform(5.0, 60.0)   # 지평선 안에서 차선 끝을 넘김
        vehicles.append(dict(
            uid=uid, role=role, lane=lane, s=s, d=rng.uniform(-0.4, 0.4), delta_s=s - s0,
            yaw=rng.uniform(-0.03, 0.03), yaw_rate=rng.uniform(-0.05, 0.05),
            v_long=max(0.0, v0 + rng.uniform(-6.0, 6.0)), length=rng.uniform(4.2, 5.2),
            width=rng.uniform(1.8, 2.0), half_s=0.0, front_delta_s=0.0, rear_delta_s=0.0,
            accel_obs=rng.choice([-1.0, 0.0, 0.7, 1.2, 2.0])))
    return ego, vehicles


def scenario_text(ego, vehicles):
    lines = [f"ego_speed {ego['v0']:.17g}", f"ego {len(ego['t'])}"]
    for k in range(len(ego["t"])):
        lines.append(" ".join(f"{ego[c][k]:.17g}" for c in ("t", "x", "y", "yaw", "v")))
    lines.append(f"vehicles {len(vehicles)}")
    for v in vehicles:
        lines.append(" ".join(str(v[c]) if isinstance(v[c], (int, str)) else f"{v[c]:.17g}"
                              for c in ("uid", "role", "lane", "s", "d", "delta_s", "yaw",
                                        "yaw_rate", "v_long", "length", "width", "half_s",
                                        "front_delta_s", "rear_delta_s", "accel_obs")))
    return "\n".join(lines) + "\n"


def python_eval(ego, vehicles, P, lanes):
    traj = EgoTrajectory(t=ego["t"], x=ego["x"][None], y=ego["y"][None],
                         yaw=ego["yaw"][None], v=ego["v"][None], valid=np.array([True]))
    tvs, accel = [], {}
    for v in vehicles:
        tvs.append(TargetVehicle(valid=True, unique_id=v["uid"], role=v["role"],
                                 lane_id=v["lane"], s=v["s"], d=v["d"], delta_s=v["delta_s"],
                                 yaw=v["yaw"], yaw_rate=v["yaw_rate"], v_long=v["v_long"],
                                 length=v["length"], width=v["width"]))
        accel[v["uid"]] = v["accel_obs"]
    return evaluate_trajectory(
        traj, tvs, lambda tv: modes_for_vehicle(tv, accel[tv.unique_id], ego["v0"], P),
        ego["v0"], lanes, P)


def as_array(values):
    return np.array([math.inf if x >= 1e299 else (-math.inf if x <= -1e299 else x) for x in values])


class RiskVsCpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HARNESS.exists():
            raise unittest.SkipTest("research/cpp_check/build.sh 를 먼저 실행")
        cls.P = load_params()
        cls.lanes = StraightLanes(cls.P.lane_width_m, LANE_END, cls.P.tv_lane_projection_extension_before_m)
        cls.tmp = tempfile.TemporaryDirectory()
        write_lanes(cls.tmp.name, cls.P)
        cls.core = subprocess.Popen(["roscore", "-p", "11411"], env=ros_env(),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            if subprocess.run(["rosparam", "list"], env=ros_env(), capture_output=True).returncode == 0:
                break
            time.sleep(0.2)
        subprocess.run(["rosparam", "load", str(CONFIG_DIR / "smpc_decision.yaml"), NODE],
                       env=ros_env(), check=True)
        subprocess.run(["rosparam", "set", f"{NODE}/waypoint_directory", cls.tmp.name],
                       env=ros_env(), check=True)

    @classmethod
    def tearDownClass(cls):
        cls.core.terminate()
        cls.core.wait()
        cls.tmp.cleanup()

    def run_cpp(self, text):
        path = Path(self.tmp.name) / "scenario.txt"
        path.write_text(text)
        out = subprocess.run([str(HARNESS), str(path), f"__name:={NODE[1:]}"], env=ros_env(),
                             capture_output=True, text=True, timeout=30)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        self.assertTrue(line, out.stdout + out.stderr)
        return json.loads(line[-1])

    def test_random_scenarios(self):
        rng = np.random.default_rng(20261103)
        worst = 0.0
        n_trunc = 0
        for i in range(60):
            ego, vehicles = make_scenario(rng, self.P)
            cpp = self.run_cpp(scenario_text(ego, vehicles))
            py = python_eval(ego, vehicles, self.P, self.lanes)
            n_trunc += cpp["detail"] == "target_prediction_truncated_at_lane_end"
            with self.subTest(scenario=i):
                self.assertEqual(cpp["valid"], bool(py.valid[0]))
                np.testing.assert_allclose(py.risk_by_step[0], as_array(cpp["risk_by_step"]), atol=1e-9)
                np.testing.assert_allclose(py.min_clearance_by_step[0],
                                           as_array(cpp["min_clear_by_step"]), atol=1e-9)
                self.assertAlmostEqual(py.peak_step_risk[0], cpp["peak"], places=9)
                self.assertAlmostEqual(py.legacy_risk[0], cpp["legacy"], places=9)
                cpp_traces = {(tr["uid"], tr["mode"]): tr for tr in cpp["traces"]}
                for tr in py.traces:
                    ref = cpp_traces[(tr.unique_id, tr.mode)]
                    self.assertAlmostEqual(tr.probability, ref["p"], places=12)
                    np.testing.assert_allclose(tr.conditional_risk[0], as_array(ref["cond"]), atol=1e-9)
                worst = max(worst, float(np.max(np.abs(py.risk_by_step[0] - as_array(cpp["risk_by_step"])))))
        print(f"\n60 scenarios: max |risk diff| = {worst:.2e}, lane-end truncation cases = {n_trunc}")
        self.assertGreater(n_trunc, 0, "차선 끝 절단 경우가 시험되지 않았다")


if __name__ == "__main__":
    unittest.main()
