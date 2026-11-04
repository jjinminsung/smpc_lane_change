"""C++ rxRollout 이 파이썬 reactive.joint_rollout 과 같은 값을 내는지 비교한다 (plan_v2 단계 6b).

    HARNESS=rx_harness research/cpp_check/build.sh
    python3 -m unittest research/sim/tests/test_reactive_vs_cpp.py -v
격리 마스터(포트 11411). 직선 차선 CSV 라 C++ 의 지도 좌표 변환도 파이썬과 같다: 자차·타겟 차량 프로필과 위험을 함께 비교한다.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from test_risk_vs_cpp import LANE_END, ros_env, write_lanes  # noqa: E402
from smpc_sim.params import CONFIG_DIR, REPO_ROOT, load_params  # noqa: E402
from smpc_sim.reactive import ReactiveParams, joint_rollout, load_acc_params  # noqa: E402
from smpc_sim.rollouts import ACCEL, CHANGE_NOW, HOLD, KEEP, YIELD, ActionSpec, RolloutContext, execution_start  # noqa: E402
from smpc_sim.vehicles import TargetSet, TargetVehicle  # noqa: E402

HARNESS = REPO_ROOT / "research/cpp_check/bin/rx_harness"
NODE = "/smpc_rx_harness"


class _Track:
    """직선 차선(LANE_END) 판 Track: reactive.joint_rollout 이 쓰는 것만."""

    def __init__(self, W):
        self.lane_width = W
        self.csv_end = dict(LANE_END)

    def center_y(self, lane):
        return lane * self.lane_width


def make_scene(rng, P):
    W = P.lane_width_m
    cur, tgt = 1, 2
    stopped = rng.random() < 0.4
    v0 = 0.0 if stopped else rng.uniform(4.0, 22.0)
    s0 = LANE_END[cur] - (rng.uniform(20.0, 30.0) if stopped else rng.uniform(40.0, 300.0))
    mid = rng.random() < 0.3
    d0 = -W if not mid else -rng.uniform(1.9, 3.4)
    vehicles = []
    uid = 10

    def add(role, lane, rel, v):
        nonlocal uid
        L = rng.uniform(4.3, 5.0)
        vehicles.append(TargetVehicle(valid=True, unique_id=uid, role=role, lane_id=lane, x=s0 + rel, y=lane * W,
                                      s=s0 + rel, d=rng.uniform(-0.3, 0.3), delta_s=rel, v_long=v, length=L,
                                      width=rng.uniform(1.8, 2.0), half_s=0.5 * L, front_delta_s=rel + 0.5 * L,
                                      rear_delta_s=rel - 0.5 * L, accel=float(rng.choice([-1.5, 0.0, 0.4, 1.0]))))
        uid += 1

    if rng.random() < 0.7:
        add("current_front", cur, rng.uniform(10.0, 50.0), max(0.0, v0 + rng.uniform(-6, 3)))
    if rng.random() < 0.85:
        add("target_front", tgt, rng.uniform(4.0, 40.0), max(0.0, rng.uniform(1.0, 16.0)))
        if rng.random() < 0.5:
            add("nearby_front", tgt, vehicles[-1].delta_s + rng.uniform(8.0, 25.0), rng.uniform(0.0, 15.0))
    for _ in range(int(rng.integers(0, 3))):
        add("nearby_rear" if vehicles and vehicles[-1].role == "target_rear" else "target_rear", tgt,
            -rng.uniform(8.0, 60.0), rng.uniform(0.0, 18.0))
    t = TargetSet(stamp=0.0, current_lane_id=cur, target_lane_id=tgt)
    t.ego_s_current, t.ego_d_target = s0, d0
    t.distance_to_lane_end = LANE_END[cur] - s0
    for v in vehicles:
        if v.role == "target_front":
            t.target_front = v
        elif v.role == "target_rear" and not t.target_rear.valid:
            t.target_rear = v
        elif v.role == "current_front":
            t.current_front = v
    t.nearby_vehicles = list(vehicles)
    if mid:
        specs = [ActionSpec(CHANGE_NOW, 0.0, 0.0, rng.uniform(1.0, 4.0), True)]
    else:
        specs = [ActionSpec(CHANGE_NOW, 0.0, 0.0, rng.uniform(3.0, 4.5), True), ActionSpec(KEEP),
                 ActionSpec(HOLD, 0.0, 0.8, 4.0, True), ActionSpec(YIELD, -0.6, 1.2, 4.0, True),
                 ActionSpec(ACCEL, 1.0, 0.4, 4.0, True)]
    connector = stopped and rng.random() < 0.8
    return dict(t=t, vehicles=vehicles, specs=specs, v0=v0, s0=s0, d0=d0, mid=mid, connector=connector,
                mission=rng.uniform(15.0, 25.0), join=rng.uniform(15.0, 25.0),
                T_rem=rng.uniform(0.8, 3.5) if mid else -1.0)


BRANCH_SETS = [
    [("yield", "base", None, 1.0)],
    [("yield", "base", None, 0.855), ("yield", "brake", "front", 0.0475), ("yield", "accel", None, 0.0475),
     ("nonyield", "base", "rear", 0.045), ("nonyield", "brake", "front", 0.0025), ("nonyield", "accel", None, 0.0025)],
]


def scene_text(sc, br, P):
    t = sc["t"]
    lines = [f"ego_speed {sc['v0']:.17g}", f"mission {sc['mission']:.17g}", f"yield_ref {sc['v0']:.17g}",
             f"targets {t.current_lane_id} {t.target_lane_id} {t.ego_d_target:.17g} {t.distance_to_lane_end:.17g}",
             f"ego_xy {sc['s0']:.17g} {t.target_lane_id * P.lane_width_m + t.ego_d_target:.17g}",
             f"mode {int(sc['connector'])} {sc['join']:.17g} {sc['T_rem'] if sc['mid'] else -1.0:.17g}",
             f"vehicles {len(sc['vehicles'])}"]
    for v in sc["vehicles"]:
        lines.append(f"{v.unique_id} {v.role} {v.lane_id} {v.s:.17g} {v.d:.17g} {v.delta_s:.17g} {v.v_long:.17g} "
                     f"{v.length:.17g} {v.width:.17g} {v.rear_delta_s:.17g} {v.front_delta_s:.17g} {v.accel:.17g}")
    lines.append(f"specs {len(sc['specs'])}")
    for s in sc["specs"]:
        t_sw = 0.0 if sc["mid"] else (execution_start(s, P) if s.change_left else 1e300)
        lines.append(f"{int(s.change_left)} {t_sw:.17g} {s.duration:.17g} {s.action} {s.prep_accel:.17g}")
    lines.append(f"branches {len(br)}")
    for b in br:
        tb = "now" if sc["mid"] else (b[2] or "-")
        lines.append(f"{b[0]} {b[1]} {tb} {b[3]:.17g}")
    return "\n".join(lines) + "\n"


class ReactiveVsCpp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HARNESS.exists():
            raise unittest.SkipTest("HARNESS=rx_harness research/cpp_check/build.sh 를 먼저 실행")
        cls.P = load_params()
        cls.A = load_acc_params()
        cls.RP = ReactiveParams()
        cls.track = _Track(cls.P.lane_width_m)
        cls.tmp = tempfile.TemporaryDirectory()
        write_lanes(cls.tmp.name, cls.P)
        cls.core = subprocess.Popen(["roscore", "-p", "11411"], env=ros_env(),
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(50):
            if subprocess.run(["rosparam", "list"], env=ros_env(), capture_output=True).returncode == 0:
                break
            time.sleep(0.2)
        subprocess.run(["rosparam", "load", str(CONFIG_DIR / "smpc_decision.yaml"), NODE], env=ros_env(), check=True)
        subprocess.run(["rosparam", "load", str(CONFIG_DIR / "acc_planner.yaml"), NODE + "/acc"], env=ros_env(), check=True)
        subprocess.run(["rosparam", "set", f"{NODE}/waypoint_directory", cls.tmp.name], env=ros_env(), check=True)

    @classmethod
    def tearDownClass(cls):
        cls.core.terminate()
        cls.core.wait()
        cls.tmp.cleanup()

    def run_cpp(self, text):
        path = Path(self.tmp.name) / "scene.txt"
        path.write_text(text)
        out = subprocess.run([str(HARNESS), str(path), f"__name:={NODE[1:]}"], env=ros_env(),
                             capture_output=True, text=True, timeout=30)
        line = [l for l in out.stdout.splitlines() if l.startswith("{")]
        self.assertTrue(line, out.stdout + out.stderr)
        return json.loads(line[-1])

    def test_random_scenes(self):
        rng = np.random.default_rng(20261104)
        worst = {"ego_s": 0.0, "ego_v": 0.0, "ego_d": 0.0, "tv_s": 0.0, "risk": 0.0}
        n = 0
        for i in range(80):
            sc = make_scene(rng, self.P)
            br = BRANCH_SETS[i % 2]
            if sc["mid"]:
                br_py = [(b[0], b[1], "now") for b in br]
            else:
                br_py = [b[:3] for b in br]
            ctx = RolloutContext(v0=sc["v0"], s0=sc["s0"], lane=1, target_lane=2, ego_d_target=sc["d0"], pre_cap=sc["mission"],
                                 mission=sc["mission"], tf_speed=float("nan"), D=4.0, yield_ref_speed=sc["v0"],
                                 preview_mode="connector" if sc["connector"] else "blend", connector_join_arc=sc["join"])
            py = joint_rollout(ctx, sc["specs"], sc["t"], sc["vehicles"], self.P, self.A, self.RP, self.track, br_py,
                               lambda v: v.accel, mid_T_rem=sc["T_rem"] if sc["mid"] else None)
            cpp = self.run_cpp(scene_text(sc, br, self.P))
            B, N, K = len(br), len(sc["specs"]), self.P.horizon_steps
            with self.subTest(scene=i):
                np.testing.assert_array_equal(py["valid"], np.array(cpp["valid"], dtype=bool))
                np.testing.assert_array_equal(py["allowed"].reshape(-1), np.array(cpp["allowed"], dtype=bool))
                cs = np.array(cpp["ego_s"]).reshape(B, 2, N, K)
                for key, arr, ref in (("ego_s", py["x"] - sc["s0"], cs), ("ego_v", py["v"], np.array(cpp["ego_v"]).reshape(B, 2, N, K)),
                                      ("ego_d", py["y"] - 2 * self.P.lane_width_m, np.array(cpp["ego_d"]).reshape(B, 2, N, K)),
                                      ("risk", py["risk"], np.array(cpp["risk"]).reshape(B, 2, N, K))):
                    # 프로필은 40 스텝 동안 부동소수점 순서 차이가 쌓여 속도에서 최대 약 6e-6 m/s 난다(이산 판정은 같다)
                    np.testing.assert_allclose(arr, ref, atol=1e-7 if key == "risk" else 1e-5, err_msg=key)
                    worst[key] = max(worst[key], float(np.max(np.abs(arr - ref))))
            n += 1
        print(f"\n{n} scenes: max diff " + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()))


if __name__ == "__main__":
    unittest.main()
