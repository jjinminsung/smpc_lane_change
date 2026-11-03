"""자차의 실제 움직임. SMPC 의 예측(선형 횡이동)과 다르게, MORAI 에서 잰 값에 맞춘다.

횡방향: 경로가 바뀌면 측정한 진행률 곡선 p(t) 를 따른다(research/results/day1_m1.json,
  주행 중 시작: t = 1/2/3/4/5 s 에 0.06/0.34/0.56/0.80/0.98, |d| ≤ 0.45 m 는 4.34 s).
  정지에서 시작하면 0.49 s 뒤 출발하고 3.81 s 에 들어온다. 복귀는 같은 곡선을 거리 비율로 줄여 쓴다.
종방향: ACC 목표 속도를 앞먹임 + 1 차 응답으로 쫓는다(가속 ≤ 3.0, 감속 ≤ 6.0, 목표 0 이면 ≤ 9.0 m/s²).
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.interpolate import PchipInterpolator

_T = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.34, 5.0, 5.6]
_P = [0.0, 0.01, 0.06, 0.34, 0.56, 0.80, 0.871, 0.98, 1.0]
PROGRESS = PchipInterpolator(_T, _P)
MOVING_DONE = 5.6
STAND_DELAY, STAND_IN = 0.49, 3.81


@dataclass
class Ego:
    s: float
    y: float
    v: float
    lane: int
    a: float = 0.0
    length: float = 4.635
    width: float = 1.892
    front: float = 3.845
    rear: float = 0.790


class EgoPlant:
    def __init__(self, ego, track, tau_v=0.7, a_max=3.0, a_min=-6.0, a_stop=-9.0, jerk=12.0):
        self.ego, self.track = ego, track
        self.tau_v, self.a_max, self.a_min, self.a_stop, self.jerk = tau_v, a_max, a_min, a_stop, jerk
        self.path_lane = ego.lane
        self.man = None
        self.v_cmd_prev = None

    def set_path(self, now, lane):
        """경로(목표 차선 중심)가 바뀌면 횡 이동을 새로 시작한다(abort 복귀 포함)."""
        if lane == self.path_lane:
            return
        self.path_lane = lane
        y0, y1 = self.ego.y, self.track.center_y(lane)
        frac = min(1.0, abs(y1 - y0) / self.track.lane_width)
        if self.ego.v < 0.2:
            scale, delay = (STAND_IN - STAND_DELAY) / 4.34 * max(0.35, frac), STAND_DELAY
        else:
            scale, delay = max(0.35, math.sqrt(frac)), 0.0
        self.man = {"y0": y0, "y1": y1, "scale": scale, "delay": delay, "tau": 0.0}

    def step(self, now, dt, v_cmd):
        e = self.ego
        # MPC 종방향은 목표 속도 변화율을 앞먹임(throttle_ff)으로 따라간다. 지연 없이 정지선에 선다.
        ff = 0.0 if self.v_cmd_prev is None else (v_cmd - self.v_cmd_prev) / dt
        self.v_cmd_prev = v_cmd
        a_des = (v_cmd - e.v) / self.tau_v + ff
        a_des = min(self.a_max, max(self.a_stop if v_cmd < 0.01 else self.a_min, a_des))
        e.a += max(-self.jerk * dt, min(self.jerk * dt, a_des - e.a))
        if e.v <= 0.0 and e.a < 0.0:
            e.a = 0.0
        v_new = max(0.0, e.v + e.a * dt)
        e.s += 0.5 * (e.v + v_new) * dt
        e.v = v_new
        if self.man is not None:
            m = self.man
            if m["delay"] > 0.0:
                m["delay"] -= dt
            elif e.v > 0.05:                     # 차가 움직여야 옆으로 간다
                m["tau"] += dt / m["scale"]
                if m["tau"] >= MOVING_DONE:
                    e.y, self.man = m["y1"], None
                else:
                    e.y = m["y0"] + (m["y1"] - m["y0"]) * float(PROGRESS(m["tau"]))
        e.lane = self.track.lane_of_y(e.y)
