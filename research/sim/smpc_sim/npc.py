"""시뮬레이터의 주변 차량(NPC). SMPC 의 예측 모델과는 일부러 다르게 만든다.

움직임 = "MORAI 에서 잰 목표 속도 프로파일" + "IDM 차량 추종".
  - 목표 속도 프로파일: 스폰 속도(50 km/h)로 나타나 spawn_accel 로 desired 까지 가속.
    순항 중에는 가속/감속 에피소드가 포아송으로 생긴다(research/results/day1_m3.json).
  - IDM: 앞차(자차 포함)와의 간격으로 실제 가속도를 제한한다. 자차가 끼어들면
    뒤차가 감속한다. SMPC 예측은 이것을 모른다(예측 모델을 자기 자신과 시험하지 않기).
"""
from dataclasses import dataclass, field
import json
import math

import numpy as np

from .params import REPO_ROOT

M3_JSON = REPO_ROOT / "research/results/day1_m3.json"


@dataclass
class NpcModel:
    spawn_speed: float = 13.9            # MORAI SpawnVelocity_Custom 50 km/h
    spawn_accel_mean: float = 1.39
    spawn_accel_std: float = 0.29
    spawn_accel_min: float = 0.68
    spawn_accel_max: float = 2.05
    desired_quantiles: tuple = ((0.1, 9.7), (0.5, 23.6), (0.9, 28.7))
    episode_rate_per_min: float = 3.0    # 가속·감속 각각
    episode_accel_mean: float = 1.07
    episode_decel_mean: float = 1.54
    episode_duration_median: float = 3.6
    # IDM (MORAI NPC SafetyDist 5 m, TTC 1.0 s)
    idm_s0: float = 5.0
    idm_T: float = 1.0
    idm_b: float = 2.5
    idm_delta: float = 4.0
    max_brake: float = 6.0
    length: float = 4.8
    width: float = 1.9

    @classmethod
    def from_measurements(cls, path=M3_JSON):
        m = cls()
        try:
            d = json.loads(path.read_text())["parametric_for_sim"]
        except (OSError, KeyError, ValueError):
            return m
        sa = d.get("spawn_accel_mps2", {})
        m.spawn_accel_mean = sa.get("mean", m.spawn_accel_mean)
        m.spawn_accel_std = sa.get("std", m.spawn_accel_std)
        m.spawn_accel_min = sa.get("min", m.spawn_accel_min)
        m.spawn_accel_max = sa.get("max", m.spawn_accel_max)
        ds = d.get("desired_speed_after_spawn_mps", {})
        if {"p10", "median", "p90"} <= set(ds):
            m.desired_quantiles = ((0.1, ds["p10"]), (0.5, ds["median"]), (0.9, ds["p90"]))
        m.episode_rate_per_min = 0.5 * (d.get("accel_episode_rate_per_npc_min", 3.0)
                                        + d.get("decel_episode_rate_per_npc_min", 3.0))
        m.episode_accel_mean = d.get("accel_mean_mps2", {}).get("median", m.episode_accel_mean)
        m.episode_decel_mean = d.get("decel_mean_mps2", {}).get("median", m.episode_decel_mean)
        m.episode_duration_median = d.get("accel_duration_s", {}).get("median", m.episode_duration_median)
        return m

    def sample_desired(self, rng):
        q = np.array([p for p, _ in self.desired_quantiles])
        v = np.array([x for _, x in self.desired_quantiles])
        if np.allclose(v, v[0]):
            return float(v[0])                 # 시나리오 고정 목표 속도
        u = rng.uniform(0.02, 0.98)
        return float(np.interp(u, np.concatenate([[0.0], q, [1.0]]),
                               np.concatenate([[v[0] - 2.0], v, [v[-1] + 1.5]])))


@dataclass
class Npc:
    uid: int
    lane: int
    s: float
    v: float
    a: float = 0.0
    v_target: float = 13.9      # 프로파일이 지금 요구하는 속도
    v_desired: float = 23.6     # 순항 목표
    accel_cap: float = 1.4      # 프로파일 가속도(스폰 가속 또는 에피소드)
    episode_until: float = -1.0
    length: float = 4.8
    width: float = 1.9
    born: float = 0.0
    history: list = field(default_factory=list)  # (t, v) 관측용

    def profile_accel(self, t, dt, model, rng):
        """목표 속도 프로파일이 원하는 가속도(앞차 무시)."""
        if t >= self.episode_until and rng.random() < model.episode_rate_per_min / 60.0 * dt:
            dur = rng.lognormal(math.log(model.episode_duration_median), 0.6)
            self.episode_until = t + dur
            if rng.random() < 0.5:
                self.accel_cap = abs(rng.normal(model.episode_accel_mean, 0.4))
                self.v_target = self.v + self.accel_cap * dur
            else:
                self.accel_cap = abs(rng.normal(model.episode_decel_mean, 0.5))
                self.v_target = max(0.0, self.v - self.accel_cap * dur)
        if t >= self.episode_until:
            self.v_target = self.v_desired
        err = self.v_target - self.v
        if abs(err) < 0.05:
            return 0.0
        return math.copysign(min(self.accel_cap, abs(err) / dt), err)


def idm_accel(v, gap, v_lead, a_free, model):
    """IDM: a = a_free · [1 − (s*/gap)²] 를 앞차가 있을 때만 적용한다.

    a_free 는 프로파일 가속도(음수일 수 있다). s* = s0 + vT + v·Δv / (2√(a·b)).
    """
    if gap is None or not math.isfinite(gap):
        return a_free
    a_ref = max(0.5, abs(a_free) if a_free > 0 else 1.0)
    dv = v - v_lead
    s_star = model.idm_s0 + max(0.0, v * model.idm_T + v * dv / (2.0 * math.sqrt(a_ref * model.idm_b)))
    interaction = (s_star / max(gap, 0.1)) ** 2
    a = min(a_free, a_ref * (1.0 - interaction)) if a_free >= 0 else a_free - a_ref * interaction
    return max(-model.max_brake, a)
