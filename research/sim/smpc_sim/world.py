"""1 차원 Frenet 고속도로 시뮬레이터. 한 에피소드 = 차선 0 에서 출발해 0→1→2→3 합류.

틱 순서(0.05 s, baseline_spec.md 6장): NPC → 인지 → SMPC(planner) → supervisor → ACC → 자차.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .acc import AccPlanner
from .npc import Npc, NpcModel, idm_accel
from .params import load_params
from .perception import build_targets
from .plant import Ego, EgoPlant
from .supervisor import Supervisor
from .track import Track

# MORAI 실측 스폰(research/results/day1_m3_tracks.csv, 시나리오 highway_spawn50).
# 차선: (위치 s, 13.9 m/s 로 나타날 확률, 간격 범위 s, 동시 최대 대수 maximumSpawnVehicle)
SPAWN = {1: (-81.0, 0.83, (3.3, 6.3), 15),
         2: (-54.0, 0.89, (4.1, 4.5), 15),
         3: (-16.0, 0.52, (4.1, 4.5), 5)}


@dataclass
class EpisodeConfig:
    seed: int = 0
    dt: float = 0.05
    warmup_sec: float = 45.0
    time_limit_sec: float = 120.0
    mission_speed: float = 25.0
    ego_s0: tuple = (-27.0, -4.0)
    ego_v0: tuple = (10.8, 16.0)
    spawn_scale: float = 1.0          # >1 이면 교통이 더 촘촘하다
    npc_lane_change_rate_per_min: float = 0.0
    record: bool = False
    collect_samples: bool = False     # RL: 0.5 s 마다 (시각, 특징) 을 남긴다
    sample_period: float = 0.5


@dataclass
class EpisodeResult:
    seed: int
    planner: str
    success: bool
    collision: bool
    final_lane: int
    t_end: float
    t_merge: float                   # 차선 3 진입 완료 시각 (실패면 nan)
    lane_end_stops: int              # 차선 끝 근처에서 정지(<0.5 m/s)한 횟수
    stop_time: float                 # 차선 0~2 에서 정지해 있던 시간
    aborts: int
    accepts: int
    min_gap: float                   # GT 최소 bumper 간격(같은 차선)
    min_ttc: float
    lane_change_times: list = field(default_factory=list)
    events: list = field(default_factory=list)
    trace: list = field(default_factory=list)
    samples: list = field(default_factory=list)       # [(t, features)]
    stop_events: list = field(default_factory=list)   # 차선 끝 정지 시각


def obb_overlap(e, n, track):
    """자차(후륜축 기준)와 NPC(중심) 박스가 겹치는지. 차선이 직선이라 축 정렬 박스로 충분하다."""
    e_front, e_back = e.s + e.front, e.s - e.rear
    n_front, n_back = n.s + 0.5 * n.length, n.s - 0.5 * n.length
    if e_back >= n_front or n_back >= e_front:
        return False
    return abs(e.y - n.y) < 0.5 * (e.width + n.width)


class World:
    def __init__(self, planner_factory, cfg=None, P=None):
        self.cfg = cfg or EpisodeConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.P = P or load_params("smpc_decision")
        self.P_sel = load_params("target_selector")
        self.P_acc = load_params("acc_planner")
        self.P_sup = load_params("lane_change_supervisor") if (self._has("lane_change_supervisor")) else load_params("supervisor")
        self.track = Track()
        self.model = NpcModel.from_measurements()
        self.npcs = []
        self.next_uid = 100
        self.next_spawn = {lane: 0.0 for lane in SPAWN}
        self.t = 0.0
        self.planner = planner_factory(self.P, self.track)
        if hasattr(self.planner, "monitor"):
            self.P_sup.active_rear_guard_enabled = False
        self.ego = None

    @staticmethod
    def _has(name):
        from .params import CONFIG_DIR
        return (CONFIG_DIR / f"{name}.yaml").exists()

    # ------------------------------------------------------------ NPC
    def spawn(self):
        for lane, (s_sp, p_fast, period, max_alive) in SPAWN.items():
            if self.t < self.next_spawn[lane]:
                continue
            if sum(1 for n in self.npcs if n.spawn_lane == lane) >= max_alive:
                self.next_spawn[lane] = self.t + 0.5
                continue
            occupied = any(n.lane == lane and abs(n.s - s_sp) < 12.0 for n in self.npcs)
            if self.ego is not None and self.ego.lane == lane and abs(self.ego.s - s_sp) < 15.0:
                occupied = True
            if occupied:
                self.next_spawn[lane] = self.t + 0.5
                continue
            m = self.model
            v0 = m.spawn_speed if self.rng.random() < p_fast else 0.0
            n = Npc(uid=self.next_uid, lane=lane, s=s_sp, v=v0, born=self.t,
                    v_desired=m.sample_desired(self.rng),
                    accel_cap=float(np.clip(self.rng.normal(m.spawn_accel_mean, m.spawn_accel_std),
                                            m.spawn_accel_min, m.spawn_accel_max)),
                    length=m.length, width=m.width)
            n.v_target = n.v_desired
            n.y, n.v_lat, n.spawn_lane = self.track.center_y(lane), 0.0, lane
            self.npcs.append(n)
            self.next_uid += 1
            gap = self.rng.uniform(*period)
            self.next_spawn[lane] = self.t + gap / self.cfg.spawn_scale

    def occupies(self, y, width, lane):
        """폭 width 인 차량이 lane 을 0.3 m 이상 침범하는지."""
        return abs(y - self.track.center_y(lane)) < 0.5 * self.track.lane_width + 0.5 * width - 0.3

    def step_npcs(self, dt):
        m = self.model
        order = sorted(self.npcs, key=lambda n: -n.s)
        for n in order:
            lead_gap, lead_v = math.inf, 0.0
            for o in self.npcs:
                if o is n or o.lane != n.lane or o.s <= n.s:
                    continue
                g = (o.s - 0.5 * o.length) - (n.s + 0.5 * n.length)
                if g < lead_gap:
                    lead_gap, lead_v = g, o.v
            e = self.ego
            if e is not None and self.occupies(e.y, e.width, n.lane) and e.s + e.front > n.s:
                g = (e.s - e.rear) - (n.s + 0.5 * n.length)
                if g < lead_gap and g > -0.5 * n.length:
                    lead_gap, lead_v = g, e.v
            a_free = n.profile_accel(self.t, dt, m, self.rng)
            n.a = idm_accel(n.v, lead_gap, lead_v, a_free, m)
            v_new = max(0.0, n.v + n.a * dt)
            n.s += 0.5 * (n.v + v_new) * dt
            n.v = v_new
        self.npcs = [n for n in self.npcs if n.s < self.track.road_end[n.lane]]

    # ------------------------------------------------------------ run
    def reset(self):
        cfg = self.cfg
        while self.t < cfg.warmup_sec:
            self.spawn()
            self.step_npcs(cfg.dt)
            self.t += cfg.dt
        s0 = self.rng.uniform(*cfg.ego_s0)
        v0 = self.rng.uniform(*cfg.ego_v0)
        self.ego = Ego(s=s0, y=self.track.center_y(0), v=v0, lane=0)
        self.plant = EgoPlant(self.ego, self.track)
        self.sup = Supervisor(self.P_sup, self.track, initial_lane=0)
        self.acc = AccPlanner(self.P_acc, self.track)
        self.acc.f = v0
        self.t0 = self.t

    def run(self):
        cfg = self.cfg
        self.reset()
        e = self.ego
        res = EpisodeResult(seed=cfg.seed, planner=getattr(self.planner, "name", "?"), success=False,
                            collision=False, final_lane=0, t_end=0.0, t_merge=math.nan, lane_end_stops=0,
                            stop_time=0.0, aborts=0, accepts=0, min_gap=math.inf, min_ttc=math.inf)
        stopped = False
        last_change_left = -1e9
        while self.t - self.t0 < cfg.time_limit_sec:
            now = self.t
            self.spawn()
            self.step_npcs(cfg.dt)
            t = build_targets(now, e, self.npcs, self.track, self.P_sel, forced_pair=self.sup.forced_pair)
            dec = self.planner.decide(now, t, e.v, cfg.mission_speed, self.acc.f, self.acc.status)
            if dec.change_left:
                last_change_left = now
                self.acc.last_change_left = now
            n_ev = len(self.sup.events)
            self.sup.on_decision(now, dec.change_left, t, e)
            self.sup.on_targets(now, t, e)
            if self.sup.active and hasattr(self.planner, "monitor"):
                # 트리 플래너: 차선 변경 중 재계획이 rear_abort 를 대체한다
                if self.planner.monitor(now, t, e.v, cfg.mission_speed) == "return":
                    self.sup._abort(now, "tree_return", self.sup.source)
            self.sup.tick(now, e)
            for ev in self.sup.events[n_ev:]:
                res.events.append((ev.kind, round(ev.t - self.t0, 2), ev.source, ev.target, ev.detail,
                                   round(e.v, 2), round(t.distance_to_lane_end, 1)))
                if ev.kind == "accept":
                    res.accepts += 1
                elif ev.kind == "abort":
                    res.aborts += 1
                elif ev.kind == "complete":
                    res.lane_change_times.append(round(ev.t - self.t0, 2))
            self.plant.set_path(now, self.sup.lane)
            intent = self.sup.active or now - last_change_left <= 0.5 or dec.request.active
            lc_elapsed = now - self.sup.change_start if self.sup.active else 0.0
            v_cmd = self.acc.step(now, cfg.dt, t, e.v, e.y, cfg.mission_speed, dec.request,
                                  self.sup.active, lc_elapsed, intent)
            self.plant.step(now, cfg.dt, v_cmd)
            self.t += cfg.dt

            # 지표
            for n in self.npcs:
                if obb_overlap(e, n, self.track):
                    res.collision = True
                if self.occupies(n.y, n.width, e.lane) or abs(n.y - e.y) < 0.5 * (e.width + n.width):
                    if n.s > e.s:
                        g = (n.s - 0.5 * n.length) - (e.s + e.front)
                        c = e.v - n.v
                    else:
                        g = (e.s - e.rear) - (n.s + 0.5 * n.length)
                        c = n.v - e.v
                    if abs(n.y - e.y) < 0.5 * (e.width + n.width) + 0.3:
                        res.min_gap = min(res.min_gap, g)
                        if c > 0.1 and g > 0:
                            res.min_ttc = min(res.min_ttc, g / c)
            lane_now = self.sup.lane if not self.sup.active else self.sup.source
            if e.v < 0.5 and lane_now < self.track.final_lane:
                res.stop_time += cfg.dt
                if not stopped and self.track.csv_end[lane_now] - e.s < 60.0:
                    res.lane_end_stops += 1
                    res.stop_events.append(round(self.t - self.t0, 2))
                stopped = True
            elif e.v > 1.0:
                stopped = False
            if cfg.collect_samples and (now - self.t0) + 1e-9 >= len(res.samples) * cfg.sample_period:
                from .rl.features import state_features
                res.samples.append((round(now - self.t0, 2), state_features(t, e).tolist()))
            if cfg.record:
                res.trace.append((round(now - self.t0, 2), e.s, e.y, e.v, self.sup.lane, int(self.sup.active),
                                  dec.selected, dec.reason, v_cmd))
            if res.collision:
                break
            if not self.sup.active and self.sup.lane == self.track.final_lane:
                res.success = True
                res.t_merge = self.t - self.t0
                break
            lane_now = self.sup.lane
            if lane_now < self.track.final_lane and e.s > self.track.csv_end[lane_now] + 5.0:
                break      # 차선 끝을 지나쳤다(경로 이탈)
        res.final_lane = self.sup.lane
        res.t_end = self.t - self.t0
        return res
