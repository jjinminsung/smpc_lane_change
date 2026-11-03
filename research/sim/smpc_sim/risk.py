"""스텝별 충돌 위험. smpc_decision_node.cpp collisionProbability / evaluateTrajectory 이식.

배열 모양 규칙: 자차 궤적은 [N, K] (N = 후보 궤적 수, K = horizon 스텝 수).
스텝 k 의 시각은 t_k = dt * (k + 1) 이다(C++ 의 k = 1..N).
"""
from dataclasses import dataclass, field

import numpy as np
from scipy.special import ndtr

from .prediction import front_bumper_gap, predict_tv, capped_accel_speed, onset_speed_and_distance

INF = float("inf")


@dataclass
class EgoTrajectory:
    t: np.ndarray        # [K]
    x: np.ndarray        # [N, K]
    y: np.ndarray        # [N, K]
    yaw: np.ndarray      # [N, K]
    v: np.ndarray        # [N, K]
    valid: np.ndarray    # [N] bool. 빈 궤적(C++ empty_ego_traj)이면 False
    step_mask: np.ndarray = None  # [N, K] bool. preview 범위 밖에서 잘린 스텝은 False

    def __post_init__(self):
        if self.step_mask is None:
            self.step_mask = np.ones(self.x.shape, dtype=bool)

    @property
    def n(self):
        return self.x.shape[0]


@dataclass
class ModeTrace:
    unique_id: int
    role: str
    mode: str
    probability: float
    conditional_risk: np.ndarray   # [N, K] 이 mode 가 사실일 때의 충돌 확률
    clearance: np.ndarray          # [N, K] signed clearance (계산 안 한 스텝은 +inf)


@dataclass
class TrajectoryEval:
    valid: np.ndarray
    risk_by_step: np.ndarray
    peak_step_risk: np.ndarray
    min_clearance_by_step: np.ndarray
    min_clearance: np.ndarray
    legacy_risk: np.ndarray
    traces: list = field(default_factory=list)


# ------------------------------------------------------------- 충돌 확률
def ego_box(x, y, yaw, P):
    """smpc_decision_node.cpp:2821 egoOrientedBox. 기준점(후륜축)에서 박스 중심으로 옮긴다."""
    off = 0.5 * (P.ego_front_extent_m - P.ego_rear_extent_m)
    return (x + off * np.cos(yaw), y + off * np.sin(yaw), yaw,
            0.5 * (P.ego_front_extent_m + P.ego_rear_extent_m), 0.5 * P.ego_width_m)


def _axis_extent(yaw, half_l, half_w, ax, ay):
    return (half_l * np.abs(np.cos(yaw) * ax + np.sin(yaw) * ay)
            + half_w * np.abs(-np.sin(yaw) * ax + np.cos(yaw) * ay))


def mode_sigma(mode_name, P):
    if mode_name == "nominal":
        return P.prediction_nominal_sigma_base_m, P.prediction_nominal_sigma_growth_mps
    if mode_name == "front_brake":
        return P.prediction_front_brake_sigma_base_m, P.prediction_front_brake_sigma_growth_mps
    if mode_name == "rear_accel":
        return P.prediction_rear_accel_sigma_base_m, P.prediction_rear_accel_sigma_growth_mps
    return P.prediction_sigma_base_m, P.prediction_sigma_growth_mps


def collision_probability(ego_x, ego_y, ego_yaw, t, tv_x, tv_y, tv_yaw, tv_len, tv_wid,
                          mode_name, P):
    """smpc_decision_node.cpp:2831. 분리축(SAT) 4개마다 겹칠 확률을 구해 최솟값을 쓴다.

    축 방향 간격 gap 이 margin 보다 작을 확률 Φ((margin - gap)/σ). 차량 길이 방향과
    나란한 축일수록 종방향 σ, 수직일수록 횡방향 σ 를 쓴다.
    """
    ex, ey, eyaw, ehl, ehw = ego_box(ego_x, ego_y, ego_yaw, P)
    thl, thw = 0.5 * max(0.1, tv_len), 0.5 * max(0.1, tv_wid)
    dx, dy = tv_x - ex, tv_y - ey
    base, growth = mode_sigma(mode_name, P)
    sigma_long = np.maximum(1e-3, base + growth * t)
    sigma_lat = np.maximum(1e-3, P.prediction_lateral_sigma_base_m
                           + P.prediction_lateral_sigma_growth_mps * t)

    prob = np.ones(np.broadcast(ex, tv_x).shape)
    best_gap = np.full(prob.shape, -INF)
    for axis_yaw, normal in ((eyaw, False), (eyaw, True), (tv_yaw, False), (tv_yaw, True)):
        ax = -np.sin(axis_yaw) if normal else np.cos(axis_yaw)
        ay = np.cos(axis_yaw) if normal else np.sin(axis_yaw)
        centre = np.abs(dx * ax + dy * ay)
        gap = centre - _axis_extent(eyaw, ehl, ehw, ax, ay) - _axis_extent(tv_yaw, thl, thw, ax, ay)
        best_gap = np.maximum(best_gap, gap)
        longness = np.maximum(np.abs(np.cos(eyaw) * ax + np.sin(eyaw) * ay),
                              np.abs(np.cos(tv_yaw) * ax + np.sin(tv_yaw) * ay))
        sigma = longness * sigma_long + (1.0 - longness) * sigma_lat
        prob = np.minimum(prob, ndtr((P.collision_margin_m - gap) / sigma))
    return np.clip(prob, 0.0, 1.0), best_gap - P.collision_margin_m


# ------------------------------------------------------------ 위험 가중치
def priority_scale(targets, v, change_left, P):
    """smpc_decision_node.cpp:2281 mergePriorityRiskScale (merge_priority 는 yaml 에서 꺼짐)."""
    if targets is None:
        return 1.0
    if P.merge_priority_enabled:
        raise NotImplementedError("merge_priority_enabled=true 는 이식하지 않았다")
    if not (P.ego_lane_priority_enabled and v.valid):
        return 1.0
    current_front, nearby = v.has_role("current_front"), v.has_role("nearby")
    if not (current_front or nearby):
        return 1.0
    if front_bumper_gap(v, P) <= P.ego_lane_priority_front_hard_gap_m:
        return 1.0
    side = min(max(P.ego_lane_priority_side_risk_scale, 0.0), 1.0)
    if current_front:
        return side if abs(v.d) > P.ego_lane_priority_current_max_abs_d_m else 1.0
    if v.lane_id == targets.current_lane_id:
        return side if abs(v.d) > P.ego_lane_priority_current_max_abs_d_m else 1.0
    if change_left and v.lane_id == targets.target_lane_id:
        return 1.0
    pressure = targets.lane_change_required or targets.lane_change_urgent or targets.emergency_stop_required
    return side if pressure else 1.0


# --------------------------------------------------------------- 궤적 평가
def evaluate_mode(ego, v, mode, ego_speed, lanes, P, react=None, ego_override_v=None):
    """한 TV 의 한 mode 에 대한 스텝별 조건부 충돌 확률과 clearance.

    react: None 이면 F7(앞차 감속 시 자차 제동 반응)을 yaml 대로 적용.
    반환: cond [N,K], clear [N,K], active [N,K] (TV 가 차선 끝을 넘은 뒤는 False),
          invalid [N] (차선 끝 전에 샘플 실패).
    """
    N, K = ego.x.shape
    t = ego.t
    if react is None:
        react = P.ego_brake_reaction_enabled and v.has_role("front")
    lag = np.zeros((N, K))
    if react:
        tv_v, _ = onset_speed_and_distance(v.v_long, mode, t)       # [K]
        react_v = np.full(N, max(0.0, ego_speed))
        acc_lag = np.zeros(N)
        trigger = np.full(N, INF)
        dec = max(0.0, P.ego_brake_reaction_max_decel_mps2) * P.prediction_dt_sec
        for k in range(K):
            ev = ego.v[:, k]
            trigger = np.where(np.isinf(trigger) & (tv_v[k] < ev), t[k], trigger)
            reacting = np.isfinite(trigger) & (t[k] >= trigger + max(0.0, P.ego_brake_reaction_delay_sec))
            react_v = np.where(reacting, np.maximum(tv_v[k], np.minimum(ev, react_v - dec)), ev)
            acc_lag = acc_lag + np.maximum(0.0, ev - react_v) * P.prediction_dt_sec
            lag[:, k] = acc_lag
    pred = predict_tv(v, mode, t[None, :], lanes, P, extra_ds=lag)
    exceeded = np.broadcast_to(pred["exceeded"], (N, K))
    active = ~np.logical_or.accumulate(exceeded, axis=1) & ego.step_mask
    valid = np.broadcast_to(pred["valid"], (N, K))
    invalid = np.any(active & ~valid, axis=1)
    cond, clear = collision_probability(ego.x, ego.y, ego.yaw, t[None, :],
                                        pred["x"], pred["y"], pred["yaw"],
                                        v.length, v.width, mode.name, P)
    cond = np.where(active, cond, 0.0)
    clear = np.where(active, clear, INF)
    return cond, clear, active, invalid


def evaluate_trajectory(ego, vehicles, modes_of, ego_speed, lanes, P, targets=None,
                        change_left=False):
    """smpc_decision_node.cpp:2893 evaluateTrajectory.

    vehicles: TargetVehicle 목록, modes_of(v) -> [TvMode].
    risk_by_step[k] = 1 - Π_v (1 - scale_v · Σ_m p_m · P_vm[k])   (차량 간 독립 가정)
    """
    N, K = ego.x.shape
    no_coll = np.ones((N, K))
    legacy_no_coll = np.ones(N)
    min_clear_step = np.full((N, K), INF)
    valid = ego.valid.copy()
    traces = []
    for v in vehicles:
        scale = priority_scale(targets, v, change_left, P)
        mixed = np.zeros((N, K))
        legacy = np.zeros(N)
        for mode in modes_of(v):
            cond, clear, active, invalid = evaluate_mode(ego, v, mode, ego_speed, lanes, P)
            valid &= ~invalid
            p = min(max(mode.probability, 0.0), 1.0)
            mixed += p * cond
            legacy += p * cond.max(axis=1)
            min_clear_step = np.minimum(min_clear_step, clear)
            traces.append(ModeTrace(v.unique_id, v.role, mode.name, p, cond, clear))
        mixed = np.clip(mixed * scale, 0.0, 1.0)
        no_coll *= (1.0 - mixed)
        legacy_no_coll *= (1.0 - np.clip(legacy * scale, 0.0, 1.0))
    risk = np.clip(1.0 - no_coll, 0.0, 1.0)
    return TrajectoryEval(
        valid=valid,
        risk_by_step=risk,
        peak_step_risk=risk.max(axis=1),
        min_clearance_by_step=min_clear_step,
        min_clearance=min_clear_step.min(axis=1),
        legacy_risk=np.clip(1.0 - legacy_no_coll, 0.0, 1.0),
        traces=traces,
    )


def time_indexed_risk_safe(ev, allowed_risk):
    """smpc_decision_node.cpp:3426. 모든 스텝에서 risk ≤ ε."""
    return ev.valid & np.all(np.isfinite(ev.risk_by_step) & (ev.risk_by_step <= allowed_risk), axis=1)


def clearance_safe(ev, P):
    """smpc_decision_node.cpp:3434. 확률과 무관하게 모든 mode 의 clearance ≥ 기준(0 m)."""
    c = ev.min_clearance_by_step
    ok = (np.isinf(c) & (c > 0)) | (np.isfinite(c) & (c >= P.behavior_min_signed_clearance_m))
    return ev.valid & np.all(ok, axis=1)
