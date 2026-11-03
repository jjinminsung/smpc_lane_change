"""주변 차량(TV) mode 와 예측. smpc_decision_node.cpp 의 해당 함수를 그대로 옮겼다.

모든 함수는 numpy 배열을 받아 브로드캐스팅한다. 트리 플래너가 수백 개 궤적을
한 번에 평가하기 위해서다.
"""
from dataclasses import dataclass
import math

import numpy as np

INF = float("inf")


@dataclass
class TvMode:
    """smpc_decision_node.cpp:492 TvMode."""

    name: str
    probability: float
    accel_mps2: float
    lateral_rate_mps: float = 0.0
    max_speed_mps: float = INF
    # 트리 플래너용 확장: 이 시각 이전에는 nominal 과 같이 움직이고, 이후 accel 로 가속.
    # 기존 SMPC 의 mode 는 모두 onset 0 이다.
    onset_sec: float = 0.0


@dataclass
class StraightLanes:
    """직선·평행 차선. lane i 의 중심 y = i * width, 유효 s 는 [-ext, end_s[i]]."""

    width: float
    end_s: dict
    extension_before_m: float = 100.0

    def center_y(self, lane_id):
        return lane_id * self.width

    def length(self, lane_id):
        return self.end_s.get(lane_id, INF)


# ---------------------------------------------------------------- 운동학
def constant_accel_distance(v0, accel, tau):
    """smpc_decision_node.cpp:1152. 감속이면 정지 후 멈춘다."""
    v0 = np.maximum(0.0, v0)
    tau = np.maximum(0.0, tau)
    accel = np.asarray(accel, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stop = np.where(accel < -1e-6, v0 / np.where(accel < -1e-6, -accel, 1.0), INF)
    used = np.minimum(tau, t_stop)
    braking = np.maximum(0.0, v0 * used + 0.5 * accel * used * used)
    cruising = np.maximum(0.0, v0 * tau + 0.5 * accel * tau * tau)
    return np.where(accel < -1e-6, braking, cruising)


def capped_accel_speed(v0, accel, tau, v_max):
    """smpc_decision_node.cpp:1655. 상한 도달 후 등속, 0 미만 금지."""
    v0 = np.maximum(0.0, v0)
    tau = np.maximum(0.0, tau)
    return np.clip(v0 + accel * tau, 0.0, np.maximum(v0, v_max))


def capped_accel_distance(v0, accel, tau, v_max):
    """smpc_decision_node.cpp:1663."""
    v0 = np.maximum(0.0, v0)
    tau = np.maximum(0.0, tau)
    plain = constant_accel_distance(v0, accel, tau)
    if not math.isfinite(v_max) or v_max <= 0.0 or accel <= 1e-6:
        return plain
    if np.all(v0 >= v_max):
        return np.maximum(0.0, v0 * tau)
    t_cap = (v_max - v0) / accel
    d_cap = v0 * t_cap + 0.5 * accel * t_cap * t_cap
    capped = np.maximum(0.0, d_cap + v_max * (tau - t_cap))
    out = np.where(tau <= t_cap, plain, capped)
    return np.where(v0 >= v_max, np.maximum(0.0, v0 * tau), out)


def onset_speed_and_distance(v0, mode, tau):
    """onset 이 있는 mode: onset 전까지 nominal(등가속 0), 이후 mode 가속.

    onset_sec = 0 이면 capped_accel_* 와 정확히 같다(기존 SMPC 와 동일).
    """
    if mode.onset_sec <= 0.0:
        return (capped_accel_speed(v0, mode.accel_mps2, tau, mode.max_speed_mps),
                capped_accel_distance(v0, mode.accel_mps2, tau, mode.max_speed_mps))
    v0 = max(0.0, float(v0))
    t_on = mode.onset_sec
    before = np.minimum(tau, t_on)
    after = np.maximum(0.0, tau - t_on)
    ds = v0 * before + capped_accel_distance(v0, mode.accel_mps2, after, mode.max_speed_mps)
    v = capped_accel_speed(v0, mode.accel_mps2, after, mode.max_speed_mps)
    return v, ds


# ------------------------------------------------------------ 간격 헬퍼
def vehicle_half_s(v):
    return v.half_s if v.half_s > 1e-3 else 0.5 * max(0.0, v.length)


def front_edge_delta_s(v):
    return v.front_delta_s if v.front_delta_s != 0.0 else v.delta_s + vehicle_half_s(v)


def rear_edge_delta_s(v):
    return v.rear_delta_s if v.rear_delta_s != 0.0 else v.delta_s - vehicle_half_s(v)


def front_bumper_gap(v, P):
    return rear_edge_delta_s(v) - P.ego_front_extent_m


def rear_bumper_gap(v, P):
    return -front_edge_delta_s(v) - P.ego_rear_extent_m


def _pressure01(value, low, high):
    if high <= low:
        return 1.0 if value >= high else 0.0
    return min(max((value - low) / (high - low), 0.0), 1.0)


# ------------------------------------------------------------- mode 확률
def adaptive_event_probability(v, accel_obs, ego_speed, P):
    """smpc_decision_node.cpp:1681 adaptiveEventProbability."""
    rear = v.has_role("rear")
    breaks = P.rear_accel_probability_accel_breaks_mps2
    table = P.rear_accel_probability_by_accel
    if rear and P.rear_accel_probability_by_accel_enabled and len(table) == len(breaks) + 1:
        b = 0
        while b < len(breaks) and accel_obs >= breaks[b]:
            b += 1
        return min(max(table[b], P.min_rear_accel_probability), P.max_rear_accel_probability)

    span = P.probability_ttc_safe_sec - P.probability_ttc_critical_sec
    if rear:
        gap = max(0.0, rear_bumper_gap(v, P))
        closing = max(0.0, v.v_long - ego_speed)
        ttc = gap / closing if closing > 0.1 else INF
        ttc_p = _pressure01(P.probability_ttc_safe_sec - ttc, 0.0, span) if math.isfinite(ttc) else 0.0
        p = (P.rear_accel_probability
             + 0.25 * ttc_p
             + 0.18 * _pressure01(closing, 0.0, P.probability_rel_speed_scale_mps)
             + 0.12 * _pressure01(max(0.0, accel_obs), 0.0, P.probability_accel_scale_mps2)
             + 0.10 * _pressure01(P.target_rear_safe_gap_m - gap, 0.0, P.target_rear_safe_gap_m))
        return min(max(p, P.min_rear_accel_probability), P.max_rear_accel_probability)

    gap = max(0.0, front_bumper_gap(v, P))
    desired = max(P.current_front_safe_gap_m, P.standstill_gap_m + P.time_headway_sec * ego_speed)
    closing = max(0.0, ego_speed - v.v_long)
    ttc = gap / closing if closing > 0.1 else INF
    ttc_p = _pressure01(P.probability_ttc_safe_sec - ttc, 0.0, span) if math.isfinite(ttc) else 0.0
    p = (P.front_brake_probability
         + 0.25 * ttc_p
         + 0.18 * _pressure01(closing, 0.0, P.probability_rel_speed_scale_mps)
         + 0.12 * _pressure01(max(0.0, -accel_obs), 0.0, P.probability_accel_scale_mps2)
         + 0.10 * _pressure01(desired - gap, 0.0, desired))
    return min(max(p, P.min_front_brake_probability), P.max_front_brake_probability)


def modes_for_vehicle(v, accel_obs, ego_speed, P):
    """smpc_decision_node.cpp:2407 modesForVehicle (횡방향 mode 는 yaml 에서 꺼져 있다)."""
    if P.lateral_mode_enabled:
        raise NotImplementedError("lateral_mode_enabled=true 는 이식하지 않았다")
    rear = v.has_role("rear")
    p_event = adaptive_event_probability(v, accel_obs, ego_speed, P)
    event_accel = P.rear_accel_mps2 if rear else P.front_brake_accel_mps2
    event = TvMode("rear_accel" if rear else "front_brake", p_event, event_accel)
    if rear and P.rear_accel_max_speed_enabled:
        raise NotImplementedError("rear_accel_max_speed_enabled=true 는 이식하지 않았다")
    modes = [TvMode("nominal", 1.0 - p_event, P.nominal_target_accel_mps2), event]
    total = max(1e-6, sum(max(0.0, m.probability) for m in modes))
    for m in modes:
        m.probability = min(max(m.probability / total, 0.0), 1.0)
    return modes


# ----------------------------------------------------------------- 예측
def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _angle_lerp(a, b, alpha):
    return _wrap(a + alpha * _wrap(b - a))


def predict_tv(v, mode, tau, lanes, P, extra_ds=0.0):
    """smpc_decision_node.cpp:2753 predictTargetPose (직선 차선판).

    tau, extra_ds 는 브로드캐스팅 가능한 배열. 반환 dict 의 exceeded 는 TV 가
    자기 차선 CSV 끝을 넘은 스텝(위험 계산을 거기서 멈춘다), valid 는 샘플 가능 여부.
    """
    tau = np.asarray(tau, dtype=float)
    v_pred, ds = onset_speed_and_distance(v.v_long, mode, tau)
    ds = ds + np.maximum(0.0, extra_ds)
    if abs(mode.lateral_rate_mps) > 1e-3:
        d = v.d + mode.lateral_rate_mps * tau
    else:
        d = v.d * np.exp(-tau / max(0.1, P.lane_center_decay_tau_sec))
    d = np.clip(d, -1.25 * P.lane_width_m, 1.25 * P.lane_width_m)
    s = v.s + ds
    exceeded = s > lanes.length(v.lane_id) + 1e-6
    valid = (~exceeded) & (s >= -lanes.extension_before_m)

    lane_yaw = 0.0
    yaw_vel = _wrap(lane_yaw + np.arctan2(mode.lateral_rate_mps, np.maximum(0.1, v_pred)))
    yaw_rate = min(max(v.yaw_rate, -P.max_abs_tv_yaw_rate), P.max_abs_tv_yaw_rate)
    yaw_rate_pred = _wrap(v.yaw + yaw_rate * np.minimum(tau, P.tv_yaw_rate_horizon_sec))
    dynamic = _angle_lerp(yaw_vel, yaw_rate_pred, min(max(P.tv_yaw_rate_weight, 0.0), 1.0))
    yaw = _angle_lerp(dynamic, lane_yaw, min(max(P.tv_csv_yaw_weight, 0.0), 1.0))
    return {
        "x": s,                       # 직선 차선이라 전역 x = s
        "y": lanes.center_y(v.lane_id) + d,
        "yaw": yaw * np.ones_like(s),
        "v": v_pred * np.ones_like(s),
        "s": s,
        "d": d * np.ones_like(s),
        "exceeded": exceeded,
        "valid": valid,
    }
