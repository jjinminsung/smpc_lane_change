"""기존 SMPC 의 결정론적 게이트. baseline_spec.md 0장·2.1·2.3 과 smpc_decision_node.cpp:3539.

모두 '확률과 무관한' 조건이다. 연구 방향 2 에서 걷어낼 대상이 여기 모여 있다.
"""
import math

import numpy as np

from .prediction import (front_bumper_gap, rear_bumper_gap, front_edge_delta_s, rear_edge_delta_s,
                         modes_for_vehicle, predict_tv)
from .rollouts import execution_start

INF = float("inf")
NEW_TRACK_GUARD_SEC, NEW_TRACK_MAX_GAP, NEW_TRACK_CLOSING = 0.80, 40.0, 15.0


def vehicle_age(hist, v, now):
    h = hist.histories.get(v.unique_id)
    return now - h.first_seen if h is not None and h.initialized else 0.0


def conservative_rear_closing(v, ego_v, hist, now, P):
    """conservativeRearClosingSpeed (smpc:1851): 새 트랙은 15 m/s 로 접근한다고 가정."""
    c = max(0.0, v.v_long - max(0.0, ego_v))
    if (vehicle_age(hist, v, now) < P.new_rear_track_guard_sec and
            rear_bumper_gap(v, P) <= P.new_rear_track_max_gap_m):
        c = max(c, P.new_rear_track_closing_upper_mps)
    return c


def target_rear_required_gap(v_e, closing, remaining, P):
    """targetRearRequiredGap (smpc:1870)."""
    base = P.target_rear_safe_gap_m
    if P.low_speed_rear_safety_enabled and v_e <= P.low_speed_rear_ego_speed_mps:
        base = max(base, P.low_speed_rear_safe_gap_m)
    return base + closing * (remaining + P.rear_gap_perception_control_delay_sec)


def target_front_is_passing(t, v_e, P):
    """targetFrontIsPassingVehicle (smpc:1993)."""
    tf = t.target_front
    return (P.pass_gap_enabled and tf.valid and tf.unique_id >= 0 and tf.lane_id == t.target_lane_id
            and tf.v_long - v_e >= P.pass_gap_min_relative_speed_mps
            and front_bumper_gap(tf, P) <= P.pass_gap_front_max_distance_m)


def target_rear_for_gap(t, exclude_id):
    """targetRearForGap (smpc:2009): 목표 차선에서 자차 뒤(또는 겹침) 차량 중 가장 가까운 것."""
    best, best_gap = None, INF
    for v in [t.target_rear] + list(t.nearby_vehicles):
        if (not v.valid or v.unique_id < 0 or v.unique_id == exclude_id or
                v.lane_id != t.target_lane_id or rear_edge_delta_s(v) > 0.0):
            continue
        g = -front_edge_delta_s(v)
        if g < best_gap:
            best, best_gap = v, g
    return best


# ------------------------------------------------------ G3 즉시 간격/TTC
def immediate_gap_gate(t, v_e, R, passing, rear_required, hist, now, P):
    """change_feasible 의 결정론적 간격 부분 (smpc:4113-4126). (ok, reason)."""
    tf = t.target_front
    tf_gap = front_bumper_gap(tf, P) if tf.valid else INF
    if passing:
        if not tf_gap >= P.pass_gap_entry_front_clearance_m:
            return False, "pass_gap_front"
    else:
        if tf.valid and not tf_gap > P.target_front_safe_gap_m:
            return False, "front_gap"
        c_f = max(0.0, v_e - max(0.0, tf.v_long)) if tf.valid else 0.0
        if tf.valid and c_f > 0.1 and max(0.0, tf_gap) / c_f < P.target_front_min_ttc_sec:
            return False, "front_ttc"
        if tf.valid and max(0.0, tf_gap) < P.target_front_min_headway_sec * c_f:
            return False, "front_headway"
    if R is not None:
        tr_gap = rear_bumper_gap(R, P)
        if not tr_gap > rear_required:
            return False, "rear_gap"
        c_r = conservative_rear_closing(R, v_e, hist, now, P)
        ttc_req = (P.low_speed_rear_min_ttc_sec if v_e <= P.low_speed_rear_ego_speed_mps
                   else P.target_rear_min_ttc_sec)
        if c_r > 0.1 and max(0.0, tr_gap) / c_r < ttc_req:
            return False, "rear_ttc"
        if max(0.0, tr_gap) < P.target_rear_min_headway_sec * c_r:
            return False, "rear_headway"
    return True, ""


# ------------------------------------------------- G7 끼어들기 차단 (B)
def cutin_interlock_safe(t, v_e, passing, passing_id, lanes, P):
    """observedTargetLaneIntrusionSafe 중 config 로 남은 (B): 목표 차선 뒤·옆 모든 차량."""
    y_tgt = lanes.center_y(t.target_lane_id)
    for v in t.nearby_vehicles:
        if not v.has_role("nearby") or v.rear_delta_s > 0.0:
            continue
        if passing and v.unique_id == passing_id:
            continue
        g = max(0.0, -v.front_delta_s - P.ego_rear_extent_m)
        c = max(0.0, v.v_long - v_e)
        half_d = 0.5 * v.width
        if abs(v.y - y_tgt) <= 0.5 * P.lane_width_m + half_d:
            if g <= max(P.target_rear_safe_gap_m, P.target_rear_min_headway_sec * c) or \
                    (c > 0.1 and g / c < P.target_rear_min_ttc_sec):
                return False
    return True


# --------------------------------------------- G5 예측 간격/TTC (smpc:3539)
def projected_gap_ttc_safe(t, traj, specs, R, passing, modes_of, lanes, hist, now, P):
    """projectedTargetGapTtcSafe 를 후보 N 개에 대해 한 번에 계산. 반환 [N] bool.

    자차 박스가 목표 차선 통로에 들어온 스텝부터, target_front 와 R(또는 target_rear)에
    대해 간격과 TTC 를 요구한다. 뒤차는 nominal mode 만, 추월 중인 앞차도 nominal 만 본다.
    """
    N, K = traj.x.shape
    ok = traj.valid.copy()
    t_sw = np.array([execution_start(s, P) for s in specs])[:, None]
    dur = np.array([max(0.1, s.duration) for s in specs])[:, None]
    change = np.array([s.change_left for s in specs])
    ok &= change
    y_tgt = lanes.center_y(t.target_lane_id)
    half_w = 0.5 * P.ego_width_m
    half_l = 0.5 * (P.ego_front_extent_m + P.ego_rear_extent_m)
    entry = 0.5 * P.lane_width_m + half_w + P.collision_margin_m
    full_occ = P.ego_reference_preview_target_completion_max_abs_d_m
    abs_d = np.abs(traj.y - y_tgt)
    in_corridor = traj.step_mask & (traj.t[None, :] + 1e-6 >= t_sw) & (abs_d <= entry)
    occupancy = np.clip((entry - abs_d) / max(0.1, entry - full_occ), 0.0, 1.0)

    def check(v, front):
        nonlocal ok
        if v is None or not v.valid:
            return
        for mode in modes_of(v):
            if mode.name != "nominal" and (not front or passing):
                continue
            pred = predict_tv(v, mode, traj.t[None, :], lanes, P)
            use = in_corridor & ~np.broadcast_to(pred["exceeded"], (N, K))
            bad_sample = use & ~np.broadcast_to(pred["valid"], (N, K))
            longitudinal = pred["x"] - traj.x
            tv_extent = (0.5 * max(0.1, v.length) * np.abs(np.cos(pred["yaw"]))
                         + 0.5 * max(0.1, v.width) * np.abs(np.sin(pred["yaw"])))
            tvf = np.maximum(0.0, pred["v"] * np.cos(pred["yaw"]))
            if front:
                gap = longitudinal - half_l - tv_extent
                ref = traj.v if passing else np.maximum(0.0, traj.v - tvf)
                full_req = np.maximum(P.target_front_safe_gap_m, P.target_front_min_headway_sec * ref)
                pass_entry = P.pass_gap_entry_front_clearance_m + P.pass_gap_entry_length_margin_m
                if passing:
                    receding = P.pass_gap_skip_ramp_while_receding & (tvf > traj.v)
                    req = np.where(receding, pass_entry,
                                   pass_entry + occupancy * np.maximum(0.0, full_req - pass_entry))
                else:
                    req = full_req
                closing = np.maximum(0.0, traj.v - tvf)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fail = ~(gap > req) | ((closing > 0.1) & (gap / closing < P.target_front_min_ttc_sec))
            else:
                gap = -longitudinal - half_l - tv_extent
                base_c = np.maximum(0.0, v.v_long - np.maximum(0.0, traj.v))
                if (vehicle_age(hist, v, now) < P.new_rear_track_guard_sec and
                        rear_bumper_gap(v, P) <= P.new_rear_track_max_gap_m):
                    base_c = np.maximum(base_c, P.new_rear_track_closing_upper_mps)
                closing = np.maximum(base_c, np.maximum(0.0, tvf - traj.v))
                low = P.low_speed_rear_safety_enabled & (traj.v <= P.low_speed_rear_ego_speed_mps)
                base_gap = np.where(low, max(P.target_rear_safe_gap_m, P.low_speed_rear_safe_gap_m),
                                    P.target_rear_safe_gap_m)
                remaining = np.maximum(0.0, t_sw + dur - traj.t[None, :])
                req = base_gap + closing * (remaining + P.rear_gap_perception_control_delay_sec)
                ttc_req = np.where(low, max(P.target_rear_min_ttc_sec, P.low_speed_rear_min_ttc_sec),
                                   P.target_rear_min_ttc_sec)
                with np.errstate(divide="ignore", invalid="ignore"):
                    fail = ~(gap > req) | ((closing > 0.1) & (gap / closing < ttc_req))
            ok &= ~np.any(use & fail, axis=1) & ~np.any(bad_sample, axis=1)

    check(t.target_front, True)
    check(R if R is not None else (t.target_rear if t.target_rear.valid else None), False)
    return ok
