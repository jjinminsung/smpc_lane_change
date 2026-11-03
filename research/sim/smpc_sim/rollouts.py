"""기존 SMPC 가 '예측하는' 자차 궤적. research/docs/baseline_spec.md 1장.

실제 자차의 움직임(plant.py)과는 다르다. 예측은 차선 변경을 시간에 선형인 횡이동으로
가정하고(alpha = (tau - tau_m)/D), 자차 yaw 는 차선 방향으로 고정한다.
"""
from dataclasses import dataclass
import math

import numpy as np

from .risk import EgoTrajectory

KEEP, WAIT, CHANGE_NOW, HOLD, YIELD, ACCEL = "keep", "wait", "change_now", "hold", "yield", "accel"
A_UP, A_DN = 2.4, 9.0              # ego_prediction_max_accel/decel_mps2
BLEND_ACC_DN = 4.0                 # change_rollout_blend_cap_decel_mps2
SOFT_ACC_START_D = 2.7             # change_rollout_soft_acc_start_abs_d_m
LAUNCH_SPEED = 1.0                 # change_rollout_launch_speed_threshold_mps
CONNECTOR_ACCEL = 4.0              # stopped_launch_connector_max_accel_mps2


@dataclass
class ActionSpec:
    action: str
    prep_accel: float = 0.0
    t_lc: float = 0.0
    duration: float = 4.0
    change_left: bool = False

    @property
    def label(self):
        if self.action in (YIELD, ACCEL):
            return f"{self.action}({self.prep_accel:+.2f})@{self.t_lc:.1f}"
        if self.action == HOLD:
            return f"hold@{self.t_lc:.1f}"
        return self.action


@dataclass
class RolloutContext:
    """한 판단 주기에서 궤적 생성에 필요한 값."""

    v0: float
    s0: float                 # 자차 후륜축 s (공통 좌표)
    lane: int                 # 현재 차선
    target_lane: int
    ego_d_target: float       # 목표 차선 기준 d (중앙이면 -3.5)
    pre_cap: float            # min(mission, ACC 목표)
    mission: float
    tf_speed: float           # target_front 속도 (없으면 nan)
    D: float                  # 예측 blend 시간 = preview.blend_duration_sec * 1.04
    yield_ref_speed: float    # YIELD floor 기준 속도
    preview_mode: str = "blend"   # 'blend' | 'connector'
    connector_join_arc: float = 20.0


def execution_start(spec, P):
    """effectiveLaneChangeStartSec: t_LC + 0.55."""
    return max(0.0, spec.t_lc) + P.behavior_execution_delay_sec if spec.change_left else math.inf


def fits_horizon(spec, P):
    """candidateFitsHorizon: t_sw + D + 1.5 ≤ 8.0."""
    if not spec.change_left:
        return True
    need = execution_start(spec, P) + max(0.1, spec.duration) + P.behavior_post_merge_buffer_sec
    return need <= P.horizon_steps * P.prediction_dt_sec + 1e-6


# --------------------------------------------------------------- 종방향
def speed_target_rollout(v0, tau, vref):
    """speedTargetRollout: vref 로 +2.4/-9.0 등가속 후 유지. tau 는 배열."""
    v0 = max(0.0, v0)
    vref = max(0.0, vref)
    if vref > v0 + 0.05:
        a = A_UP
    elif vref < v0 - 0.05:
        a = -A_DN
    else:
        return np.full_like(tau, v0), v0 * tau
    t_r = (vref - v0) / a
    v = np.where(tau <= t_r, v0 + a * tau, vref)
    ds = np.where(tau <= t_r, v0 * tau + 0.5 * a * tau * tau,
                  v0 * t_r + 0.5 * a * t_r * t_r + vref * (tau - t_r))
    return v, ds


def bounded_preparation_rollout(v0, tau, a, cap, floor):
    """boundedPreparationRollout: 준비 가속도를 지평선 전체에 유지(기존 SMPC 의 '지속' 가정)."""
    v0 = max(0.0, v0)
    if abs(a) < 1e-9:
        return np.full_like(tau, v0), v0 * tau
    if a > 0:
        if v0 >= cap:
            return np.full_like(tau, cap), cap * tau
        t_r = (cap - v0) / a
        v = np.where(tau <= t_r, v0 + a * tau, cap)
        ds = np.where(tau <= t_r, v0 * tau + 0.5 * a * tau * tau,
                      v0 * t_r + 0.5 * a * t_r * t_r + cap * (tau - t_r))
        return v, ds
    fl = min(max(floor, 0.0), v0)
    t_r = (fl - v0) / a
    v = np.where(tau <= t_r, v0 + a * tau, fl)
    ds = np.where(tau <= t_r, v0 * tau + 0.5 * a * tau * tau,
                  v0 * t_r + 0.5 * a * t_r * t_r + fl * (tau - t_r))
    return v, ds


def post_completion_speed(ctx):
    """changePostCompletionSpeedMps: 목표 차선 앞차가 있으면 그 속도, 없으면 미션 속도."""
    if math.isfinite(ctx.tf_speed):
        return min(ctx.mission, max(0.0, ctx.tf_speed))
    return max(ctx.pre_cap, ctx.mission)


def blend_stage_rollout(ctx, taus, t_sw, D, P):
    """blendStageRollout (CHANGE_NOW). 0.2 s 단위로 pre_cap → v_post 를 쫓는다."""
    d0 = abs(ctx.ego_d_target)
    switch = t_sw + D * (1.0 - SOFT_ACC_START_D / d0) if d0 > SOFT_ACC_START_D else t_sw
    v_post = post_completion_speed(ctx)
    launch = ctx.v0 < LAUNCH_SPEED
    cap_conn = P.stopped_launch_connector_speed_cap_by_target_lane_mps
    v1 = max(ctx.pre_cap, cap_conn[ctx.target_lane]) if launch else ctx.pre_cap
    a_up = CONNECTOR_ACCEL if launch else A_UP
    dt = P.prediction_dt_sec
    v, ds = max(0.0, ctx.v0), 0.0
    out_v, out_ds = [], []
    sigma = 0.0
    for tau in taus:
        while sigma < tau - 1e-9:
            ref = v1 if sigma < switch else v_post
            if ref > v + 0.05:
                vn = min(ref, v + a_up * dt)
            elif ref < v - 0.05:
                vn = max(ref, v - BLEND_ACC_DN * dt)
            else:
                vn = v
            ds += 0.5 * dt * (v + vn)
            v = max(0.0, vn)
            sigma += dt
        out_v.append(v)
        out_ds.append(ds)
    return np.array(out_v), np.array(out_ds)


def action_rollout(ctx, spec, taus, P):
    """actionLongitudinalRollout (+ CHANGE_NOW 는 blendStage)."""
    if spec.action in (KEEP, WAIT):
        return speed_target_rollout(ctx.v0, taus, ctx.pre_cap)
    if spec.action == CHANGE_NOW:
        return blend_stage_rollout(ctx, taus, execution_start(spec, P), spec.duration, P)
    floor = max(0.0, ctx.yield_ref_speed - P.behavior_yield_max_speed_drop_mps) if spec.action == YIELD else 0.0
    cap = ctx.pre_cap if spec.action == ACCEL else ctx.v0
    return bounded_preparation_rollout(ctx.v0, taus, spec.prep_accel, cap, floor)


def connector_launch_rollout(ctx, spec, taus, t_sw, P):
    """connectorLaunchRollout: t_sw 까지는 기본 법칙(CHANGE_NOW 는 KEEP 법칙), 이후 4.0 으로 cap 까지."""
    base_spec = ActionSpec(KEEP) if spec.action == CHANGE_NOW else spec
    v_base, ds_base = action_rollout(ctx, base_spec, taus, P)
    cap = P.stopped_launch_connector_speed_cap_by_target_lane_mps[ctx.target_lane]
    i_sw = np.searchsorted(taus, t_sw - 1e-9)
    v_sw = float(np.interp(t_sw, np.concatenate([[0.0], taus]), np.concatenate([[ctx.v0], v_base])))
    ds_sw = float(np.interp(t_sw, np.concatenate([[0.0], taus]), np.concatenate([[0.0], ds_base])))
    tt = np.maximum(0.0, taus - t_sw)
    v_l, ds_l = bounded_preparation_rollout(v_sw, tt, CONNECTOR_ACCEL, max(cap, v_sw), 0.0)
    v = np.where(np.arange(len(taus)) < i_sw, v_base, v_l)
    ds = np.where(np.arange(len(taus)) < i_sw, ds_base, ds_sw + ds_l)
    return v, ds


def connector_travel_time(v0, dist, target_lane, P):
    """connectorTravelTimeSec: 4.0 m/s² 로 cap 까지 가속하며 dist 를 가는 시간."""
    cap = max(P.stopped_launch_connector_speed_cap_by_target_lane_mps[target_lane], v0, 0.1)
    v0 = max(0.0, v0)
    t_acc = (cap - v0) / CONNECTOR_ACCEL
    d_acc = v0 * t_acc + 0.5 * CONNECTOR_ACCEL * t_acc * t_acc
    if dist <= d_acc:
        return (-v0 + math.sqrt(v0 * v0 + 2 * CONNECTOR_ACCEL * dist)) / CONNECTOR_ACCEL
    return t_acc + (dist - d_acc) / cap


# ------------------------------------------------------------- 궤적
def build_trajectories(ctx, specs, track, P):
    """후보 목록 → EgoTrajectory [N, K]. 빈 궤적은 valid=False."""
    K, dt, W = P.horizon_steps, P.prediction_dt_sec, track.lane_width
    taus = dt * np.arange(1, K + 1)
    N = len(specs)
    x = np.zeros((N, K))
    y = np.zeros((N, K))
    v = np.zeros((N, K))
    mask = np.ones((N, K), dtype=bool)
    valid = np.ones(N, dtype=bool)
    y_cur, y_tgt = track.center_y(ctx.lane), track.center_y(ctx.target_lane)
    src_end = track.csv_end[ctx.lane]
    tgt_end = track.csv_end.get(ctx.target_lane, math.inf)
    lookahead = P.get("ego_reference_preview_lookahead_m", 220.0)
    for i, spec in enumerate(specs):
        if not spec.change_left:
            vi, dsi = action_rollout(ctx, spec, taus, P)
            s = ctx.s0 + dsi
            if np.any(s > src_end + 1e-6):           # KEEP 은 차선 끝을 넘으면 빈 궤적
                valid[i] = False
            x[i], v[i] = s, vi
            y[i] = y_cur + (ctx.ego_d_target + W) * np.exp(-taus / max(0.1, P.lane_center_decay_tau_sec))
            continue
        t_sw = execution_start(spec, P)
        if ctx.preview_mode == "connector":
            vi, dsi = connector_launch_rollout(ctx, spec, taus, t_sw, P)
        else:
            vi, dsi = action_rollout(ctx, spec, taus, P)
        ds_sw = float(np.interp(t_sw, np.concatenate([[0.0], taus]), np.concatenate([[0.0], dsi])))
        if ctx.preview_mode == "connector":
            # smpc:2507-2530 정지 connector 는 출발 전 0.5 m 이내, 연결 완주 + 1.5 s 가 지평선 안
            if ds_sw > P.stopped_launch_connector_max_preswitch_distance_m:
                valid[i] = False
                continue
            v_sw = float(np.interp(t_sw, np.concatenate([[0.0], taus]), np.concatenate([[ctx.v0], vi])))
            if t_sw + connector_travel_time(v_sw, ctx.connector_join_arc, ctx.target_lane, P) + \
                    P.behavior_post_merge_buffer_sec > K * dt + 1e-6:
                valid[i] = False
                continue
        moving = (taus >= t_sw - 1e-6) & ((vi >= P.ego_reference_preview_motion_start_speed_mps) |
                                           (dsi - ds_sw >= P.ego_reference_preview_motion_start_distance_m))
        if not moving.any():
            valid[i] = False
            continue
        tau_m = taus[np.argmax(moving)]
        s = ctx.s0 + dsi
        if ctx.preview_mode == "connector":
            u = np.clip((dsi - ds_sw) / max(1.0, ctx.connector_join_arc), 0.0, 1.0)
            alpha = np.where(taus >= tau_m, u * u * u * (10.0 - 15.0 * u + 6.0 * u * u), 0.0)
        else:
            alpha = np.clip((taus - tau_m) / spec.duration, 0.0, 1.0)
        yi = y_cur + W * alpha
        yi = np.where(s > src_end, y_tgt, yi)          # 원 차선 polyline 끝 뒤는 목표 차선 점
        covered = (s - ctx.s0 <= lookahead) & (s <= tgt_end)
        if not covered.all():
            last = np.argmin(covered) - 1              # 처음 벗어나기 직전 스텝
            if last < 0 or taus[last] < t_sw + spec.duration + P.behavior_post_merge_buffer_sec - 1e-6:
                valid[i] = False
                continue
            mask[i, last + 1:] = False
            terminal = last
        else:
            terminal = K - 1
        if abs(yi[terminal] - y_tgt) > P.ego_reference_preview_target_completion_max_abs_d_m:
            valid[i] = False
            continue
        x[i], y[i], v[i] = s, yi, vi
    return EgoTrajectory(t=taus, x=x, y=y, yaw=np.zeros((N, K)), v=v, valid=valid, step_mask=mask)
