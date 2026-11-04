"""반응형 예측 (research/docs/plan_v2.md 단계 1·2). 자차와 타겟 차량을 0.2 s 단위로 함께 굴린다.

자차 = ACC 폐루프. 매 스텝 ACC 법칙(acc.py 의 앞차 추종 식, 차선 끝 정지, 차선 변경 중 차선 끝 상한,
  준비 행동 속도 상한)으로 목표 속도를 내고, 변화율 필터(+4.0 / −7.0 m/s²)를 거쳐
  1 차 응답(plant.py 와 같은 식)으로 실제 속도를 만든다. 기존 예측의 '합류 뒤 속도 = min(미션, 앞차)'
  고정값과 '연결 경로 상한까지 앞차를 무시하고 가속'을 대신한다.
타겟 차량 = 반응 모델. 가지(branch) = (뒤차 mode, 앞차 mode).
  뒤차  yield    : 자차 차체가 목표 차선에 rear_react_intrusion_m 이상 들어오고 rear_react_delay_s 가 지나면
                   자차를 앞차로 보고 IDM 으로 따라간다(감속 한계 rear_max_brake). 그 전에는 자기 앞차/등속.
        nonyield : 자차를 무시한다(자기 앞차/등속).
  앞차  base     : 자기 앞차가 보이면 IDM, 아니면 현재 가속도가 accel_decay_s 로 사라지는 등가속.
        brake    : front_brake m/s² 로 정지까지.     accel : front_accel m/s² (현재 속도 + 5 m/s 까지).
예측용 반응 파라미터는 시뮬 NPC(npc.py) 값과 일부러 다르게 둔다(예측 모델을 자기 자신으로 시험하지 않기).
단계 5 의 MORAI 측정으로 바꾼다.

대응: 0 = 계속, 1 = 복귀(가지가 드러나는 시점 t_b 부터 측정 복귀 곡선으로 원 차선, 이후 ACC 는 원 차선 기준).
행 순서: (가지 b, 대응 r, 루트 n) 을 한 배열로 쌓아 한 번에 굴린다.
"""
from dataclasses import dataclass
import math

import numpy as np

from .params import load_params
from .risk import collision_probability
from .rollouts import ACCEL, HOLD, YIELD, execution_start

CONTINUE, RETURN = 0, 1
BRAKING_DECEL = 3.8                      # acc.py speed_for_lead


@dataclass
class ReactiveParams:
    # 자차 응답 (sim/plant.py EgoPlant 와 같은 값. MORAI 맞춤: τ 0.4, 가속 ≤ 2.5 — results/m6_ego_response.json)
    plant_tau: float = 0.7
    plant_a_max: float = 3.0
    plant_a_min: float = -6.0
    plant_a_stop: float = -9.0
    plant_jerk: float = 12.0
    acc_up: float = 4.0
    acc_dn: float = 7.0
    mission_ramp: float = 0.0            # 시뮬은 미션 속도 일정. MORAI 는 차선 전환 뒤 약 2.5 m/s² 로 회복
    cruise_max: float = 0.0              # 회복 상한(0 이면 지금 미션 속도). MORAI: 차선 끝 서행 3.15 m/s → 순항
    lane_end_decel: float = 2.5          # acc.py 차선 끝 정지 한계 감속
    # 뒤차 반응 (임시값, 단계 5 측정 전)
    rear_react_intrusion_m: float = 0.3
    rear_react_delay_s: float = 1.0
    rear_detect_s: float = 0.5           # 반응 지연이 지난 뒤 '감속 없음'을 알아채는 데 걸리는 시간
    rear_idm_s0: float = 3.0
    rear_idm_T: float = 1.0
    rear_idm_a: float = 1.0
    rear_idm_b: float = 2.0
    rear_max_brake: float = 3.0
    # 앞차
    front_brake: float = -2.5
    front_accel: float = 1.0
    front_detect_s: float = 0.3
    accel_decay_s: float = 1.5
    front_idm_s0: float = 3.0
    front_idm_T: float = 1.0
    front_idm_a: float = 1.0
    front_idm_b: float = 2.0
    # 예외 확률 최솟값
    p_exc_floor: float = 0.05
    # 자차가 ACC 로 따라가는 앞차는 위치 불확실성 σ 의 시간 인자를 이 값에서 멈춘다: 앞차가 예측과 다르게
    # 움직여도 자차가 반응(피드백)하므로 열린 루프처럼 σ 가 끝없이 커지지 않는다(Nair 외 TCST 2025 의 상태 피드백과 같은 취지).
    follow_sigma_cap_s: float = 1.0
    abort_before_abs_d: float = 1.75     # 목표 차선 경계(이 안쪽이면 복귀 불가, supervisor 와 같다)


def _idm(v, gap, v_lead, v0, a, b, s0, T):
    gap = np.maximum(gap, 0.1)
    s_star = s0 + np.maximum(0.0, v * T + v * (v - v_lead) / (2.0 * math.sqrt(a * b)))
    return a * (1.0 - (v / np.maximum(v0, 0.5)) ** 4 - (s_star / gap) ** 2)


def speed_for_lead(gap, v_lead, v_e, cruise, A):
    """acc.py AccPlanner.speed_for_lead 의 벡터판(앞차 추종 목표 속도). hard 여부도 낸다."""
    vL = np.minimum(np.maximum(v_lead, 0.0), cruise)
    c = np.maximum(0.0, v_e - vL)
    g_star = A.standstill_gap_m + A.time_headway_sec * v_e + A.closing_speed_gap_time_sec * c + c * c / (2 * BRAKING_DECEL)
    cmd = np.maximum(vL + A.gap_gain * (gap - g_star), vL - A.max_follow_speed_drop_mps)
    cmd = np.minimum(cmd, np.sqrt(vL * vL + 2 * BRAKING_DECEL * np.maximum(0.0, gap - A.emergency_gap_m)))
    closing = v_e - np.maximum(0.0, v_lead)
    ttc = np.where(closing > 0.1, gap / np.maximum(closing, 1e-6), np.inf)
    cmd = np.where(closing > 0.1, np.minimum(cmd, vL + np.maximum(0.0, gap - A.emergency_gap_m) / A.soft_ttc_sec), cmd)
    hard = (closing > 0.1) & (ttc <= A.hard_ttc_sec)
    cmd = np.where(hard, np.minimum(cmd, vL), cmd)
    zero = (gap <= A.critical_gap_m) | ((closing > 0.1) & (ttc <= A.critical_ttc_sec))
    cmd = np.where(zero, 0.0, cmd)
    return np.clip(cmd, 0.0, cruise), hard | zero


class _Tv:
    def __init__(self, v, M, role, accel0):
        self.uid, self.role, self.lane = v.unique_id, role, v.lane_id
        self.len, self.wid, self.d0 = v.length, v.width, v.d
        self.s = np.full(M, float(v.s))
        self.v = np.full(M, max(0.0, v.v_long))
        self.v0 = max(0.0, v.v_long)
        self.a0 = accel0
        self.lead = None
        self.out_s = []


def target_lane_vehicles(t, vehicles):
    """(현재 차선 앞차, 목표 차선 앞차, 목표 차선 뒤·옆 차량들). G7 이 보던 차량 전부를 트리가 본다."""
    cf = t.current_front if t.current_front.valid else None
    tf = t.target_front if t.target_front.valid else None
    # 목표 차선 앞차 = target_front 와 목표 차선에서 자차 앞(rear_delta_s > 0)에 있는 차 중 가장 가까운 차
    # (선택기가 차선을 넘나드는 차를 target_front 에서 잠깐 빼도 예측에서 빠지지 않게. C++ rxVehicles 와 같다).
    for v in vehicles:
        if v.valid and v.lane_id == t.target_lane_id and v.rear_delta_s > 0.0 and (tf is None or v.delta_s < tf.delta_s):
            tf = v
    seen = {tf.unique_id} if tf else set()
    rears = []
    for v in vehicles:
        if not v.valid or v.unique_id in seen or v.lane_id != t.target_lane_id or v.rear_delta_s > 0.0:
            continue
        if tf is not None and v.s >= tf.s:
            continue
        seen.add(v.unique_id)
        rears.append(v)
    rears.sort(key=lambda v: -v.s)
    return cf, tf, rears


def joint_rollout(ctx, specs, t, vehicles, P, A, RP, track, branches, accel_of, mid_T_rem=None):
    """branches: [(rear_mode, front_mode, tb_rule)], tb_rule ∈ {None, 'rear', 'front'} 은 복귀 대응의 분기 시점.
    mid_T_rem: 차선 변경 도중(모니터)이면 남은 횡이동 시간(blend) — 루트 0 = 지금부터 계속, 대응 1 = 지금 복귀.
    반환: risk [B,2,N,K], allowed [B,2,N], x/y/v [B,2,N,K], valid [N], mask [N,K]."""
    from .tree import RETURN_PROGRESS as RET_P, RETURN_PROGRESS_T as RET_T
    K, dt, W = P.horizon_steps, P.prediction_dt_sec, track.lane_width
    taus = dt * np.arange(1, K + 1)
    N, R, B = len(specs), 2, len(branches)
    M = B * R * N
    mid = mid_T_rem is not None
    y_src, y_tgt = track.center_y(ctx.lane), track.center_y(ctx.target_lane)
    src_end = track.csv_end[ctx.lane]
    tgt_end = track.csv_end.get(ctx.target_lane, math.inf)
    half_w = 0.5 * P.ego_width_m
    cruise = ctx.mission
    connector = ctx.preview_mode == "connector"

    rep = lambda arr: np.tile(np.asarray(arr), B * R)
    change = rep([s.change_left for s in specs])
    t_sw = rep([0.0 if mid else execution_start(s, P) for s in specs])
    dur = rep([max(0.1, s.duration) for s in specs])
    act = rep(np.array([s.action for s in specs], dtype=object))
    prep_a = rep([s.prep_accel for s in specs])
    resp = np.tile(np.repeat(np.arange(R), N), B)
    br = np.repeat(np.arange(B), R * N)
    rear_mode = np.array([branches[b][0] for b in br])
    front_mode = np.array([branches[b][1] for b in br])
    tb_rule = np.array([branches[b][2] or "" for b in br])
    yield_floor = max(0.0, ctx.yield_ref_speed - P.behavior_yield_max_speed_drop_mps)

    s = np.full(M, ctx.s0)
    v = np.full(M, max(0.0, ctx.v0))
    a = np.zeros(M)
    f = v.copy()
    ref_prev = f.copy()
    y0 = y_tgt + ctx.ego_d_target
    y = np.full(M, y0)
    ds_sw = np.where(mid & change, 0.0, np.nan)
    tau_m = np.where(mid & change, 0.0, np.nan)
    ret_t = np.full(M, np.inf)
    y_ret0 = np.zeros(M)
    t_in = np.full(M, np.inf)
    L_rem = max(2.0, ctx.connector_join_arc * abs(ctx.ego_d_target) / W) if mid else max(1.0, ctx.connector_join_arc)

    cf, tf, rears = target_lane_vehicles(t, vehicles)
    tvs = ([_Tv(cf, M, "cf", accel_of(cf))] if cf else []) + ([_Tv(tf, M, "tf", accel_of(tf))] if tf else []) + \
        [_Tv(r, M, "rear", accel_of(r)) for r in rears]
    ctx_rows = []
    for x in [o for o in tvs if o.role in ("cf", "tf")]:
        ahead = [w for w in vehicles if w.valid and w.lane_id == x.lane and w.s > x.s[0] + 1e-3 and w.unique_id != x.uid]
        if ahead:
            c = _Tv(min(ahead, key=lambda w: w.s), M, "ctx", 0.0)
            ctx_rows.append(c)
            x.lead = c
    for x in tvs:
        if x.role == "rear":
            ahead = [o for o in tvs if o is not x and o.lane == x.lane and o.s[0] > x.s[0]]
            x.lead = min(ahead, key=lambda o: o.s[0]) if ahead else None

    out_x, out_y, out_v = np.zeros((M, K)), np.zeros((M, K)), np.zeros((M, K))
    tf_hard_follow = []
    for k, tau in enumerate(taus):
        tau0 = tau - dt
        # ---- 횡 위치 -------------------------------------------------------------------
        moved = s - ctx.s0
        if not mid:
            start = change & np.isnan(ds_sw) & (tau0 >= t_sw - 1e-6)
            ds_sw = np.where(start, moved, ds_sw)
            motion = change & ~np.isnan(ds_sw) & np.isnan(tau_m) & (
                (v >= P.ego_reference_preview_motion_start_speed_mps) |
                (moved - np.nan_to_num(ds_sw) >= P.ego_reference_preview_motion_start_distance_m))
            tau_m = np.where(motion, tau0, tau_m)
        if connector:
            u = np.clip((moved - np.nan_to_num(ds_sw)) / L_rem, 0.0, 1.0)
            alpha = np.where(np.isnan(tau_m), 0.0, u * u * u * (10.0 - 15.0 * u + 6.0 * u * u))
        else:
            T = mid_T_rem if mid else dur
            alpha = np.where(np.isnan(tau_m), 0.0, np.clip((tau0 - np.nan_to_num(tau_m)) / T, 0.0, 1.0))
        y_start = y0 if mid else y_src
        y_go = np.where(change, y_start + (y_tgt - y_start) * alpha,
                        y_src + (y0 - y_src) * math.exp(-tau0 / max(0.1, P.lane_center_decay_tau_sec)))
        y_go = np.where(change & (s > src_end), y_tgt, y_go)
        tb = np.where(tb_rule == "front", RP.front_detect_s,
             np.where(tb_rule == "rear", t_in + RP.rear_react_delay_s + RP.rear_detect_s, np.inf))
        if mid:
            tb = np.where(tb_rule != "", 0.0, tb)
        begin = (resp == RETURN) & change & np.isinf(ret_t) & (tau0 >= tb - 1e-9) & (np.abs(y - y_tgt) > RP.abort_before_abs_d)
        if k == 0:
            begin &= np.abs(y0 - y_tgt) > RP.abort_before_abs_d
        ret_t = np.where(begin, tau0, ret_t)
        y_ret0 = np.where(begin, y if k else y0, y_ret0)
        returning = np.isfinite(ret_t)
        scale = np.maximum(0.35, np.sqrt(np.minimum(1.0, np.abs(y_ret0 - y_src) / W)))
        fr = np.interp(np.maximum(0.0, tau0 - np.where(returning, ret_t, tau0)) / scale, RET_T, RET_P)
        y = np.where(returning, y_ret0 + (y_src - y_ret0) * fr, y_go)
        d_t = np.abs(y - y_tgt)
        intr = 0.5 * W + half_w - d_t
        t_in = np.where(np.isinf(t_in) & (intr >= RP.rear_react_intrusion_m), tau0, t_in)
        in_target = d_t < 0.5 * W
        lc_active = change & (tau0 >= t_sw) & ~returning

        # ---- 타겟 차량 -----------------------------------------------------------------
        new = []
        for x in tvs:
            if x.lead is not None:
                gap_l = x.lead.s - 0.5 * x.lead.len - x.s - 0.5 * x.len
                if x.role == "rear":
                    base = _idm(x.v, gap_l, x.lead.v, max(x.v0, 0.5), RP.rear_idm_a, RP.rear_idm_b, RP.rear_idm_s0, RP.rear_idm_T)
                else:
                    base = _idm(x.v, gap_l, x.lead.v, max(x.v0, 0.5), RP.front_idm_a, RP.front_idm_b, RP.front_idm_s0, RP.front_idm_T)
            else:
                base = np.full(M, x.a0 * math.exp(-tau0 / RP.accel_decay_s))
            if x.role in ("cf", "tf"):
                acc = np.where(front_mode == "brake", RP.front_brake,
                      np.where(front_mode == "accel", np.where(x.v < x.v0 + 5.0, RP.front_accel, 0.0), base))
            else:
                react = (rear_mode == "yield") & (tau0 >= t_in + RP.rear_react_delay_s) & (intr > 0.0) & (s > x.s)
                gap_e = (s - P.ego_rear_extent_m) - (x.s + 0.5 * x.len)
                a_e = _idm(x.v, gap_e, v, max(x.v0, 0.5), RP.rear_idm_a, RP.rear_idm_b, RP.rear_idm_s0, RP.rear_idm_T)
                acc = np.maximum(np.where(react, np.minimum(base, a_e), base), -RP.rear_max_brake)
            vn = np.maximum(0.0, x.v + acc * dt)
            new.append((x, x.s + 0.5 * (x.v + vn) * dt, vn))

        # ---- 자차 ACC 목표 속도 (acc.py AccPlanner.step 과 같은 규칙) --------------------------
        # 미션 속도: 차선 전환(t_sw) 뒤부터 mission_ramp 로 cruise_max 까지 회복(MORAI 대회 스택). 시뮬은 일정.
        if RP.mission_ramp > 0 and RP.cruise_max > 0:
            cruise_k = np.minimum(RP.cruise_max, cruise + RP.mission_ramp * np.maximum(0.0, tau0 - np.where(change, t_sw, np.inf)))
        else:
            cruise_k = np.full(M, cruise)
        raw = cruise_k.copy()
        hard = np.zeros(M, dtype=bool)
        soft_bind = np.zeros(M, dtype=bool)
        bind_c = np.full(M, np.nan)
        bind_g = np.full(M, np.nan)
        c_max = np.zeros(M)
        for x in tvs:
            if x.role == "cf":
                use, allow_hard = ~in_target | returning, np.ones(M, dtype=bool)
            elif x.role == "tf":
                # 목표 차선 앞차: 중앙 0.75 m 안(또는 이미 그 차선)이면 정상 추종, 차선 변경 중 2.7 m 안이면
                # 약한 추종(감속 ≤ 1.5 m/s², hard 없음) — acc.py candidates
                entry = d_t <= A.target_front_acc_entry_abs_d_m
                use = entry | (lc_active & (d_t <= A.target_front_overlap_soft_acc_start_abs_d_m))
                allow_hard = entry
            else:
                continue
            gap = (x.s - 0.5 * x.len) - (s + P.ego_front_extent_m)
            ahead = gap > -x.len
            cmd, hd = speed_for_lead(gap, x.v, v, cruise_k, A)
            c = np.maximum(0.0, v - np.minimum(np.maximum(x.v, 0.0), cruise_k))
            closing = v - np.maximum(0.0, x.v)
            rapid = (A.rapid_lead_brake_enabled & (c >= A.rapid_lead_min_closing_speed_mps) &
                     ((gap / np.maximum(closing, 1e-6) <= A.rapid_lead_ttc_sec) |
                      ((gap > A.emergency_gap_m) & (c * c / (2 * np.maximum(gap - A.emergency_gap_m, 1e-6)) >= 5.0))))
            act_l = use & ahead
            c_max = np.where(act_l, np.maximum(c_max, c), c_max)
            take = act_l & (cmd < raw)
            raw = np.where(take, cmd, raw)
            soft_bind = np.where(take, ~allow_hard, soft_bind)
            bind_c = np.where(take, c, bind_c)
            bind_g = np.where(take, gap, bind_g)
            hard |= act_l & (hd | rapid) & allow_hard
        dist = src_end - s
        stay = ~lc_active & ~in_target & (dist < 200.0)
        lim_stay = np.sqrt(2.0 * RP.lane_end_decel * np.maximum(0.0, dist - A.lane_end_merge_wait_stop_distance_m))
        lim = stay & (lim_stay < raw)
        raw = np.where(lim, lim_stay, raw)
        soft_bind &= ~lim
        bind_c = np.where(lim, np.nan, bind_c)
        elapsed = np.maximum(0.0, tau0 - t_sw)
        lim_lc = np.minimum(np.maximum((dist - A.lane_change_lane_end_margin_m) / np.maximum(0.5, 4.2 - elapsed),
                                       np.minimum(A.lane_change_lane_end_min_cap_mps, cruise_k)), cruise_k)
        lim = lc_active & (d_t > A.lane_change_lane_end_cap_release_abs_d_m) & (lim_lc < raw)
        raw = np.where(lim, lim_lc, raw)
        soft_bind &= ~lim
        bind_c = np.where(lim, np.nan, bind_c)
        pre = ~(change & (tau0 >= t_sw))
        cap = np.where(act == HOLD, ctx.v0,
              np.where(act == YIELD, np.maximum(yield_floor, ctx.v0 + prep_a * tau0),
              np.where(act == ACCEL, np.minimum(cruise_k, ctx.v0 + prep_a * tau0), np.inf)))
        lim = pre & (cap < raw)
        raw = np.where(lim, cap, raw)
        soft_bind &= ~lim
        bind_c = np.where(lim, np.nan, bind_c)
        # 필터: 감속 한계 = hard 12 / 기본 7 + 접근 속도 가산(최대 11), 약한 추종 1.5, 앞차 요구 감속 한계
        boost = np.clip((c_max - A.closing_decel_boost_start_mps) /
                        (A.closing_decel_boost_full_mps - A.closing_decel_boost_start_mps), 0.0, 1.0)
        dec = np.where(hard, 12.0, RP.acc_dn + boost * (11.0 - RP.acc_dn))
        dec = np.where(soft_bind, np.minimum(dec, 1.5), dec)
        if A.lead_required_decel_limit_enabled:
            margin = bind_g - A.emergency_gap_m - A.lead_required_decel_limit_headway_sec * v
            lim = ~np.isnan(bind_c) & ~hard & (margin > 0.5)
            req = np.maximum(1.5, A.lead_required_decel_limit_gain * np.nan_to_num(bind_c) ** 2 /
                             (2 * np.maximum(np.nan_to_num(margin), 1e-6)))
            dec = np.where(lim, np.minimum(dec, req), dec)
        bypass = hard & A.hard_brake_bypass_filter
        f = np.where(bypass, raw, np.clip(raw, f - dec * dt, f + RP.acc_up * dt))
        f = np.clip(f, 0.0, cruise_k)
        tf_hard_follow.append(np.zeros(M, dtype=bool) if tf is None else (d_t <= A.target_front_acc_entry_abs_d_m))

        # ---- 자차 응답 -----------------------------------------------------------------
        ff = (f - ref_prev) / dt
        ref_prev = f
        a_des = np.clip((f - v) / RP.plant_tau + ff, np.where(f < 0.01, RP.plant_a_stop, RP.plant_a_min), RP.plant_a_max)
        a = a + np.clip(a_des - a, -RP.plant_jerk * dt, RP.plant_jerk * dt)
        a = np.where((v <= 0.0) & (a < 0.0), 0.0, a)
        vn = np.maximum(0.0, v + a * dt)
        s = s + 0.5 * (v + vn) * dt
        v = vn
        for x, sn, vx in new:
            x.s, x.v = sn, vx
            x.out_s.append(sn)
        for c in ctx_rows:
            c.s = c.s + c.v * dt
        out_x[:, k], out_y[:, k], out_v[:, k] = s, y, v

    # ---- 위험 ----------------------------------------------------------------------------
    no_coll = np.ones((M, K))
    for x in tvs:
        tx = np.stack(x.out_s, axis=1)
        ty = (track.center_y(x.lane) + x.d0 * np.exp(-taus / max(0.1, P.lane_center_decay_tau_sec)))[None, :]
        front = x.role in ("cf", "tf")
        brake_rows = (front_mode == "brake")[:, None] if front else np.zeros((M, 1), dtype=bool)
        if x.role == "cf":
            t_sig = np.minimum(taus, RP.follow_sigma_cap_s)[None, :]
        elif x.role == "tf":
            t_sig = np.where(np.stack(tf_hard_follow, axis=1), np.minimum(taus, RP.follow_sigma_cap_s)[None, :], taus[None, :])
        else:
            t_sig = taus[None, :]
        cn, _ = collision_probability(out_x, out_y, 0.0, t_sig, tx, ty, 0.0, x.len, x.wid, "nominal", P)
        if brake_rows.any():
            cb, _ = collision_probability(out_x, out_y, 0.0, t_sig, tx, ty, 0.0, x.len, x.wid, "front_brake", P)
            cn = np.where(brake_rows, cb, cn)
        alive = tx <= track.csv_end.get(x.lane, math.inf) + 1e-6
        no_coll *= 1.0 - np.where(alive, cn, 0.0)
    risk = 1.0 - no_coll

    shp = lambda z: z.reshape(B, R, N, *z.shape[1:])
    X, Y = shp(out_x), shp(out_y)
    # 유효성(G2): 계속 대응·첫 가지 기준. 차선 변경 루트는 움직이기 시작해 지평선 끝에 목표 차선에 있어야 하고
    # (목표 차선 CSV 가 먼저 끝나면 그 끝까지), 정지 connector 는 출발 전 0.5 m 이내. 원 차선 유지는 차선 끝 전.
    x0, y0r = X[0, 0], Y[0, 0]
    chg = np.array([sp.change_left for sp in specs])
    valid = np.ones(N, dtype=bool)
    if not mid:
        valid &= ~chg | ~np.isnan(tau_m[:N])
        if connector:
            valid &= ~chg | (np.nan_to_num(ds_sw[:N], nan=0.0) <= P.stopped_launch_connector_max_preswitch_distance_m)
    valid &= ~chg | (np.abs(y0r[:, -1] - y_tgt) <= P.ego_reference_preview_target_completion_max_abs_d_m) | (x0[:, -1] > tgt_end)
    valid &= chg | (x0[:, -1] <= src_end + 1e-6)
    mask = x0 <= tgt_end + 1e-6
    risk = shp(risk) * mask[None, None]
    allowed = shp(np.where(resp == RETURN, np.isfinite(ret_t), True))
    return {"risk": risk, "allowed": allowed, "x": X, "y": Y, "v": shp(out_v), "valid": valid, "mask": mask,
            "taus": taus}


def load_acc_params():
    return load_params("acc_planner")
