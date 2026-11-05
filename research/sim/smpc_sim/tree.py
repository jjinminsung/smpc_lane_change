"""시나리오 트리 플래너 (연구 방향 1). 기존 SMPC 와 같은 루트 후보를 쓰되, mode 가 드러난 뒤의
대응까지 함께 평가한다.

트리 한 개 = 루트 행동 a0 하나.
  가지 j : 정상 / 뒤차 가속(시작 0·1·2·3 s) / 앞차 급제동.  확률 p_j.
  분기 시점 t_b(j) = 사건 시작 + 감지 지연.  그 전까지는 모든 가지가 a0 를 똑같이 실행한다.
  대응 r : 계속 / 계속+가속 / 복귀 / 복귀+감속 / 제동.  t_b 이후 궤적만 바뀐다.
  가지 j 에서 대응 r 을 했을 때의 스텝별 충돌 확률 r_j(r, k), 비용 J_j(r).
루트 a0 의 값 = min over (가지별 대응 조합 c)  Σ_j p_j [J_j(c_j) + V(잎)]
              s.t. 모든 스텝 k 에서 Σ_j p_j r_j(c_j, k) ≤ ε.
대응이 모두 이산이라 조합을 전부 열거하면 근사 없이 풀린다(위험 배분이 조합 선택 안에서 정해진다).
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np

from .baseline import BaselineSmpc, Decision, gap_penalty, waypoint_blend_duration, _row
from .acc import BehaviorRequest
from .gates import (conservative_rear_closing, cutin_interlock_safe, immediate_gap_gate,
                    projected_gap_ttc_safe, target_front_is_passing, target_rear_for_gap,
                    target_rear_required_gap)
from .prediction import TvMode, front_bumper_gap, rear_bumper_gap, adaptive_event_probability
from .risk import EgoTrajectory, evaluate_mode
from .rollouts import (ACCEL, CHANGE_NOW, HOLD, KEEP, YIELD, ActionSpec, RolloutContext,
                       build_trajectories, execution_start, fits_horizon)

CONTINUE, CONT_ACCEL, RETURN, RETURN_DECEL, BRAKE = range(5)
RESPONSE_NAMES = ["continue", "continue+accel", "return", "return+decel", "brake"]

# 복귀 횡이동 진행률 p(τ/scale): plant.py 의 MORAI 측정 곡선(PCHIP)을 0.1 s 로 표로 만든 것(C++ 과 같은 표).
# scale = max(0.35, √(복귀 거리/차선 폭)). 선형 복귀(예전 모델)는 실제보다 2–3 배 빨리 빠져나간다(리뷰 #8).
RETURN_PROGRESS_DT = 0.1
RETURN_PROGRESS = np.array([
    0.0000, 0.0005, 0.0019, 0.0041, 0.0068, 0.0100, 0.0151, 0.0233, 0.0339, 0.0464, 0.0600, 0.0770, 0.0992,
    0.1255, 0.1551, 0.1867, 0.2194, 0.2521, 0.2838, 0.3135, 0.3400, 0.3641, 0.3871, 0.4093, 0.4309, 0.4521,
    0.4732, 0.4943, 0.5156, 0.5375, 0.5600, 0.5833, 0.6073, 0.6317, 0.6563, 0.6811, 0.7058, 0.7302, 0.7541,
    0.7775, 0.8000, 0.8219, 0.8432, 0.8634, 0.8825, 0.9026, 0.9228, 0.9419, 0.9586, 0.9717, 0.9800, 0.9853,
    0.9901, 0.9941, 0.9973, 0.9993, 1.0000])
RETURN_PROGRESS_T = RETURN_PROGRESS_DT * np.arange(len(RETURN_PROGRESS))


def return_progress(tau, dist, lane_width):
    """복귀 시작 후 τ 초의 진행률(0→1). dist = 복귀할 횡거리."""
    scale = max(0.35, math.sqrt(min(1.0, abs(dist) / lane_width)))
    return np.interp(np.asarray(tau) / scale, RETURN_PROGRESS_T, RETURN_PROGRESS)


@dataclass
class TreeConfig:
    onsets: tuple = (0.0, 1.0, 2.0, 3.0)
    rear_detect_delay: float = 0.5      # GT 49 Hz: 1 m/s² 가속이 0.5 를 넘는 데 0.29 s, 1.0 은 0.56 s (M4)
    front_detect_delay: float = 0.3
    escape_accel: float = 1.5
    return_time_full: float = 3.1       # 차선 폭 하나 복귀(M2 중앙값)
    return_decel: float = -1.5
    brake_decel: float = -3.0
    abort_before_abs_d: float = 1.75    # 목표 차선 경계(supervisor 와 같다)
    cost_response: tuple = (0.0, 1.0, 30.0, 35.0, 5.0)
    risk_weight: float = 80.0
    probability_table: str = "yaml"     # 'yaml' (기존 SMPC) | 'gt' (GT 입력으로 다시 맞춘 표)
    keep_gates: bool = True             # False 면 연구 방향 2: G3/G5 제거, hard safety 만
    abort_risk_factor: float = 2.0      # 실행 중 복귀 판단 문턱 = 2ε
    monitor_tree: bool = True           # 반응형: 실행 중 계속 위험도 승인처럼 가지별 나중 대응을 넣어 푼다(C++ reactive/monitor_tree)
    use_value: bool = False
    wait_root: bool = False             # V 가 있을 때 '기다리기' 루트를 함께 비교(차선 끝 비용 대신 V)
    # 실행 계층이 실제로 하는 대응만 쓴다(리뷰 #8): 계속(ACC 그대로), 복귀(supervisor abort),
    # 제동(ACC 앞차 추종). 계속+가속·복귀+감속은 아무도 명령하지 않는다.
    executable_only: bool = True
    decision: str = "tree"              # 'baseline' = 기존 SMPC 판단 + 트리 모니터(2×2 비교용)
    monitor: bool = True                # False = 실행 중 재계획 대신 기존 rear_abort
    plant_return: bool = True           # 복귀 횡이동 = 측정 진행률 곡선. False 면 예전 선형 모델


# research/results/day1_m4.json: GT 49 Hz 필터로 다시 잰 '5 s 뒤 12.5 m 이상 앞섬' 사건 비율.
GT_REAR_TABLE = [0.006, 0.007, 0.079, 0.275, 0.672]


class TreePlanner(BaselineSmpc):
    name = "tree"

    def __init__(self, P, track, cfg=None, value_fn=None):
        super().__init__(P, track)
        self.cfg = cfg or TreeConfig()
        self.value_fn = value_fn
        self.last_tree = None
        if self.cfg.probability_table == "gt":
            P.rear_accel_probability_by_accel = list(GT_REAR_TABLE)
            P.min_rear_accel_probability = 0.0
            self.name = "tree_gt"
        if not self.cfg.keep_gates:
            self.name += "_nogate"
        if value_fn is not None:
            self.name += "_Vw" if self.cfg.wait_root else "_V"
        self.keep_gates = self.cfg.keep_gates          # decision='baseline' 일 때 BaselineSmpc 가 읽는다
        self.use_monitor = self.cfg.monitor

    # ------------------------------------------------------- 가지 구성
    def branches(self, t, v_e, R):
        """[(이름, 확률, t_b, {uid: TvMode})]. 확률 합 = 1. 다른 차량은 nominal."""
        P, c = self.P, self.cfg
        out = []
        rear = R if R is not None else (t.target_rear if t.target_rear.valid else None)
        p_rear = adaptive_event_probability(rear, self.hist.observed_accel(rear), v_e, P) if rear is not None else 0.0
        fronts = [v for v in (t.target_front, t.current_front) if v.valid]
        p_front = max([adaptive_event_probability(v, self.hist.observed_accel(v), v_e, P) for v in fronts] or [0.0])
        p_rear = min(p_rear, 0.95)
        p_front = min(p_front, 0.95 - p_rear)
        out.append(("nominal", 1.0 - p_rear - p_front, math.inf, {}))
        if rear is not None and p_rear > 1e-6:
            already = self.hist.observed_accel(rear) >= 0.5
            onsets = (0.0,) if already else c.onsets
            for o in onsets:
                mode = TvMode("rear_accel", 1.0, P.rear_accel_mps2, onset_sec=o)
                tb = 0.0 if already else o + c.rear_detect_delay
                # 이미 드러난 가속('rear_now')은 미래 정보가 아니다. 복귀로 피하는 계획을 허용하면
                # "바꾸되 곧바로 복귀"가 공짜로 안전해져 승인→복귀가 반복된다(MORAI S2, 2026-11-04).
                name = "rear_now" if already else f"rear@{o:.0f}"
                out.append((name, p_rear / len(onsets), tb, {rear.unique_id: mode}))
        if fronts and p_front > 1e-6:
            modes = {v.unique_id: TvMode("front_brake", 1.0, P.front_brake_accel_mps2) for v in fronts}
            out.append(("front_brake", p_front, self.cfg.front_detect_delay, modes))
        return out

    # ------------------------------------------------------- 대응 궤적
    def respond(self, traj, specs, t, tb, y_src, y_tgt):
        """루트 궤적 [N,K] → 대응별 궤적 [N,5,K] (x, y, v, 허용 여부 [N,5])."""
        c, P = self.cfg, self.P
        N, K = traj.x.shape
        dt = P.prediction_dt_sec
        tk = traj.t
        after = tk[None, :] > tb + 1e-9
        X = np.repeat(traj.x[:, None, :], 5, axis=1)
        Y = np.repeat(traj.y[:, None, :], 5, axis=1)
        V = np.repeat(traj.v[:, None, :], 5, axis=1)
        allowed = np.ones((N, 5), dtype=bool)
        if not math.isfinite(tb) or tb >= tk[-1]:
            allowed[:, 1:] = False
            return X, Y, V, allowed
        kb = int(np.searchsorted(tk, tb - 1e-9))            # 첫 분기 스텝
        y_now = y_tgt + t.ego_d_target                    # t=0 의 실제 자차 횡위치 (C++ 과 같다)
        y_b = np.array([np.interp(tb, np.concatenate([[0.0], tk]), np.concatenate([[y_now], traj.y[i]]))
                        for i in range(N)])
        v_b = np.array([np.interp(tb, np.concatenate([[0.0], tk]), np.concatenate([[traj.v[i, 0]], traj.v[i]]))
                        for i in range(N)])
        tau = np.maximum(0.0, tk[None, :] - tb)
        changing = np.array([s.change_left for s in specs])
        # 계속+가속: 횡은 그대로, 종방향은 v_root + a·τ (미션 속도 상한)
        v_acc = np.minimum(traj.v + c.escape_accel * tau, max(self.mission, 0.0))
        V[:, CONT_ACCEL] = np.where(after, np.maximum(traj.v, v_acc), traj.v)
        # 제동: 횡은 그대로, v_b 에서 감속
        V[:, BRAKE] = np.where(after, np.maximum(0.0, v_b[:, None] + c.brake_decel * tau), traj.v)
        # 복귀: 목표 차선 경계 전이면 원 차선으로 선형 복귀. 차선 변경 시작 전이면 취소.
        can_return = changing & (np.abs(y_b - y_tgt) > c.abort_before_abs_d)
        allowed[:, RETURN] = can_return
        allowed[:, RETURN_DECEL] = can_return
        allowed[~changing, CONT_ACCEL] = True
        allowed[~changing, RETURN] = False
        allowed[~changing, RETURN_DECEL] = False
        if c.plant_return:
            frac = np.stack([return_progress(tau[0], y_b[i] - y_src, P.lane_width_m) for i in range(N)])
        else:
            dur = np.maximum(0.8, c.return_time_full * np.abs(y_b - y_src) / P.lane_width_m)
            frac = np.clip(tau / dur[:, None], 0.0, 1.0)
        y_ret = np.where(after, y_b[:, None] + (y_src - y_b[:, None]) * frac, traj.y)
        Y[:, RETURN] = y_ret
        Y[:, RETURN_DECEL] = y_ret
        V[:, RETURN_DECEL] = np.where(after, np.maximum(0.5 * v_b[:, None], v_b[:, None] + c.return_decel * tau), traj.v)
        # 위치 적분(분기 이후만 다시 적분)
        for q in (CONT_ACCEL, RETURN_DECEL, BRAKE):
            ds = 0.5 * (V[:, q, 1:] + V[:, q, :-1]) * dt
            x = np.concatenate([traj.x[:, :1], traj.x[:, :1] + np.cumsum(ds, axis=1)], axis=1)
            X[:, q] = np.where(after, x, traj.x)
        return X, Y, V, allowed

    # ------------------------------------------------------- 트리 평가
    def evaluate_tree(self, traj, specs, t, v_e, R, vehicles, root_cost, candidate_mask=None):
        """루트별 (feasible, 기대 비용, 최적 조합, 기대 위험 peak)."""
        P, c = self.P, self.cfg
        N, K = traj.x.shape
        y_src, y_tgt = self.track.center_y(t.current_lane_id), self.track.center_y(t.target_lane_id)
        br = self.branches(t, v_e, R)
        B = len(br)
        risk = np.zeros((B, N, 5, K))
        allowed = np.zeros((B, N, 5), dtype=bool)
        nominal_clear = np.full((N,), True)
        leaf_v = np.zeros((B, N, 5))
        for b, (name, p, tb, modes) in enumerate(br):
            X, Y, V, ok = self.respond(traj, specs, t, tb, y_src, y_tgt)
            allowed[b] = ok
            flat = EgoTrajectory(t=traj.t, x=X.reshape(N * 5, K), y=Y.reshape(N * 5, K),
                                 yaw=np.zeros((N * 5, K)), v=V.reshape(N * 5, K),
                                 valid=np.repeat(traj.valid, 5), step_mask=np.repeat(traj.step_mask, 5, axis=0))
            no_coll = np.ones((N * 5, K))
            min_clear = np.full((N * 5, K), np.inf)
            bad = np.zeros(N * 5, dtype=bool)
            for v in vehicles:
                mode = modes.get(v.unique_id, TvMode("nominal", 1.0, P.nominal_target_accel_mps2))
                cond, clear, active, invalid = evaluate_mode(flat, v, mode, v_e, self.lanes, P)
                no_coll *= 1.0 - cond
                min_clear = np.minimum(min_clear, clear)
                bad |= invalid
            risk[b] = (1.0 - no_coll).reshape(N, 5, K)
            # 차선 끝 전에 TV 예측이 실패하면 그 대응은 쓸 수 없다(기존 SMPC 처럼 fail-closed, 리뷰 #11)
            allowed[b] &= ~bad.reshape(N, 5)
            if name == "nominal":
                nominal_clear = (min_clear.reshape(N, 5, K)[:, CONTINUE] >= P.behavior_min_signed_clearance_m).all(axis=1) \
                    | ~traj.valid
            if self.value_fn is not None:
                leaf_v[b] = self.value_fn(self.leaf_features(X, Y, V, traj, vehicles, modes)).reshape(N, 5)
        p = np.array([b[1] for b in br])
        cost_r = np.array(c.cost_response)
        # 가지별 의미 있는 대응만 열거: 정상=계속, 뒤차 가속=계속/가속/복귀/복귀+감속, 앞차 제동=계속/제동/복귀
        ex = c.executable_only
        def options(name):
            if name == "nominal":
                return [CONTINUE]
            if name == "rear_now":                  # 이미 관측된 상황: 계속해도 안전해야 승인
                return [CONTINUE] if ex else [CONTINUE, CONT_ACCEL]
            if name.startswith("rear"):
                return [CONTINUE, RETURN] if ex else [CONTINUE, CONT_ACCEL, RETURN, RETURN_DECEL]
            return [CONTINUE, BRAKE, RETURN]
        combos = np.array(list(itertools.product(*[options(b[0]) for b in br])))   # [C, B]
        feas = np.zeros(N, dtype=bool)
        best_cost = np.full(N, np.inf)
        best_combo = np.zeros((N, B), dtype=int)
        best_risk = np.ones(N)
        for i in range(N):
            if not traj.valid[i] or (candidate_mask is not None and not candidate_mask[i]):
                continue
            ok_c = np.ones(len(combos), dtype=bool)
            r_c = np.zeros((len(combos), K))
            j_c = np.zeros(len(combos))
            for b in range(B):
                q = combos[:, b]
                ok_c &= allowed[b, i, q]
                r_c += p[b] * risk[b, i][q]                 # 스텝별 Σ_j p_j r_j(k)
                j_c += p[b] * (cost_r[q] - leaf_v[b, i, q])
            peak = r_c.max(axis=1)
            total = root_cost[i] + j_c + c.risk_weight * peak
            good = ok_c & (peak <= P.risk_epsilon) & nominal_clear[i]
            if good.any():
                k = int(np.argmin(np.where(good, total, np.inf)))
                feas[i], best_cost[i], best_combo[i], best_risk[i] = True, total[k], combos[k], peak[k]
            elif ok_c.any():
                k = int(np.argmin(np.where(ok_c, peak, np.inf)))
                best_risk[i], best_combo[i] = peak[k], combos[k]
        self.last_tree = {"branches": [(b[0], round(b[1], 3), b[2]) for b in br]}
        return feas, best_cost, best_combo, best_risk, br

    def leaf_features(self, X, Y, V, traj, vehicles, modes):
        """잎(마지막 유효 스텝)의 자차 상태와, 같은 가지 mode 로 예측한 대표 차량 상태 → 특징 [N*5, DIM]."""
        from .prediction import predict_tv
        from .rl.features import leaf_features_batch
        N, Q, K = X.shape
        last = np.where(traj.step_mask.any(axis=1), traj.step_mask.shape[1] - 1 - np.argmax(traj.step_mask[:, ::-1], axis=1), K - 1)
        t_leaf = traj.t[last]                                   # [N]
        tv = []
        for i in range(N):
            items = []
            for v in vehicles:
                mode = modes.get(v.unique_id, TvMode("nominal", 1.0, self.P.nominal_target_accel_mps2))
                pr = predict_tv(v, mode, np.array([t_leaf[i]]), self.lanes, self.P)
                items.append((float(pr["x"][0]), float(pr["y"][0]), float(pr["v"][0]), v.length))
            tv.extend([items] * Q)
        idx = np.arange(N)
        return leaf_features_batch(X[idx, :, last].reshape(-1), Y[idx, :, last].reshape(-1),
                                   V[idx, :, last].reshape(-1), tv)

    # ------------------------------------------------------------ 판단
    def decide(self, now, t, v_e, mission, acc_target, acc_status):
        P = self.P
        self.mission = mission
        self.last_acc_target = acc_target
        if self.cfg.decision == "baseline":
            return BaselineSmpc.decide(self, now, t, v_e, mission, acc_target, acc_status)
        self.hist.update(t, now)
        self.scene.update(t, now, P)
        pre_cap = min(mission, acc_target) if acc_target is not None else mission
        D = waypoint_blend_duration(v_e) * P.ego_reference_preview_duration_scale
        modes_of = self.modes_of(v_e)
        passing = target_front_is_passing(t, v_e, P)
        pid = t.target_front.unique_id if passing else -1
        R = target_rear_for_gap(t, pid)
        vehicles = []
        for v in (t.current_front, t.target_front, t.target_rear):
            if v.valid and v.unique_id not in {w.unique_id for w in vehicles}:
                vehicles.append(v)
        if R is not None and R.unique_id not in {w.unique_id for w in vehicles}:
            vehicles.append(R)
        yield_ref = self.yield_episode_speed if (self.last_action == "yield" and self.yield_episode_speed is not None) else v_e
        tf = t.target_front
        near_end = t.distance_to_lane_end <= 30.0 and v_e <= 1.0
        if near_end:
            D = P.waypoint_blend_duration_sec
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=pre_cap, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=D, yield_ref_speed=yield_ref,
                             preview_mode="connector" if near_end else "blend", connector_join_arc=20.0)
        has_target = t.target_lane_id == t.current_lane_id + 1
        cf = t.current_front
        cur_gap = front_bumper_gap(cf, P) if cf.valid else math.inf
        tf_gap = front_bumper_gap(tf, P) if tf.valid else math.inf
        tr_gap = rear_bumper_gap(R, P) if R is not None else math.inf
        current_safe = max(P.current_front_safe_gap_m, P.standstill_gap_m + P.time_headway_sec * v_e)
        current_blocked = cf.valid and cur_gap < P.current_front_trigger_gap_m
        endpoint = t.lane_change_prepare or t.lane_change_urgent or t.emergency_stop_required
        c_r = conservative_rear_closing(R, v_e, self.hist, now, P) if R is not None else 0.0
        eff_rear_req = target_rear_required_gap(v_e, c_r, D, P) if R is not None else P.target_rear_safe_gap_m
        target_gap_cost = gap_penalty(tf_gap, P.target_front_safe_gap_m, P) + gap_penalty(tr_gap, eff_rear_req, P)

        # 루트 후보: CHANGE_NOW + 미래 후보(기존 SMPC 와 같은 집합)
        specs = [ActionSpec(CHANGE_NOW, 0.0, 0.0, D, True)]
        tr = t.target_rear
        accel_ok = P.behavior_execute_accel and tr.valid and conservative_rear_closing(tr, v_e, self.hist, now, P) >= P.behavior_rear_accel_min_closing_mps
        a_r = 0.0
        if accel_ok:
            c_tr = conservative_rear_closing(tr, v_e, self.hist, now, P)
            g_tr = max(0.0, rear_bumper_gap(tr, P))
            accel_ok = g_tr > P.target_rear_safe_gap_m and g_tr / max(0.1, c_tr) > P.behavior_rear_accel_emergency_ttc_sec
            a_r = min(max(P.behavior_rear_accel_gain * c_tr, P.behavior_rear_accel_min_mps2),
                      max(P.behavior_rear_accel_min_mps2, P.behavior_accel_accel_mps2))
        for t_lc in P.behavior_lc_start_times_sec:
            specs.append(ActionSpec(HOLD, P.behavior_hold_accel_mps2, t_lc, D, True))
            for a in P.behavior_yield_decel_candidates_mps2:
                specs.append(ActionSpec(YIELD, a, t_lc, D, True))
            if accel_ok:
                specs.append(ActionSpec(ACCEL, a_r, t_lc, D, True))
        specs = [s for s in specs if fits_horizon(s, P)]

        keep_traj = build_trajectories(ctx, [ActionSpec(KEEP)], self.track, P)
        keep_valid = bool(keep_traj.valid[0])
        lane_end_cost = 0.0
        if t.lane_change_required:
            lane_end_cost = (P.lane_end_emergency_cost if t.emergency_stop_required else
                             P.lane_end_urgent_cost if t.lane_change_urgent else
                             P.lane_end_prepare_cost if t.lane_change_prepare else 0.0)
        keep_cost = gap_penalty(cur_gap, current_safe, P) + lane_end_cost + 500.0 * (not keep_valid) + \
            P.blocked_cost_weight * current_blocked
        desire = endpoint or current_blocked
        g1 = has_target
        g6 = ((now - self.scene.stable_since >= P.change_commit_scene_observation_sec and
               self.veh_obs_stable(tf if tf.valid else None, now) and
               (not (tr.valid and abs(tr.delta_s) <= P.change_commit_scene_rear_max_range_m) or self.veh_obs_stable(tr, now)))
              or (passing and now - self.scene.pair_since >= P.pass_gap_lane_pair_observation_sec and
                  self.veh_obs_stable(tf, now) and self.veh_obs_stable(R, now)))
        g7 = cutin_interlock_safe(t, v_e, passing, pid, self.lanes, P)
        g8 = self.acc_stable(now, acc_status)
        request = BehaviorRequest(lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
        selected, reason, change = "", "", False
        if desire and g1 and specs:
            traj = self.root_trajectories(ctx, specs, t)
            root_cost = (P.lane_change_base_cost + target_gap_cost + P.behavior_delay_cost_weight *
                         np.array([s.t_lc for s in specs]) + P.behavior_accel_cost_weight *
                         np.array([s.prep_accel for s in specs]) ** 2)
            gate_ok = traj.valid.copy()
            if self.cfg.keep_gates:
                g3, why3 = immediate_gap_gate(t, v_e, R, passing, eff_rear_req, self.hist, now, P)
                gate_ok[0] &= g3
                gate_ok &= projected_gap_ttc_safe(t, traj, specs, R, passing, modes_of, self.lanes, self.hist, now, P)
            compat = (pre_cap + 0.10 >= v_e) or self.acc_ack(t, acc_status)
            gate_ok[1:] &= compat
            common = self.common_ok(g6, g7, g8)
            feas = np.zeros(len(specs), dtype=bool)
            cost = np.full(len(specs), np.inf)
            combo = np.zeros((len(specs), 1), dtype=int)
            if common and gate_ok.any():
                feas, cost, combo, peak, br = self.evaluate_tree(traj, specs, t, v_e, R, vehicles, root_cost,
                                                                 candidate_mask=gate_ok)
            ok = feas & gate_ok & common
            if ok[0] and self.cfg.monitor and self.monitor_core(t, v_e, mission, connector=near_end) == "return":
                # 같은 상태에서 실행 중 재계획이 곧바로 복귀를 고를 바꾸기는 승인하지 않는다(리뷰 #6)
                ok[0] = False
            wait_wins = False
            if self.cfg.wait_root and self.value_fn is not None and ok.any() and keep_valid:
                # '기다리기' 루트: 차선 끝 휴리스틱 비용 대신 -λV(잎) 로 지금 바꿀지/기다릴지를 V 가 정한다.
                wait_cost = gap_penalty(cur_gap, current_safe, P) + P.blocked_cost_weight * current_blocked
                kf, kc, _, _, _ = self.evaluate_tree(keep_traj, [ActionSpec(KEEP)], t, v_e, R, vehicles,
                                                     np.array([wait_cost]))
                wait_wins = bool(kf[0]) and kc[0] < np.min(np.where(ok, cost, np.inf))
                if wait_wins:
                    selected, reason = "wait(V)", "value_prefers_wait"
            if ok.any() and not wait_wins:
                i = int(np.argmin(np.where(ok, cost, np.inf)))
                spec = specs[i]
                selected = spec.label + " " + ",".join(RESPONSE_NAMES[r] for r in combo[i][1:])
                if spec.action == CHANGE_NOW:
                    change = True
                    self.lc_connector = near_end      # 실행 방식: 실행 중 모니터가 같은 방식으로 본다
                else:
                    t_sw = execution_start(spec, P)
                    if spec.action == HOLD:
                        cap = v_e
                    elif spec.action == YIELD:
                        floor = max(0.0, yield_ref - P.behavior_yield_max_speed_drop_mps)
                        cap = max(min(floor, v_e), v_e + spec.prep_accel * t_sw)
                    else:
                        cap = min(pre_cap, v_e + spec.prep_accel * t_sw)
                    request = BehaviorRequest(action=spec.action, active=True, speed_cap=cap, accel=spec.prep_accel,
                                              lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
            elif not wait_wins:
                reason = ("traj" if not traj.valid.any() else "scene/acc" if not common else
                          "gate" if not gate_ok.any() else "tree_risk")
        if desire and not change and not request.active:
            request = BehaviorRequest(action="wait", active=True, speed_cap=pre_cap,
                                      lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
            selected = selected or "wait"
        if request.action == "yield" and self.last_action != "yield":
            self.yield_episode_speed = v_e
        self.last_action = "change" if change else (request.action if request.active else "none")
        self.last_request = request if request.active else BehaviorRequest()
        return Decision(change_left=change, request=request, desire=desire, feasible=change,
                        reason=reason, selected=selected)

    # ------------------------------------------------- 하위 클래스 훅
    def root_trajectories(self, ctx, specs, t):
        """루트 후보의 자차 궤적. tree_reactive 는 가지마다 따로 굴리므로 여기서는 ctx 만 넘긴다."""
        return build_trajectories(ctx, specs, self.track, self.P)

    def common_ok(self, g6, g7, g8):
        return g6 and g7 and g8

    # ----------------------------------------------- 실행 중 재계획
    def monitor(self, now, t, v_e, mission):
        """차선 변경 중 매 틱: '지금 계속(이후 분기 허용)' 과 '지금 복귀' 두 루트를 트리로 비교한다.

        지금 행동은 모든 가지에 공통이어야 한다(아직 드러나지 않은 mode 에 따라 다르게 할 수 없다).
        계속 루트가 위험 제약을 지키면 계속, 아니면 복귀 루트가 지키면 복귀, 둘 다 못 지키면
        위험이 작은 쪽. 목표 차선 경계(1.75 m)를 넘었으면 복귀하지 않는다(supervisor 와 같다).
        """
        self.mission = mission
        self.hist.update(t, now)
        return self.monitor_core(t, v_e, mission)

    def monitor_core(self, t, v_e, mission, connector=None):
        """monitor 본체(이력 갱신 없음). decide 도 승인 전에 같은 상태로 불러 본다. connector 는 반응형만 쓴다."""
        P, c = self.P, self.cfg
        d_t = abs(t.ego_d_target)
        if d_t <= c.abort_before_abs_d:
            return None
        y_src, y_tgt = self.track.center_y(t.current_lane_id), self.track.center_y(t.target_lane_id)
        K, dt = P.horizon_steps, P.prediction_dt_sec
        taus = dt * np.arange(1, K + 1)
        y0 = y_tgt + t.ego_d_target
        T_rem = max(0.5, waypoint_blend_duration(v_e) * d_t / P.lane_width_m)
        y_go = y0 + (y_tgt - y0) * np.clip(taus / T_rem, 0.0, 1.0)
        if c.plant_return:
            y_back = y0 + (y_src - y0) * return_progress(taus, y0 - y_src, P.lane_width_m)
        else:
            T_back = max(0.8, c.return_time_full * abs(y0 - y_src) / P.lane_width_m)
            y_back = y0 + (y_src - y0) * np.clip(taus / T_back, 0.0, 1.0)
        # 승인할 때와 같은 종방향 모델: 계속 = blendStage(목표 차선 속도로), 복귀 = KEEP 법칙
        tf = t.target_front
        acc_target = getattr(self, "last_acc_target", None)
        pre_cap = min(mission, acc_target) if acc_target is not None else mission
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=pre_cap, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=T_rem, yield_ref_speed=v_e)
        from .rollouts import blend_stage_rollout, speed_target_rollout, bounded_preparation_rollout
        v_go, ds_go = blend_stage_rollout(ctx, taus, 0.0, T_rem, P)
        v_back, ds_back = speed_target_rollout(v_e, taus, pre_cap)
        if v_e < 1.0:
            # 정지·저속: 계획과 같은 connector 출발(가속 후 cap, 횡이동은 남은 연결 길이에 비례). C++ 과 같다.
            cap = P.stopped_launch_connector_speed_cap_by_target_lane_mps[t.target_lane_id]
            dt_ = P.prediction_dt_sec
            v, ds, vs, dss = v_e, 0.0, [], []
            for _ in taus:
                vn = min(cap, v + P.stopped_launch_connector_max_accel_mps2 * dt_) if cap > v + 0.05 else v
                ds += 0.5 * dt_ * (v + vn)
                v = max(0.0, vn)
                vs.append(v)
                dss.append(ds)
            v_go, ds_go = np.array(vs), np.array(dss)
            L_rem = max(2.0, 20.0 * d_t / P.lane_width_m)
            u = np.clip(ds_go / L_rem, 0.0, 1.0)
            y_go = y0 + (y_tgt - y0) * (u * u * u * (10.0 - 15.0 * u + 6.0 * u * u))
        traj = EgoTrajectory(t=taus, x=t.ego_s_current + np.vstack([ds_go, ds_back]),
                             y=np.vstack([y_go, y_back]), yaw=np.zeros((2, K)),
                             v=np.vstack([v_go, v_back]), valid=np.array([True, True]))
        passing = target_front_is_passing(t, v_e, P)
        R = target_rear_for_gap(t, t.target_front.unique_id if passing else -1)
        vehicles = []
        for v in (t.current_front, t.target_front, t.target_rear):      # C++ uniqueVehicles 처럼 중복 제거
            if v.valid and v.unique_id not in {w.unique_id for w in vehicles}:
                vehicles.append(v)
        if R is not None and R.unique_id not in {v.unique_id for v in vehicles}:
            vehicles.append(R)
        if not vehicles:
            return None
        specs = [ActionSpec(CHANGE_NOW, 0.0, 0.0, T_rem, True), ActionSpec(KEEP)]
        root_cost = np.array([0.0, c.cost_response[RETURN]])
        eps = P.risk_epsilon
        P.risk_epsilon = c.abort_risk_factor * eps        # 실행 중에는 2ε 이력: 새 정보 없이 흔들리지 않게
        try:
            feas, cost, combo, peak, br = self.evaluate_tree(traj, specs, t, v_e, R, vehicles, root_cost)
        finally:
            P.risk_epsilon = eps
        names = [b[0] for b in br]
        revealed = R is not None and self.hist.observed_accel(R) >= 0.5
        if feas[0]:
            # 이미 드러난 뒤차 가속 가지에서 최적 대응이 복귀면 복귀한다
            if revealed and any(n.startswith("rear") and combo[0][b] in (RETURN, RETURN_DECEL)
                                for b, n in enumerate(names)):
                return "return"
            return None
        if feas[1] or peak[1] < peak[0]:
            return "return"
        return None


def make_tree_planner(name):
    """run_batch.py 용: 'tree', 'tree_gt', 'tree_gt_nogate', 'tree_gt_V:<path>' ..."""
    parts = name.split(":")
    base = parts[0]
    # 2×2·민감도 접미사: _bdec(기존 판단+트리 모니터) _nomon(트리 판단+rear_abort) _full(실행 안 되는 대응 포함)
    # _linret(예전 선형 복귀 모델)
    flags = set(base.split("_"))
    cfg = TreeConfig(probability_table="gt" if "gt" in flags else "yaml", keep_gates="nogate" not in flags,
                     use_value="V" in flags or "Vw" in flags, wait_root="Vw" in flags,
                     executable_only="full" not in flags, plant_return="linret" not in flags,
                     decision="baseline" if "bdec" in flags else "tree", monitor="nomon" not in flags)
    value_fn = None
    if ("V" in flags or "Vw" in flags) and len(parts) > 1:
        from .rl.value import load_value_fn
        lam = float(parts[2]) if len(parts) > 2 else None
        value_fn = load_value_fn(parts[1], lam)

    def factory(P, track):
        tp = TreePlanner(P, track, cfg=TreeConfig(**cfg.__dict__), value_fn=value_fn)
        tp.name = base
        return tp
    return factory
