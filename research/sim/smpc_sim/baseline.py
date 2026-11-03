"""기존 SMPC 결정기 (smpc_decision_node.cpp timerCallback). baseline_spec.md 2장.

열린 루프 후보 평가: 후보마다 궤적 하나를 고정해 두고, 그 궤적이 모든 게이트를 통과해야 한다.
차선 변경 뒤 뒤차가 가속하면 어떻게 대응할지는 평가에 없다(실행에서는 supervisor 의 rear_abort).
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .acc import BehaviorRequest
from .gates import (conservative_rear_closing, cutin_interlock_safe, immediate_gap_gate,
                    projected_gap_ttc_safe, target_front_is_passing, target_rear_for_gap,
                    target_rear_required_gap)
from .prediction import front_bumper_gap, rear_bumper_gap, modes_for_vehicle
from .risk import evaluate_trajectory, time_indexed_risk_safe, clearance_safe
from .rollouts import (ACCEL, CHANGE_NOW, HOLD, KEEP, YIELD, ActionSpec, RolloutContext,
                       build_trajectories, fits_horizon, execution_start, bounded_preparation_rollout)
from .vehicles import HistoryBank

INF = float("inf")
CONNECTOR_TRIGGER_M, CONNECTOR_JOIN_M = 30.0, 20.0   # PROGRESS_0910 F12


def waypoint_blend_duration(v):
    """waypoint_system 적응형 blend 시간 T(v) (local_path_publisher_node.cpp:372)."""
    a = np.interp(v, [8.0, 28.0], [1.3, 1.0])
    return float(min(max(math.sqrt(5.7735 * 3.5 / a), 2.6), 5.0))


def gap_penalty(g, g_star, P):
    if not math.isfinite(g):
        return 0.0
    return P.gap_cost_weight * max(0.0, g_star - g) ** 2 / max(1.0, g_star)


@dataclass
class Decision:
    change_left: bool
    request: BehaviorRequest
    desire: bool = False
    feasible: bool = False
    reason: str = ""
    selected: str = ""
    debug: dict = field(default_factory=dict)


class SceneHistory:
    """2.2 장면 관측 시계."""

    def __init__(self):
        self.init = False
        self.key = None
        self.pair = None
        self.last = -1e9
        self.stable_since = 0.0
        self.pair_since = 0.0

    def update(self, t, now, P):
        tf, tr = t.target_front, t.target_rear
        front = (tf.valid and tf.unique_id >= 0, tf.unique_id if tf.valid else -1)
        rear_present = tr.valid and tr.unique_id >= 0 and abs(tr.delta_s) <= P.change_commit_scene_rear_max_range_m
        rear = (rear_present, tr.unique_id if rear_present else -1)
        pair = (t.current_lane_id, t.target_lane_id)
        pair_changed = (not self.init) or now - self.last > P.change_commit_observation_gap_timeout_sec or pair != self.pair
        if pair_changed or (front, rear) != self.key:
            self.stable_since = now
        if pair_changed:
            self.pair_since = now
        self.init, self.key, self.pair, self.last = True, (front, rear), pair, now


class BaselineSmpc:
    name = "baseline"
    keep_gates = True          # False: G3/G5 와 모든-mode clearance 를 걷어내고 확률 제약만 남긴다

    def __init__(self, P, track):
        self.P, self.track = P, track
        self.lanes = track.lanes_for_prediction(P.tv_lane_projection_extension_before_m)
        self.hist = HistoryBank.from_params(P)
        self.scene = SceneHistory()
        self.acc_stable_since = None
        self.yield_key, self.yield_since = None, None
        self.last_action = "none"
        self.yield_episode_speed = None
        self.last_request = BehaviorRequest()

    # ------------------------------------------------------------ helpers
    def veh_obs_stable(self, v, now):
        if v is None or not v.valid:
            return True
        h = self.hist.histories.get(v.unique_id)
        if h is None or not h.initialized:
            return False
        return now - h.first_seen >= self.P.change_commit_vehicle_observation_sec and \
            now - h.last_seen <= self.P.change_commit_observation_gap_timeout_sec

    def acc_stable(self, now, acc_status):
        r = acc_status.reason
        blocked = any(w in r for w in ("hard", "rapid", "emergency")) or ("lane_end" in r and "lane_end_wait" not in r)
        if blocked:
            self.acc_stable_since = None
            return False
        if self.acc_stable_since is None:
            self.acc_stable_since = now
        return now - self.acc_stable_since >= self.P.change_commit_acc_stable_sec

    def acc_ack(self, t, acc_status):
        r = self.last_request
        return (r.active and r.action in ("yield", "hold", "accel") and r.lane_pair == (t.current_lane_id, t.target_lane_id)
                and acc_status.cap_applied and not acc_status.external
                and abs(acc_status.applied_cap - r.speed_cap) <= 0.1)

    def modes_of(self, v_e):
        return lambda v: modes_for_vehicle(v, self.hist.observed_accel(v), v_e, self.P)

    # ------------------------------------------------------------- decide
    def decide(self, now, t, v_e, mission, acc_target, acc_status):
        P = self.P
        self.hist.update(t, now)
        self.scene.update(t, now, P)
        pre_cap = min(mission, acc_target) if acc_target is not None else mission
        D = waypoint_blend_duration(v_e) * P.ego_reference_preview_duration_scale
        modes_of = self.modes_of(v_e)

        passing = target_front_is_passing(t, v_e, P)
        pid = t.target_front.unique_id if passing else -1
        R = target_rear_for_gap(t, pid)
        vehicles = [v for v in (t.current_front, t.target_front, t.target_rear) if v.valid]
        ids = {v.unique_id for v in vehicles}
        vehicles = [v for i, v in enumerate(vehicles) if v.unique_id not in {w.unique_id for w in vehicles[:i]}]
        if R is not None and R.unique_id not in ids:
            vehicles.append(R)

        yield_ref = self.yield_episode_speed if (self.last_action == "yield" and self.yield_episode_speed is not None) else v_e
        tf = t.target_front
        # waypoint_system connector: 정지(≤1 m/s) + 남은 거리 ≤ 30 m (GitHub 시절 F12 값). connector 일 때
        # egoChangeBlendDuration 은 배율 없는 4.0 s 다(smpc:1214).
        near_end = t.distance_to_lane_end <= CONNECTOR_TRIGGER_M and v_e <= 1.0
        if near_end:
            D = P.waypoint_blend_duration_sec
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=pre_cap, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=D, yield_ref_speed=yield_ref,
                             preview_mode="connector" if near_end else "blend",
                             connector_join_arc=CONNECTOR_JOIN_M)
        has_target = t.target_lane_id == t.current_lane_id + 1
        specs = [ActionSpec(KEEP), ActionSpec(CHANGE_NOW, 0.0, 0.0, D, True)]
        traj = build_trajectories(ctx, specs, self.track, P)
        ev = evaluate_trajectory(traj, vehicles, modes_of, v_e, self.lanes, P, targets=t, change_left=True)
        keep_valid, change_valid = bool(traj.valid[0] and ev.valid[0]), bool(traj.valid[1] and ev.valid[1] and has_target)

        cf = t.current_front
        cur_gap = front_bumper_gap(cf, P) if cf.valid else INF
        tf_gap = front_bumper_gap(tf, P) if tf.valid else INF
        tr_gap = rear_bumper_gap(R, P) if R is not None else INF
        current_safe = max(P.current_front_safe_gap_m, P.standstill_gap_m + P.time_headway_sec * v_e)
        current_blocked = cf.valid and cur_gap < P.current_front_trigger_gap_m
        endpoint = t.lane_change_prepare or t.lane_change_urgent or t.emergency_stop_required
        if R is not None:
            c_r = conservative_rear_closing(R, v_e, self.hist, now, P)
            eff_rear_req = target_rear_required_gap(v_e, c_r, D, P)
        else:
            eff_rear_req = P.target_rear_safe_gap_m

        g1 = has_target
        g2 = change_valid
        g3, g3_why = immediate_gap_gate(t, v_e, R, passing, eff_rear_req, self.hist, now, P)
        g4 = g2 and bool(time_indexed_risk_safe(ev, P.risk_epsilon)[1]) and \
            (bool(clearance_safe(ev, P)[1]) or not self.keep_gates)
        g5 = g2 and bool(projected_gap_ttc_safe(t, _row(traj, 1), [specs[1]], R, passing, modes_of,
                                                self.lanes, self.hist, now, P)[0])
        if not self.keep_gates:
            g3, g5 = True, True
        scene_ok = (now - self.scene.stable_since >= P.change_commit_scene_observation_sec and
                    self.veh_obs_stable(tf if tf.valid else None, now) and
                    (not (t.target_rear.valid and abs(t.target_rear.delta_s) <= P.change_commit_scene_rear_max_range_m)
                     or self.veh_obs_stable(t.target_rear, now)))
        pass_ok = (passing and now - self.scene.pair_since >= P.pass_gap_lane_pair_observation_sec and
                   self.veh_obs_stable(tf, now) and self.veh_obs_stable(R, now))
        g6 = scene_ok or pass_ok
        g7 = cutin_interlock_safe(t, v_e, passing, pid, self.lanes, P)
        g8 = self.acc_stable(now, acc_status)
        feasible = g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8

        lane_end_cost = 0.0
        if t.lane_change_required:
            lane_end_cost = (P.lane_end_emergency_cost if t.emergency_stop_required else
                             P.lane_end_urgent_cost if t.lane_change_urgent else
                             P.lane_end_prepare_cost if t.lane_change_prepare else 0.0)
        keep_cost = (P.risk_cost_weight * float(ev.legacy_risk[0]) + gap_penalty(cur_gap, current_safe, P)
                     + lane_end_cost + 500.0 * (not keep_valid) + P.blocked_cost_weight * current_blocked)
        target_gap_cost = gap_penalty(tf_gap, P.target_front_safe_gap_m, P) + gap_penalty(tr_gap, eff_rear_req, P)
        change_cost = (P.lane_change_base_cost + P.risk_cost_weight * float(ev.legacy_risk[1]) + target_gap_cost
                       + 500.0 * (not g1) + 500.0 * (not g2) + 180.0 * (not g3) + 180.0 * (not g4) + 180.0 * (not g5))
        desire = endpoint or current_blocked or (change_cost + P.decision_hysteresis_cost < keep_cost)
        request_change = desire and feasible
        reason = ""
        if desire and not feasible:
            reason = next(n for n, ok in (("target", g1), ("traj", g2), (g3_why or "gap", g3), ("risk", g4),
                                          ("projected", g5), ("scene", g6), ("cutin", g7), ("acc", g8)) if not ok)

        selected, request = "", BehaviorRequest(lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
        if request_change:
            selected = "change_now"
            self.yield_key = None
        elif desire and g1 and g6 and g7 and g8 and P.behavior_planner_enabled:
            best = self.best_future(now, t, v_e, ctx, D, R, passing, vehicles, modes_of, target_gap_cost, acc_status)
            if best is not None:
                spec, cap = best
                selected = spec.label
                request = BehaviorRequest(action=spec.action, active=True, speed_cap=cap, accel=spec.prep_accel,
                                          lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
        if desire and not request_change and not request.active:
            request = BehaviorRequest(action="wait", active=True, speed_cap=pre_cap,
                                      lane_pair=(t.current_lane_id, t.target_lane_id), stamp=now)
            selected = selected or "wait"

        if request.action == "yield" and self.last_action != "yield":
            self.yield_episode_speed = v_e
        self.last_action = "change" if request_change else (request.action if request.active else "none")
        self.last_request = request if request.active else BehaviorRequest()
        return Decision(change_left=request_change, request=request, desire=desire, feasible=feasible,
                        reason=reason, selected=selected,
                        debug={"keep_cost": keep_cost, "change_cost": change_cost, "peak": float(ev.peak_step_risk[1]),
                               "gates": (g1, g2, g3, g4, g5, g6, g7, g8)})

    # --------------------------------------------------- 2.6 future 후보
    def best_future(self, now, t, v_e, ctx, D, R, passing, vehicles, modes_of, target_gap_cost, acc_status):
        P = self.P
        specs = []
        c_r = conservative_rear_closing(R, v_e, self.hist, now, P) if R is not None else 0.0
        tr = t.target_rear
        accel_ok = (P.behavior_execute_accel and tr.valid and
                    conservative_rear_closing(tr, v_e, self.hist, now, P) >= P.behavior_rear_accel_min_closing_mps)
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
        if not specs:
            return None
        compat = (ctx.pre_cap + 0.10 >= v_e) or self.acc_ack(t, acc_status)
        if not compat:
            return None
        traj = build_trajectories(ctx, specs, self.track, P)
        ev = evaluate_trajectory(traj, vehicles, modes_of, v_e, self.lanes, P, targets=t, change_left=True)
        feas = traj.valid & time_indexed_risk_safe(ev, P.risk_epsilon)
        if self.keep_gates:
            feas &= clearance_safe(ev, P) & projected_gap_ttc_safe(t, traj, specs, R, passing, modes_of,
                                                                   self.lanes, self.hist, now, P)
        cost = (P.lane_change_base_cost + target_gap_cost + P.risk_cost_weight * ev.peak_step_risk +
                P.behavior_delay_cost_weight * np.array([s.t_lc for s in specs]) +
                P.behavior_accel_cost_weight * np.array([s.prep_accel for s in specs]) ** 2)

        def best_of(kind):
            idx = [i for i, s in enumerate(specs) if s.action == kind and feas[i]]
            return min(idx, key=lambda i: cost[i]) if idx else None

        hold, yld, acc = best_of(HOLD), best_of(YIELD), best_of(ACCEL)
        rear_press = R is not None and c_r >= 1.0
        choice = None
        if rear_press and acc is not None:
            choice = acc
        elif hold is not None:
            choice = hold
        elif not rear_press and yld is not None:
            key = (t.current_lane_id, t.target_lane_id, t.target_front.unique_id, t.target_rear.unique_id)
            if self.yield_key != key:
                self.yield_key, self.yield_since = key, now
            if now - self.yield_since >= P.behavior_yield_confirm_sec:
                choice = yld
            else:
                return None
        elif acc is not None:
            choice = acc
        if choice is None or specs[choice].action != YIELD:
            self.yield_key = None
        if choice is None:
            return None
        spec = specs[choice]
        t_sw = execution_start(spec, P)
        if spec.action == HOLD:
            cap = v_e
        elif spec.action == YIELD:
            floor = max(0.0, ctx.yield_ref_speed - P.behavior_yield_max_speed_drop_mps)
            cap = max(min(floor, v_e), v_e + spec.prep_accel * t_sw)
        else:
            cap = min(ctx.pre_cap, v_e + spec.prep_accel * t_sw)
        return spec, cap


def _row(traj, i):
    from .risk import EgoTrajectory
    return EgoTrajectory(t=traj.t, x=traj.x[i:i + 1], y=traj.y[i:i + 1], yaw=traj.yaw[i:i + 1],
                         v=traj.v[i:i + 1], valid=traj.valid[i:i + 1], step_mask=traj.step_mask[i:i + 1])
