"""acc_speed_planner 이식 (baseline_spec.md 4장). 최종 목표 속도를 단독으로 낸다."""
from dataclasses import dataclass
import math

from .prediction import front_edge_delta_s, rear_edge_delta_s

BRAKING_DECEL, MAX_ACCEL = 3.8, 4.0
DEC_BASE, DEC_BOOST, DEC_HARD = 7.0, 11.0, 12.0


@dataclass
class AccStatus:
    reason: str = "none"
    cap_applied: bool = False
    external: bool = False
    applied_cap: float = math.inf
    final: float = 0.0
    hard: bool = False


@dataclass
class BehaviorRequest:
    action: str = "none"      # none | yield | hold | accel | wait
    active: bool = False
    speed_cap: float = math.inf
    accel: float = 0.0
    lane_pair: tuple = (-1, -1)
    stamp: float = -1.0


class AccPlanner:
    def __init__(self, P, track):
        self.P, self.track = P, track
        self.f = None
        self.status = AccStatus()
        self.wait_latch_until = -1.0
        self.last_change_left = -1e9

    # ------------------------------------------------------------ 4.1
    def speed_for_lead(self, L, v_e, cruise):
        P = self.P
        g = rear_edge_delta_s(L) - P.ego_front_extent_m
        vL = min(max(L.v_long, 0.0), cruise)
        c = max(0.0, v_e - vL)
        if g <= P.critical_gap_m:
            return 0.0, True, False, c, g
        g_star = P.standstill_gap_m + P.time_headway_sec * v_e + P.closing_speed_gap_time_sec * c + c * c / (2 * BRAKING_DECEL)
        cmd = max(vL + P.gap_gain * (g - g_star), vL - P.max_follow_speed_drop_mps)
        cmd = min(cmd, math.sqrt(vL * vL + 2 * BRAKING_DECEL * max(0.0, g - P.emergency_gap_m)))
        hard = rapid = False
        closing = v_e - max(0.0, L.v_long)
        if closing > 0.1:
            ttc = g / closing
            if ttc <= P.critical_ttc_sec:
                return 0.0, True, False, c, g
            cmd = min(cmd, vL + max(0.0, g - P.emergency_gap_m) / P.soft_ttc_sec)
            if ttc <= P.hard_ttc_sec:
                cmd = min(cmd, vL)
            rapid = (P.rapid_lead_brake_enabled and c >= P.rapid_lead_min_closing_speed_mps and
                     (ttc <= P.rapid_lead_ttc_sec or (g > P.emergency_gap_m and c * c / (2 * (g - P.emergency_gap_m)) >= 5.0)))
        return min(max(cmd, 0.0), cruise), hard, rapid, c, g

    def lead_in_front(self, L):
        return L.valid and front_edge_delta_s(L) > -self.P.ego_rear_extent_m and L.delta_s > 0.0

    # ----------------------------------------------------------- step
    def step(self, now, dt, t, v_e, ego_y, cruise, request, lc_active, lc_elapsed, intent, connector_cap=None):
        P = self.P
        if self.f is None:
            self.f = v_e
        raw, reason, hard_any, c_max = cruise, "none", False, 0.0
        soft_binding, lead_bind = False, None
        lane_end_pressure = t.lane_change_required and (t.lane_change_prepare or t.lane_change_urgent or t.emergency_stop_required)
        d_t = abs(t.ego_d_target)

        candidates = []
        if t.current_front.valid:
            candidates.append(("current_front", t.current_front, True))
        prep_fresh = request.active and now - request.stamp <= P.behavior_request_timeout_sec
        if t.target_front.valid and (intent or prep_fresh or lane_end_pressure):
            if d_t <= P.target_front_acc_entry_abs_d_m or t.current_lane_id == t.target_lane_id:
                candidates.append(("target_front", t.target_front, True))
            elif lc_active and d_t <= P.target_front_overlap_soft_acc_start_abs_d_m:
                candidates.append(("target_front", t.target_front, False))
        for v in t.nearby_vehicles:
            if abs(v.y - ego_y) <= 0.5 * P.ego_width_m + 0.5 * v.width + P.nearby_overlap_lateral_margin_m:
                if v.unique_id not in (t.current_front.unique_id, t.target_front.unique_id):
                    candidates.append(("nearby_overlap", v, True))
        for label, L, allow_hard in candidates:
            if not self.lead_in_front(L):
                continue
            cmd, hard, rapid, c, g = self.speed_for_lead(L, v_e, cruise)
            if not allow_hard:
                hard = rapid = False
            c_max = max(c_max, c)
            if cmd < raw:
                raw = cmd
                tag = f"{label}:{L.unique_id}" + (":hard" if hard else ":rapid" if rapid else "")
                reason = tag if allow_hard else tag + ":overlap_soft"
                lead_bind = (c, g)
                soft_binding = not allow_hard
            hard_any |= (hard or rapid) and allow_hard

        # 4.3 SMPC 요청 (차선 변경 중에는 무시)
        external = reason not in ("none", "merge_keep_rolling")
        cap_applied, applied_cap, yield_slew = False, math.inf, None
        recent_change = now - self.last_change_left <= 0.5
        pair_ok = request.lane_pair == (t.current_lane_id, t.target_lane_id)
        prep_active = (P.behavior_request_enabled and prep_fresh and pair_ok and not lc_active and not recent_change
                       and request.action in ("yield", "hold", "accel", "wait"))
        if prep_active and request.action == "wait":
            self.wait_latch_until = now + P.lane_end_hold_wait_for_gap_latch_sec
        if prep_active and request.action != "wait":
            cap = min(max(request.speed_cap, 0.0), cruise)
            applied_cap = cap
            if cap < raw:
                raw, cap_applied = cap, True
                lead_bind, soft_binding = None, False
                if not hard_any and not external and request.action == "yield" and request.accel < 0:
                    yield_slew = abs(request.accel)
        wait_active = now <= self.wait_latch_until and not lc_active

        # 4.4 차선 끝
        dist = t.distance_to_lane_end
        tf = t.target_front
        tf_blocks = tf.valid and self.lead_in_front(tf) and rear_edge_delta_s(tf) - P.ego_front_extent_m <= P.min_target_front_gap_m
        merge_wait = (not lc_active and t.lane_change_required and t.lane_change_urgent and
                      (tf_blocks or wait_active))
        lane_end_hard = False
        if merge_wait:
            hold = P.lane_end_merge_wait_stop_distance_m
            lim = 0.0 if dist <= hold else math.sqrt(2 * 2.5 * (dist - hold))
            if lim < raw:      # acc:1193-1207 (urgent). 아래 emergency 분기는 같은 값이라 '<' 에서 걸리지 않는다
                raw = lim
                lead_bind, soft_binding = None, False
                reason = "lane_end_wait_front_gap" if tf_blocks else "lane_end_wait_no_safe_gap"
                lane_end_hard = lim <= 0.01
        lane3_stop = False
        if t.current_lane_id in P.lane_end_stop_lane_ids and (dist <= P.lane_end_stop_prepare_distance_m or
                                                              t.time_to_lane_end <= P.lane_end_stop_prepare_time_sec):
            lim = math.sqrt(2 * 2.5 * max(0.0, dist - P.lane_end_stop_margin_m))
            lane3_stop = True
            if lim < raw:
                raw, reason = lim, "lane_end_stop"
                lead_bind, soft_binding = None, False
        if (P.merge_keep_rolling_enabled and (intent or lane_end_pressure) and not lane3_stop and not merge_wait
                and not hard_any and not lane_end_hard and not self._slow_lead_near(t, candidates)):
            floor = min(P.merge_keep_rolling_min_speed_mps, cruise)
            if raw < floor:
                raw, reason = floor, "merge_keep_rolling"
        if lc_active and P.lane_change_lane_end_cap_enabled and d_t > P.lane_change_lane_end_cap_release_abs_d_m:
            lim = (dist - P.lane_change_lane_end_margin_m) / max(0.5, 4.2 - lc_elapsed)
            lim = min(max(lim, min(P.lane_change_lane_end_min_cap_mps, cruise)), cruise)
            if lim < raw:
                raw, reason = lim, "lane_change_lane_end_cap"
                lead_bind, soft_binding = None, False
        if connector_cap is not None:
            raw = min(raw, connector_cap)

        # 4.6 필터
        boost = min(max((c_max - P.closing_decel_boost_start_mps) /
                        (P.closing_decel_boost_full_mps - P.closing_decel_boost_start_mps), 0.0), 1.0)
        dec = DEC_HARD if hard_any else DEC_BASE + boost * (DEC_BOOST - DEC_BASE)
        if yield_slew is not None:
            dec = min(dec, yield_slew)
        if soft_binding:
            dec = min(dec, 1.5)
        if lead_bind is not None and not hard_any and P.lead_required_decel_limit_enabled:
            c, g = lead_bind
            margin = g - P.emergency_gap_m - P.lead_required_decel_limit_headway_sec * v_e
            if margin > 0.5:
                dec = min(dec, max(1.5, P.lead_required_decel_limit_gain * c * c / (2 * margin)))
        if (hard_any or (lane_end_hard and reason != "merge_keep_rolling")) and P.hard_brake_bypass_filter:
            self.f = raw
        else:
            self.f = min(max(raw, self.f - dec * dt), self.f + MAX_ACCEL * dt)
        self.f = min(max(self.f, 0.0), cruise)
        self.status = AccStatus(reason=reason + (":hard" if hard_any and ":hard" not in reason and reason.startswith(("current", "target", "nearby")) else ""),
                                cap_applied=cap_applied, external=external, applied_cap=applied_cap,
                                final=self.f, hard=hard_any)
        return self.f

    def _slow_lead_near(self, t, candidates):
        P = self.P
        for _, L, _ in candidates:
            if not self.lead_in_front(L):
                continue
            g = rear_edge_delta_s(L) - P.ego_front_extent_m
            if g <= P.merge_keep_rolling_front_block_gap_m:
                return True
            if L.v_long <= P.merge_keep_rolling_low_speed_lead_mps and g <= P.merge_keep_rolling_low_speed_lead_max_gap_m:
                return True
        return False
