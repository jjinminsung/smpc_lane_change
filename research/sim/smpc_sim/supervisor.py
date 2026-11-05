"""lane_change_supervisor 이식 (baseline_spec.md 3장). 승인, 완료, rear_abort."""
from dataclasses import dataclass
import math

ESCAPE_ACCEL = 2.0


@dataclass
class SupervisorEvent:
    kind: str          # accept | complete | abort | timeout_abort
    t: float
    source: int
    target: int
    detail: str = ""


class Supervisor:
    def __init__(self, P_sup, track, initial_lane):
        self.P, self.track = P_sup, track
        self.lane = initial_lane          # 활성 경로 번호(/path_number)
        self.active = False
        self.source = self.target = -1
        self.change_start = -1e9
        self.clock_start = None
        self.last_switch = -1e9
        self.consecutive = 0
        self.d_init = 3.5
        self.rate_f = 0.0
        self.prev_dt_abs = None
        self.prev_stamp = None
        self.abort_count = 0
        self.abort_since = None
        self.complete_count = 0
        self.first_seen = {}
        self.last_seen = {}
        self.start_pose = None
        self.events = []
        self.last_guard = ""
        # 첫 요청부터 경로 전환까지 지연(MORAI: supervisor 확인 + /path_number → waypoint 전환 ≈ 0.55 s, 예측의
        # behavior_execution_delay_sec 와 같은 값). 0 이면 확인 횟수를 채우는 즉시 전환(11-05 까지의 시뮬).
        self.switch_delay = 0.0
        self.first_valid = None
        self.pending_at = None

    @property
    def forced_pair(self):
        return (self.source, self.target) if self.active else None

    # ------------------------------------------------------------ 3.1
    def on_decision(self, now, change_left, t, ego):
        if self.active:
            return
        if self.pending_at is not None:          # 확인은 끝났고 경로 전환을 기다리는 중(MORAI 처럼 되돌리지 않는다)
            if now >= self.pending_at - 1e-9:
                self.pending_at = None
                self._switch(now, t, ego)
            return
        valid = (change_left and t.current_lane_id == self.lane and t.target_lane_id == self.lane + 1)
        if not valid:
            self.consecutive = 0
            return
        if now - self.last_switch < self.P.cooldown_sec:
            return
        self.consecutive += 1
        if self.consecutive == 1:
            self.first_valid = now
        if self.consecutive < self.P.confirmation_count:
            return
        if self.switch_delay > 0.0 and now < self.first_valid + self.switch_delay - 1e-9:
            self.pending_at = self.first_valid + self.switch_delay
            self.consecutive = 0
            return
        self._switch(now, t, ego)

    def _switch(self, now, t, ego):
        self.active = True
        self.source, self.target = self.lane, self.lane + 1
        self.change_start = self.last_switch = now
        self.consecutive = 0
        self.d_init = max(0.85, abs(t.ego_d_target))
        self.rate_f, self.prev_dt_abs, self.prev_stamp = 0.0, None, None
        self.abort_count, self.abort_since = 0, None
        self.complete_count = 0
        self.start_pose = (ego.s, ego.v)
        self.clock_start = now if ego.v >= self.P.motion_start_speed_mps else None
        self.lane = self.target
        self.events.append(SupervisorEvent("accept", now, self.source, self.target))

    # ------------------------------------------------------------ 3.3
    def _closing(self, v, v_e, now):
        g = max(0.0, -v.front_delta_s - self.P.ego_rear_extent_m)
        c = max(0.0, v.v_long - v_e)
        if now - self.first_seen.get(v.unique_id, now) < self.P.active_rear_guard_new_track_guard_sec \
                and g <= self.P.active_rear_guard_new_track_max_gap_m:
            c = max(c, self.P.active_rear_guard_new_track_closing_upper_mps)
        return g, c

    def _remaining(self, d_t, v_e):
        P = self.P
        span = max(0.1, self.d_init - P.completion_lateral_d_m)
        rl = max(0.0, d_t - P.completion_lateral_d_m)
        T = P.lane_change_duration_sec * min(max(rl / span, 0.0), 1.0)
        if v_e < P.motion_start_speed_mps:
            T = max(T, P.active_rear_guard_stationary_remaining_sec)
        elif self.clock_start is not None and self.rate_f >= P.active_rear_guard_min_lateral_rate_mps:
            T = max(T, rl / max(0.05, self.rate_f))
        return min(max(T, P.active_rear_guard_min_remaining_sec), P.active_rear_guard_max_remaining_sec)

    def _requires_abort(self, now, t, v_e):
        P = self.P
        if not P.active_rear_guard_enabled or not self.active:
            return False, False
        if (t.current_lane_id, t.target_lane_id) != (self.source, self.target):
            return False, False
        d_t = abs(t.ego_d_target)
        if d_t <= P.active_rear_guard_abort_before_target_center_abs_d_m:
            return False, False
        T = self._remaining(d_t, v_e)
        cap = P.active_rear_guard_launch_speed_cap_by_target_lane_mps[self.target]
        threat = emergency = need_abort = need_launch = False
        details = []
        for v in [t.target_rear] + list(t.nearby_vehicles):
            if not v.valid or v.lane_id != self.target or v.rear_delta_s > 0.0:
                continue
            g, c = self._closing(v, v_e, now)
            intrudes = g <= P.active_rear_guard_min_gap_m + c * (T + P.active_rear_guard_perception_control_delay_sec) \
                or (c > 0.1 and g / c <= P.active_rear_guard_min_ttc_sec)
            if not intrudes:
                continue
            threat = True
            v_r = max(v.v_long, v_e + c)
            ego_run = v_e * T + 0.5 * ESCAPE_ACCEL * T * T
            escapable = g - (v_r * T - ego_run) > P.active_rear_guard_min_gap_m + 0.5 * c
            v_top = max(v_e, cap)
            t_acc = max(0.0, (v_top - v_e) / ESCAPE_ACCEL)
            if T <= t_acc:
                launch_run = v_e * T + 0.5 * ESCAPE_ACCEL * T * T
            else:
                launch_run = v_e * t_acc + 0.5 * ESCAPE_ACCEL * t_acc * t_acc + v_top * (T - t_acc)
            launch_esc = g - (v_r * T - launch_run) > P.active_rear_guard_min_gap_m + 0.5 * c
            emergency |= g <= P.active_rear_guard_emergency_gap_m or (c > 0.1 and g / c <= P.active_rear_guard_emergency_ttc_sec)
            need_abort |= not escapable
            need_launch |= not launch_esc
            details.append(f"id={v.unique_id} g={g:.1f} c={c:.1f} T={T:.2f} esc={int(escapable)}")
        self.last_guard = " ".join(details)
        if not threat:
            return False, False
        tf = t.target_front
        front_ok = (not tf.valid) or max(0.0, tf.rear_delta_s - P.ego_front_extent_m) >= max(
            P.active_rear_guard_front_min_gap_m, P.active_rear_guard_front_min_headway_sec * max(v_e, tf.v_long))
        if v_e < P.motion_start_speed_mps:
            if (now - self.change_start < P.active_rear_guard_stationary_launch_grace_sec and not emergency
                    and not need_launch and front_ok):
                return False, False
            return True, emergency
        abort = (not front_ok) or need_abort
        return abort, emergency and abort

    def on_targets(self, now, t, ego):
        v_e = abs(ego.v)
        for v in [t.target_rear] + list(t.nearby_vehicles):
            if v.valid and v.lane_id == t.target_lane_id and v.rear_delta_s <= 0.0:
                if now - self.last_seen.get(v.unique_id, -1e9) > self.P.rear_track_observation_gap_timeout_sec:
                    self.first_seen[v.unique_id] = now
                self.last_seen[v.unique_id] = now
        if not self.active:
            return
        d_t = abs(t.ego_d_target)
        if self.prev_dt_abs is not None and self.prev_stamp is not None and now > self.prev_stamp:
            raw = max(0.0, (self.prev_dt_abs - d_t) / (now - self.prev_stamp))
            a = self.P.active_rear_guard_lateral_rate_alpha
            self.rate_f = a * raw + (1 - a) * self.rate_f
        self.prev_dt_abs, self.prev_stamp = d_t, now
        need, emergency = self._requires_abort(now, t, v_e)
        if not need:
            self.abort_count, self.abort_since = 0, None
            return
        if not emergency:
            if self.abort_since is None:
                self.abort_since, self.abort_count = now, 1
                return
            self.abort_count += 1
            if self.abort_count < self.P.active_rear_guard_abort_confirmation_count or \
                    now - self.abort_since < self.P.active_rear_guard_abort_confirmation_sec:
                return
        self._abort(now, "rear_emergency" if emergency else "rear_intrusion", self.source)

    def _abort(self, now, why, lane):
        self.events.append(SupervisorEvent("abort", now, self.source, self.target, why + " " + self.last_guard))
        self.active = False
        self.lane = lane
        self.last_switch = now
        self.consecutive = 0
        self.source = self.target = -1

    # ------------------------------------------------------------ 3.2
    def tick(self, now, ego):
        if not self.active:
            return
        P = self.P
        if self.clock_start is None:
            moved = abs(ego.s - self.start_pose[0]) >= P.motion_start_displacement_m
            if ego.v >= P.motion_start_speed_mps or moved:
                self.clock_start = now
            else:
                return
        d_t = abs(ego.y - self.track.center_y(self.target))
        d_s = abs(ego.y - self.track.center_y(self.source))
        elapsed = now - self.clock_start
        if elapsed >= P.completion_min_active_sec and d_t <= P.completion_lateral_d_m:
            self.complete_count += 1
        else:
            self.complete_count = 0
        if self.complete_count >= P.completion_confirmation_count:
            self.events.append(SupervisorEvent("complete", now, self.source, self.target))
            self.active = False
            self.lane = self.target
            self.source = self.target = -1
            return
        if elapsed >= P.completion_timeout_sec:
            if d_t <= P.completion_lateral_d_m:
                self.events.append(SupervisorEvent("complete", now, self.source, self.target, "timeout"))
                self.active, self.lane = False, self.target
                self.source = self.target = -1
            elif d_s <= P.completion_failed_source_lateral_d_m and d_s + P.completion_lane_distance_margin_m < d_t:
                self._abort(now, "timeout_source", self.source)
