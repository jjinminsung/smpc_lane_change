"""target_selector 이식: 월드 상태 → TargetSet (baseline_spec.md 5장). GT 인지(object_topic) 가정."""
import math

from .vehicles import TargetSet, TargetVehicle

MAX_RANGE = 120.0
MEMBER_D, REP_D, NEARBY_D = 2.4, 1.35, 3.4
NEARBY_FRONT, NEARBY_REAR, NEARBY_MAX = 150.0, 120.0, 40


def _tv(npc, ego_s, lane_of, track, role=""):
    lane = lane_of(npc)
    half_s = 0.5 * npc.length
    delta = npc.s - ego_s
    return TargetVehicle(
        valid=True, unique_id=npc.uid, role=role, lane_id=lane, x=npc.s, y=npc.y,
        s=npc.s, d=npc.y - track.center_y(lane), delta_s=delta, yaw=0.0, yaw_rate=0.0,
        v_long=npc.v, v_lat=npc.v_lat, length=npc.length, width=npc.width, half_s=half_s,
        front_delta_s=delta + half_s, rear_delta_s=delta - half_s, track_hits=100, accel=getattr(npc, "a", 0.0))


def build_targets(now, ego, npcs, track, P_sel, forced_pair=None):
    """ego: plant.Ego (s=후륜축, y, v, lane). forced_pair: 차선 변경 중 (source, target)."""
    if forced_pair is not None:
        cur, tgt = forced_pair
    else:
        cur = ego.lane
        tgt = cur + 1 if cur < track.final_lane else cur
    t = TargetSet(stamp=now, current_lane_id=cur, target_lane_id=tgt)
    t.ego_s_current = ego.s
    t.ego_d_current = ego.y - track.center_y(cur)
    t.ego_d_target = ego.y - track.center_y(tgt)
    t.current_lane_length = track.csv_end[cur] - track.csv_start[cur]
    dist = max(0.0, track.csv_end[cur] - ego.s)
    ttle = dist / ego.v if ego.v > 0.1 else math.inf
    req = cur < track.final_lane
    t.distance_to_lane_end, t.time_to_lane_end = dist, ttle
    t.lane_change_required = req
    t.lane_change_prepare = req and (dist <= P_sel.lane_end_prepare_distance_m or ttle <= P_sel.lane_end_prepare_time_sec)
    t.lane_change_urgent = req and (dist <= P_sel.lane_end_urgent_distance_m or ttle <= P_sel.lane_end_urgent_time_sec)
    t.emergency_stop_required = req and (dist <= P_sel.lane_end_emergency_distance_m or ttle <= P_sel.lane_end_emergency_time_sec)

    def best_lane(npc):
        return track.lane_of_y(npc.y)

    best = {"current_front": (MAX_RANGE, None), "target_front": (MAX_RANGE, None),
            "target_rear": (MAX_RANGE, None)}
    nearby = []
    for npc in npcs:
        lane = best_lane(npc)
        d_l = npc.y - track.center_y(lane)
        if abs(d_l) - 0.5 * npc.width > NEARBY_D:
            continue
        base = _tv(npc, ego.s, best_lane, track)
        if base.front_delta_s >= -NEARBY_REAR and base.rear_delta_s <= NEARBY_FRONT:
            nearby.append(base)
        def representative(lane_id):
            d_lane = npc.y - track.center_y(lane_id)
            return abs(d_lane) - 0.5 * npc.width <= MEMBER_D and abs(d_lane) <= REP_D

        def offer(role, key):
            if key < best[role][0]:
                best[role] = (key, npc)

        if representative(cur) and base.front_delta_s > 0.0:
            offer("current_front", max(0.0, base.rear_delta_s))
        if tgt != cur and representative(tgt):
            if base.front_delta_s >= 0.0:
                offer("target_front", max(0.0, base.rear_delta_s))
            if base.rear_delta_s <= 0.0:
                offer("target_rear", max(0.0, -base.front_delta_s))
    for role, (_, npc) in best.items():
        if npc is not None:
            v = _tv(npc, ego.s, best_lane, track, role)
            v.lane_id = cur if role == "current_front" else tgt
            v.d = npc.y - track.center_y(v.lane_id)
            setattr(t, role, v)
    nearby.sort(key=lambda v: abs(v.delta_s))
    for v in nearby[:NEARBY_MAX]:
        v.role = ("nearby_front" if v.rear_delta_s > 0.0 else
                  "nearby_rear" if v.front_delta_s < 0.0 else "nearby_overlap")
        t.nearby_vehicles.append(v)
    return t
