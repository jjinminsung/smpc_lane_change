#!/usr/bin/env python3
"""MORAI bag 의 모니터 틱을 C++ 재생 하네스(research/cpp_check/rx_replay.cpp) 입력으로 뽑는다.

    python3 rx_replay_ticks.py <bag> <t_from> <t_to> [--all <step>] > ticks.txt      (t = bag 시작 기준 초)

틱 = /smpc/tree_debug 의 "tree_monitor" 메시지 시각. 그 직전의 /smpc/targets, /odom, 미션 속도를 쓴다.
/path_switch_preview 는 bag 에 없어서 blend 시간은 waypoint_system base.yaml 의 속도 적응식으로 다시 구한다
(차선 변경 시작 시점 속도, smpc_decision.yaml ego_reference_preview_duration_scale 1.04 곱).
"""
import bisect
import math
import sys

import rosbag


def blend_duration(v):
    r = min(max((v - 8.0) / 20.0, 0.0), 1.0)
    a = (1.0 - r) * 1.3 + r * 1.0
    return min(max(math.sqrt(5.7735 * 3.5 / a), 2.6), 5.0) * 1.04


def last_before(ts, xs, t):
    i = bisect.bisect_right(ts, t) - 1
    return xs[max(i, 0)]


def veh_line(slot, v):
    return (f"veh {slot} {v.unique_id} {v.lane_id} {v.x:.4f} {v.y:.4f} {v.s:.4f} {v.d:.4f} {v.delta_s:.4f} {v.yaw:.5f} "
            f"{v.v_long:.4f} {v.v_lat:.4f} {v.length:.3f} {v.width:.3f} {v.half_s:.3f} {v.half_d:.3f} "
            f"{v.front_delta_s:.4f} {v.rear_delta_s:.4f} {v.track_hits} {v.accel:.4f} {v.role or '-'}")


def main():
    path, t_from, t_to = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    b = rosbag.Bag(path)
    t0 = b.get_start_time()
    rd = lambda topic: [(m.timestamp.to_sec() - t0, m.message) for m in b.read_messages(topics=[topic])]
    mon = [(t, m.data) for t, m in rd("/smpc/tree_debug") if m.data.startswith("tree_monitor")]
    if len(sys.argv) > 4 and sys.argv[4] == "--all":      # 모니터 틱 대신 /smpc/targets 를 step 초 간격으로
        step = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
        ts_all = [t for t, _ in rd("/smpc/targets")]
        mon, nxt = [], -1e9
        for t in ts_all:
            if t >= nxt:
                mon.append((t, "all"))
                nxt = t + step
    tg, od, ms = rd("/smpc/targets"), rd("/odom"), rd("/control_feedback/mission_speed_target_mps")
    tgt, odt, mst = [x[0] for x in tg], [x[0] for x in od], [x[0] for x in ms]
    lc_start_v = None
    for t, s in mon:
        if not (t_from <= t <= t_to):
            continue
        m = last_before(tgt, [x[1] for x in tg], t)
        o = last_before(odt, [x[1] for x in od], t)
        mis = last_before(mst, [x[1] for x in ms], t).data
        v = math.hypot(o.twist.twist.linear.x, o.twist.twist.linear.y)
        if lc_start_v is None:
            lc_start_v = v
        q = o.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        print(f"tick {t:.3f} ego {o.pose.pose.position.x:.4f} {o.pose.pose.position.y:.4f} {yaw:.5f} {v:.4f} "
              f"mission {mis:.4f} blend {blend_duration(lc_start_v):.4f}")
        print(f"targets {m.current_lane_id} {m.target_lane_id} {m.ego_s_current:.4f} {m.ego_d_current:.4f} "
              f"{m.ego_d_target:.4f} {m.current_lane_length:.4f} {m.distance_to_lane_end:.4f} {min(m.time_to_lane_end, 1e9):.4f}")
        for slot, v_ in (("cf", m.current_front), ("tf", m.target_front), ("tr", m.target_rear)):
            if v_.valid:
                print(veh_line(slot, v_))
        for v_ in m.nearby_vehicles:
            if v_.valid:
                print(veh_line("nb", v_))
        print("end")
        print(f"# recorded {s}", file=sys.stderr)


if __name__ == "__main__":
    main()
