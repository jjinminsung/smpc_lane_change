#!/usr/bin/env python3
"""MORAI 연구 주행 한 번의 종료 시점만 판단한다. 지표는 bag 에서 analysis/morai_metrics.py 로 낸다.

종료 조건 (먼저 오는 것):
  done     : 차선 3 으로의 변경이 완료되고 5 s 더 달림
  stuck    : 자차가 움직인 뒤 정지(<0.3 m/s)가 60 s 이어짐
  overrun  : 차선 0~2 에서 그 차선 CSV 끝을 20 m 넘음
  timeout  : 자차가 움직이기 시작한 뒤 180 s
  no_start : 노드가 뜬 뒤 90 s 안에 자차가 움직이지 않음
종료 코드 0 = done, 그 밖은 실패/중단. 마지막 줄에 JSON 요약을 출력한다.
"""
import json
import math
import sys
import time

import rospy
from morai_msgs.msg import EgoVehicleStatus
from smpc_lane_change.msg import LaneChangeStatus, TargetVehicleSet

CSV_LEN = {0: 1218.6, 1: 403.9, 2: 417.9}


class Monitor:
    def __init__(self):
        self.v = 0.0
        self.lane = None
        self.ego_s = None
        self.lane_len = None
        self.moved_at = None
        self.stop_since = None
        self.done_at = None
        self.lane_since = None
        self.completions = []
        self.prev_active = False
        self.prev_pair = None
        # 속도는 MORAI 가 직접 주는 값(EKF 초기 잡음에 속지 않게)
        rospy.Subscriber("/Competition_topic", EgoVehicleStatus, self.ego, queue_size=5)
        rospy.Subscriber("/smpc/targets", TargetVehicleSet, self.targets, queue_size=5)
        rospy.Subscriber("/smpc/lane_change_status", LaneChangeStatus, self.status, queue_size=20)

    def ego(self, m):
        self.v = math.hypot(m.velocity.x, m.velocity.y)

    def targets(self, m):
        if m.current_lane_id != self.lane:
            self.lane_since = rospy.get_time()
        self.lane = m.current_lane_id
        self.ego_s = m.ego_s_current
        self.lane_len = m.current_lane_length

    def status(self, m):
        if self.prev_active and not m.active and self.prev_pair is not None:
            # 활성 → 비활성. 완료인지 abort 인지는 bag 에서 가린다. 목표가 3 이면 완료 후보.
            self.completions.append((rospy.get_time(), self.prev_pair))
        self.prev_active = m.active
        self.prev_pair = (m.source_lane_id, m.target_lane_id)


def main():
    rospy.init_node("research_episode_monitor", anonymous=True)
    mon = Monitor()
    t_start = time.time()
    reason = "timeout"
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        now = time.time()
        if mon.moved_at is None:
            if mon.v > 0.5:
                mon.moved_at = now
            elif now - t_start > 90.0:
                reason = "no_start"
                break
        else:
            # 2→3 전환(완료 또는 abort) 뒤 차선 3 이 3 s 이상 이어져야 완료로 본다. abort 직후 targets 가
            # 한 번 lane 3 을 내는 경우가 있다(2026-11-04 v3_C_G_s70_d40: 거짓 done 으로 재시도 중에 끊김).
            if (mon.lane == 3 and mon.done_at is None and any(p == (2, 3) for _, p in mon.completions)
                    and mon.lane_since is not None and rospy.get_time() - mon.lane_since >= 3.0):
                mon.done_at = now
            if mon.done_at is not None and now - mon.done_at > 5.0:
                reason = "done"
                break
            if mon.v < 0.3:
                mon.stop_since = mon.stop_since or now
                if now - mon.stop_since > 60.0:
                    reason = "stuck"
                    break
            else:
                mon.stop_since = None
            if mon.lane in CSV_LEN and mon.ego_s is not None and mon.ego_s > CSV_LEN[mon.lane] + 20.0:
                reason = "overrun"
                break
            if now - mon.moved_at > 180.0:
                reason = "timeout"
                break
        rate.sleep()
    out = {"reason": reason, "wall_s": round(time.time() - t_start, 1),
           "moving_s": round(time.time() - mon.moved_at, 1) if mon.moved_at else None,
           "final_lane": mon.lane}
    print(json.dumps(out), flush=True)
    sys.exit(0 if reason == "done" else 1)


if __name__ == "__main__":
    main()
