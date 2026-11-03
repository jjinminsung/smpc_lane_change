#!/usr/bin/env python3
"""MORAI 접촉 장면 분석: 접촉 직전 2 s 의 자차·상대 NPC 상대 위치/속도/heading, 자차 차선 변경 상태.

    python3 contact_scene.py <bag> <uid> <t_contact_rel>
"""
import bisect
import math
import sys

import numpy as np
import rosbag

from morai_metrics import npc_center


def main():
    bag, uid, tc = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
    b = rosbag.Bag(bag)
    odom, objs, status, comp = [], [], [], []
    for top, m, t in b.read_messages(topics=["/odom", "/Object_topic", "/smpc/lane_change_status", "/Competition_topic"]):
        ts = t.to_sec()
        if top == "/odom":
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            odom.append((ts, m.pose.pose.position.x, m.pose.pose.position.y, yaw))
        elif top == "/Competition_topic":
            comp.append((ts, math.hypot(m.velocity.x, m.velocity.y)))
        elif top == "/Object_topic":
            for o in m.npc_list:
                if o.unique_id == uid:
                    objs.append((ts,) + npc_center(o) + (o.velocity.x / 3.6,))
        else:
            status.append((ts, m.active, m.source_lane_id, m.target_lane_id))
    b.close()
    odom = np.array(odom)
    comp = np.array(comp)
    t0 = comp[np.where(comp[:, 1] > 0.5)[0][0], 0]
    ot = odom[:, 0]
    ct = comp[:, 0]
    for ts, x, y, yaw, hl, hw, v in objs:
        if not (tc - 2.5 <= ts - t0 <= tc + 0.5):
            continue
        i = min(bisect.bisect_left(ot, ts), len(ot) - 1)
        _, ex, ey, eyaw = odom[i]
        ev = comp[min(bisect.bisect_left(ct, ts), len(ct) - 1), 1]
        dx, dy = x - ex, y - ey
        lon = dx * math.cos(eyaw) + dy * math.sin(eyaw)
        lat = -dx * math.sin(eyaw) + dy * math.cos(eyaw)
        st = [s for s in status if s[0] <= ts]
        lc = st[-1][1:] if st else None
        print(f"t={ts - t0:6.2f} npc lon={lon:+6.2f} lat={lat:+5.2f} dyaw={math.degrees(yaw - eyaw):+5.1f} v_npc={v:5.2f} v_ego={ev:5.2f} lc={lc}")


if __name__ == "__main__":
    main()
