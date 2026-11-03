#!/usr/bin/env python3
"""MORAI 연구 주행 bag → 지표 json.

    python3 morai_metrics.py ~/bags/research/<tag>.bag [...]   (결과: 같은 이름 .metrics.json)

GT 기준(판단기가 본 값이 아니라 시뮬레이터 실제 값):
  자차  : /odom (후륜축, UTM). 박스 = 앞 3.845 m, 뒤 0.790 m, 폭 1.892 m.
  NPC   : /Object_topic position(MORAI 로컬, 후륜축) + UTM 오프셋, heading - 1.365°,
          박스 중심 = 후륜축 + shift (승용 1.25 m, 7.5 m 이상 0.5L - 1.1 m). research/results/gtcheck_report.md
  최소 간격 = 두 회전 박스 사이 거리(겹치면 음수 = 겹친 깊이). 충돌 = 간격 < 0.
"""
import bisect
import json
import math
import sys
from pathlib import Path

import numpy as np
import rosbag

OFF = (302595.0, 4124145.0)
HEADING_OFFSET_DEG = -1.365
EGO_FRONT, EGO_REAR, EGO_HALF_W = 3.845, 0.790, 0.946


def box(cx, cy, yaw, half_l, half_w):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[cx + c * dx - s * dy, cy + s * dx + c * dy]
                     for dx, dy in ((half_l, half_w), (half_l, -half_w), (-half_l, -half_w), (-half_l, half_w))])


def seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab)), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * ab)))


def signed_distance(A, B):
    """볼록 사각형 A, B. 떨어져 있으면 최소 거리(+), 겹치면 -최소 겹침(SAT)."""
    min_overlap = math.inf
    for P in (A, B):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]]) / max(1e-12, np.linalg.norm(e))
            pa, pb = A @ n, B @ n
            gap = max(pb.min() - pa.max(), pa.min() - pb.max())
            if gap > 0:
                d = min(seg_dist(p, Q[j], Q[(j + 1) % 4]) for p, Q in ((p, B) for p in A) for j in range(4))
                d = min(d, min(seg_dist(p, A[j], A[(j + 1) % 4]) for p in B for j in range(4)))
                return d
            min_overlap = min(min_overlap, -gap)
    return -min_overlap


def npc_center(o):
    # morai_object_info_node(UDP) 가 낸 /Object_topic 은 이미 UTM + heading 보정이다(|x| > 1e5).
    # MORAI ROS 브리지(9월 bag) 는 시나리오 로컬 좌표라 여기서 보정한다.
    utm = abs(o.position.x) > 1e5
    yaw = math.radians(o.heading + (0.0 if utm else HEADING_OFFSET_DEG))
    L, W = max(o.size.x, o.size.y), min(o.size.x, o.size.y)
    shift = max(0.0, 0.5 * L - 1.1) if L >= 7.5 else 1.25
    x = o.position.x + (0.0 if utm else OFF[0]) + shift * math.cos(yaw)
    y = o.position.y + (0.0 if utm else OFF[1]) + shift * math.sin(yaw)
    return x, y, yaw, 0.5 * L, 0.5 * W


def analyse(path):
    b = rosbag.Bag(str(path))
    odom, status, targets, sup_dbg, tree_dbg, objs, coll = [], [], [], [], [], [], []
    for top, m, t in b.read_messages(topics=["/odom", "/smpc/lane_change_status", "/smpc/targets",
                                             "/smpc/lane_change_supervisor_debug", "/smpc/tree_debug",
                                             "/Object_topic", "/CollisionData"]):
        ts = t.to_sec()
        if top == "/CollisionData":
            # MORAI 자체 접촉 기록. collision_object 에 자차(uid 0)는 늘 들어 있다 → 0 이 아닌 uid 만 접촉.
            others = [c.unique_id for c in m.collision_object if c.unique_id != 0]
            if others:
                coll.append((ts, others))
            continue
        if top == "/odom":
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            odom.append((ts, m.pose.pose.position.x, m.pose.pose.position.y, yaw,
                         math.hypot(m.twist.twist.linear.x, m.twist.twist.linear.y)))
        elif top == "/smpc/lane_change_status":
            status.append((ts, m.active, m.source_lane_id, m.target_lane_id))
        elif top == "/smpc/targets":
            targets.append((ts, m.current_lane_id, m.distance_to_lane_end))
        elif top == "/smpc/lane_change_supervisor_debug":
            sup_dbg.append((ts, m.data))
        elif top == "/smpc/tree_debug":
            tree_dbg.append((ts, m.data))
        else:
            objs.append((ts, [(o.unique_id, npc_center(o)) for o in m.npc_list]))
    b.close()
    if not odom:
        return {"bag": str(path), "error": "no odom"}
    odom = np.array(odom)
    moving = np.where(odom[:, 4] > 0.5)[0]
    t0 = odom[moving[0], 0] if len(moving) else odom[0, 0]

    # 차선 변경 이벤트: status 상승/하강 + supervisor debug 의 abort 문자열
    events = []
    prev = None
    for ts, act, src, tgt in status:
        if prev is not None and act and not prev[1]:
            events.append(("accept", ts - t0, src, tgt))
        if prev is not None and not act and prev[1]:
            events.append(("end", ts - t0, prev[2], prev[3]))
        prev = (ts, act, src, tgt)
    aborts = [(ts - t0, d) for ts, d in sup_dbg if "abort" in d and ("confirmed" in d or "emergency" in d or "tree_return" in d)
              or d.startswith("tree_return")]
    abort_times = [a for a, _ in aborts]
    completes = []
    for k, ts, src, tgt in events:
        if k == "end" and not any(abs(ts - a) < 0.6 for a in abort_times):
            completes.append((ts, src, tgt))
    t_merge = next((ts for ts, src, tgt in completes if tgt == 3), None)

    # 차선 끝 정지
    tt = [x[0] for x in targets]
    stops, stop_time, stopped = 0, 0.0, False
    for i in range(1, len(odom)):
        ts, v = odom[i, 0], odom[i, 4]
        j = max(0, min(bisect.bisect_left(tt, ts), len(tt) - 1)) if tt else None
        lane, dist = (targets[j][1], targets[j][2]) if j is not None else (3, math.inf)
        if v < 0.5 and lane < 3 and (t_merge is None or ts - t0 < t_merge):
            stop_time += odom[i, 0] - odom[i - 1, 0]
            if not stopped and dist < 60.0:
                stops += 1
            stopped = True
        elif v > 1.0:
            stopped = False

    # GT 최소 간격 (자차 박스 vs NPC 박스), 30 m 이내만
    ot = odom[:, 0]
    min_gap, min_gap_t, n_coll_frames = math.inf, None, 0
    near_ts = {}   # uid -> [(t, GT 박스 간격)] (자차 30 m 이내)
    for ts, npcs in objs:
        if ts < t0 or (t_merge is not None and ts - t0 > t_merge + 5.0):
            continue
        i = min(bisect.bisect_left(ot, ts), len(ot) - 1)
        _, ex, ey, eyaw, ev = odom[i]
        dt = ts - odom[i, 0]
        ex += ev * math.cos(eyaw) * dt
        ey += ev * math.sin(eyaw) * dt
        c = 0.5 * (EGO_FRONT - EGO_REAR)
        E = box(ex + c * math.cos(eyaw), ey + c * math.sin(eyaw), eyaw, 0.5 * (EGO_FRONT + EGO_REAR), EGO_HALF_W)
        for uid_, (x, y, yaw, hl, hw) in npcs:
            if math.hypot(x - ex, y - ey) > 30.0:
                continue
            g = signed_distance(E, box(x, y, yaw, hl, hw))
            near_ts.setdefault(uid_, []).append((ts, g))
            if g < min_gap:
                min_gap, min_gap_t = g, ts - t0
            if g < 0:
                n_coll_frames += 1
    # MORAI 접촉 사건: 같은 NPC 와 1 s 안에 이어지면 한 사건. 자차가 차선 변경 중이었는지로 분류.
    contacts = []

    def ego_gap_at(uid, ts):
        """그 시각 자차와 uid 의 GT 박스 간격(가장 가까운 프레임, 0.2 s 이내). 30 m 밖이면 inf."""
        best = None
        for t_, g_ in near_ts.get(uid, []):
            if abs(t_ - ts) <= 0.2 and (best is None or abs(t_ - ts) < best[0]):
                best = (abs(t_ - ts), g_)
        return best[1] if best else math.inf

    for ts, others in coll:
        for uid in others:
            # /CollisionData 에는 NPC 끼리의 접촉(예: 스폰 지점)도 들어온다. 자차와 0.3 m 이내일 때만 자차 접촉.
            if ego_gap_at(uid, ts) > 0.3:
                continue
            if contacts and contacts[-1]["uid"] == uid and ts - contacts[-1]["t_last"] < 1.0:
                contacts[-1]["t_last"] = ts
                continue
            st = [x for x in status if x[0] <= ts]
            active = bool(st and st[-1][1])
            tg = [x for x in targets if x[0] <= ts]
            lane = tg[-1][1] if tg else None
            contacts.append({"uid": int(uid), "t": round(float(ts - t0), 2), "t_last": float(ts), "ego_lc_active": bool(active),
                             "ego_lane": lane, "after_merge": bool(t_merge is not None and ts - t0 >= t_merge)})
    for c in contacts:
        c["t_last"] = round(float(c["t_last"] - t0), 2)
        c["kind"] = "during_ego_lane_change" if c["ego_lc_active"] else (
            "npc_cut_in_after_merge" if c["after_merge"] else "ego_lane_keeping")
    return {
        "bag": Path(path).name,
        "morai_contacts": contacts,
        "morai_collision": bool(contacts),
        "success": t_merge is not None,
        "t_merge": t_merge,
        "accepts": sum(1 for e in events if e[0] == "accept"),
        "completes": [(round(a, 2), s, g) for a, s, g in completes],
        "aborts": [(round(a, 2), d[:80]) for a, d in aborts],
        "lane_end_stops": stops,
        "stop_time": round(stop_time, 2),
        "min_gap_m": round(min_gap, 3) if math.isfinite(min_gap) else None,
        "min_gap_t": round(min_gap_t, 2) if min_gap_t is not None else None,
        "collision": n_coll_frames > 0,
        "collision_frames": n_coll_frames,
        "duration": round(odom[-1, 0] - t0, 1),
    }


def main():
    for p in sys.argv[1:]:
        r = analyse(p)
        out = Path(p).with_suffix(".metrics.json")
        out.write_text(json.dumps(r, indent=1))
        print(json.dumps(r))


if __name__ == "__main__":
    main()
