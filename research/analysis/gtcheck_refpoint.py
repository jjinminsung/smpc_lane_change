#!/usr/bin/env python3
"""GT check B: reference point of /Object_topic npc.position and /Ego_topic position.

Two independent methods.

1. Kinematic side slip (GT only, all 09-* bags).  For a reference point x m
   ahead of the zero-slip point (rear axle at low lateral acceleration), the
   velocity direction differs from the heading by beta = heading - course =
   -atan(x * kappa).  After removing the constant heading bias gamma (MORAI
   heading is true-north based, the map is the UTM grid; gamma = median beta
   on straights), x = -(beta - gamma) / kappa.  Rear axle -> x ~ 0 (growing
   slightly negative with speed as tyre slip moves the zero-slip point
   forward); bbox centre -> x ~ +(L/2 - rear overhang) ~ +1.2..1.5 m.

2. LiDAR faces.  /velodyne_points (per-point capture time field) transformed
   to the map with /odom (rear axle per the stack), lidar_x = 1.0 m on the
   centreline (launch/smpc_integration.launch), yaw = /odom yaw - gamma.  For
   an NPC directly ahead (behind) in the ego lane, 5-40 m, the measured rear
   (front) face along the NPC heading is compared with GT position -/+ L/2.
   offset = measured face - predicted face.  If npc.position is the bbox
   centre, offset ~ 0 on both sides; if it is a point c m behind the centre
   (rear axle: c ~ L/2 - rear overhang), offset ~ +c on both sides.  Adjacent
   NPCs alongside the ego give both faces in one scan (length check).
   Two timing hypotheses: H1 each point transformed with the ego pose at its
   capture time (stamp + time), H2 the whole scan with the stamp pose.

Output: research/results/gtcheck_refpoint.json, gtcheck_refpoint_faces.csv
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np
from scipy.signal import savgol_filter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bagtools as bt  # noqa: E402
import gtcheck_common as gc  # noqa: E402

LIDAR_X = 1.0
PT = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4"), ("ring", "<u2"),
               ("time", "<f4")])


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


# --------------------------------------------------------------------------
# 1. kinematic test
# --------------------------------------------------------------------------
def kin_track(t, x, y, hdeg, dtg=0.02):
    m = np.concatenate([[True], (np.diff(x) != 0) | (np.diff(y) != 0)]) & \
        np.concatenate([[True], np.diff(t) > 1e-4])
    t, x, y, h = t[m], x[m], y[m], np.unwrap(np.radians(hdeg[m]))
    if len(t) < 50:
        return None
    g = np.arange(t[0], t[-1], dtg)
    gx, gy, gh = np.interp(g, t, x), np.interp(g, t, y), np.interp(g, t, h)
    gap = np.zeros(len(g), bool)
    for i in np.flatnonzero(np.diff(t) > 0.15):
        gap |= (g > t[i] - 0.1) & (g < t[i + 1] + 0.1)
    vx = savgol_filter(gx, 11, 2, deriv=1, delta=dtg)
    vy = savgol_filter(gy, 11, 2, deriv=1, delta=dtg)
    hs = savgol_filter(gh, 11, 2)
    om = savgol_filter(gh, 11, 2, deriv=1, delta=dtg)
    v = np.hypot(vx, vy)
    ok = ~gap & (v > 3.0)
    return wrap(hs - np.arctan2(vy, vx))[ok], om[ok], v[ok]


def kinematic(bags):
    acc = {"ego": [], "npc": []}
    names = {}
    for p in bags:
        d = bt.load_bag(p)
        e = d["ego"]
        r = kin_track(e["t"], e["x"], e["y"], e["heading"])
        if r:
            acc["ego"].append(np.c_[r[0], r[1], r[2], np.full(len(r[0]), 4.635)])
        tr = bt.gt_tracks(d, {"dx": 0.0, "dy": 0.0})
        for u, t in tr.items():
            r = kin_track(t["t"], t["x"], t["y"], t["heading"])
            if r:
                acc["npc"].append(np.c_[r[0], r[1], r[2], np.full(len(r[0]), np.median(t["length"]))])
    out = {}
    for who, rows in acc.items():
        a = np.concatenate(rows)
        beta, om, v, L = a.T
        kap = om / v
        st = np.abs(kap) < 0.002
        gamma = float(np.median(beta[st]))
        cv = np.abs(kap) > 0.008
        x_est = -(beta - gamma) / np.where(cv, kap, np.nan)
        o = {"n": int(len(a)), "gamma_deg": float(np.degrees(gamma)),
             "gamma_p10_p90_deg": [float(np.degrees(np.percentile(beta[st], 10))),
                                   float(np.degrees(np.percentile(beta[st], 90)))],
             "x_ref_by_speed": []}
        for lo, hi in ((3, 6), (6, 9), (9, 12), (12, 16), (16, 30)):
            m = cv & (v >= lo) & (v < hi)
            if m.sum() < 50:
                continue
            o["x_ref_by_speed"].append({"v": "%d-%d" % (lo, hi), "n": int(m.sum()),
                                        "x_ref_med_m": float(np.median(x_est[m])),
                                        "x_ref_lsq_m": float(-np.linalg.lstsq(
                                            kap[m][:, None], beta[m] - gamma, rcond=None)[0][0])})
        if who == "npc":
            o["x_ref_by_length_v_lt_9"] = []
            for lo, hi in ((3.5, 4.4), (4.4, 4.7), (4.7, 5.0), (5.0, 5.5), (5.5, 7.0), (7.0, 13.0)):
                m = cv & (v < 9) & (L >= lo) & (L < hi)
                if m.sum() < 50:
                    continue
                o["x_ref_by_length_v_lt_9"].append({"L": "%.1f-%.1f" % (lo, hi), "n": int(m.sum()),
                                                    "x_ref_med_m": float(np.median(x_est[m]))})
        out[who] = o
    return out


# --------------------------------------------------------------------------
# 2. LiDAR faces
# --------------------------------------------------------------------------
def lidar_stamps(path):
    T = gc.extract_targets(path)          # recorded LiDAR-mode targets: stamp == cloud stamp
    return T["t"]


def ego_lane(lanes, x, y):
    best, lid, dd, ss, yy = np.inf, -1, np.nan, np.nan, np.nan
    for k, l in lanes.items():
        s, d, yw, _ = l.project(np.array([x]), np.array([y]), 0.0)
        if 1.0 < s[0] < l.length - 1.0 and abs(d[0]) < best:
            best, lid, dd, ss, yy = abs(d[0]), k, d[0], s[0], yw[0]
    return lid, dd, ss, yy


def candidates(path, lanes, gamma, max_per_episode=25, stride=2):
    d = bt.load_bag(path)
    G = gc.GTFrames(d)
    tr = bt.gt_tracks(d, {"dx": gc.OFF_X, "dy": gc.OFF_Y})
    o = d["odom"]
    stamps = lidar_stamps(path)
    out = []
    count = {}
    for t in stamps[::stride]:
        if t < o["t"][0] or t > o["t"][-1]:
            continue
        ex, ey, eyaw = gc.interp_pose(o, t)
        ev = float(np.interp(t, o["t"], o["v"]))
        # straight driving only (yaw rate)
        yr = (np.interp(t + 0.1, o["t"], np.unwrap(o["yaw"])) -
              np.interp(t - 0.1, o["t"], np.unwrap(o["yaw"]))) / 0.2
        if abs(yr) > 0.03:
            continue
        lid, de, se, ly = ego_lane(lanes, ex, ey)
        if lid < 0 or abs(de) > 0.8:
            continue
        i = int(np.clip(np.searchsorted(G.t, t), 0, len(G.t) - 1))
        f = G.frame(i)
        dt = t - G.t[i]
        fx = f["x"] + f["v"] * np.cos(f["yaw"] - gamma) * dt
        fy = f["y"] + f["v"] * np.sin(f["yaw"] - gamma) * dt
        s, dd, yw, _ = lanes[lid].project(fx, fy, 0.0)
        ds = s - se
        hdg = np.abs(wrap(f["yaw"] - gamma - yw))
        same = (np.abs(dd) < 0.8) & (hdg < np.radians(5))
        sens_s = se + LIDAR_X
        for side in ("ahead", "behind"):
            if side == "ahead":
                face = (s - f["length"] / 2) - sens_s
                m = same & (ds > 0)
                rng = face
            else:
                face = sens_s - (s + f["length"] / 2)
                m = same & (ds < 0)
                rng = face
            if not m.any():
                continue
            j = np.flatnonzero(m)[np.argmin(np.abs(ds[m]))]
            if not (5.0 <= rng[j] <= 40.0):
                continue
            u = int(f["uid"][j])
            key = (u, side)
            if count.get(key, 0) >= max_per_episode:
                continue
            count[key] = count.get(key, 0) + 1
            out.append({"t": float(t), "uid": u, "side": side, "range": float(rng[j]),
                        "ego_v": ev, "npc_v": float(f["v"][j]), "L": float(f["length"][j]),
                        "W": float(f["width"][j]), "lane": int(lid)})
        # alongside: adjacent lane, longitudinal overlap with the sensor +-3 m
        for k in lanes:
            if abs(k - lid) != 1:
                continue
            s2, d2, y2, _ = lanes[k].project(fx, fy, 0.0)
            e2 = lanes[k].project(np.array([ex]), np.array([ey]), 0.0)[0][0]
            ds2 = s2 - (e2 + LIDAR_X)
            m = (np.abs(d2) < 0.6) & (np.abs(wrap(f["yaw"] - gamma - y2)) < np.radians(4)) & \
                (np.abs(ds2) < 1.5)
            for j in np.flatnonzero(m):
                u = int(f["uid"][j])
                key = (u, "side")
                if count.get(key, 0) >= max_per_episode:
                    continue
                count[key] = count.get(key, 0) + 1
                out.append({"t": float(t), "uid": u, "side": "side", "range": float(ds2[j]),
                            "ego_v": ev, "npc_v": float(f["v"][j]), "L": float(f["length"][j]),
                            "W": float(f["width"][j]), "lane": int(k)})
    return out, tr


def read_clouds(path, stamps):
    import rosbag
    import rospy
    want = sorted(set(round(s, 6) for s in stamps))
    clouds = {}
    if not want:
        return clouds
    with rosbag.Bag(path) as bag:
        # bag receive time is the stamp + a few ms; read in coarse windows
        groups = []
        for s in want:
            if groups and s - groups[-1][-1] < 2.0:
                groups[-1].append(s)
            else:
                groups.append([s])
        for g in groups:
            for _, m, _ in bag.read_messages(topics=["/velodyne_points"],
                                             start_time=rospy.Time.from_sec(g[0] - 0.3),
                                             end_time=rospy.Time.from_sec(g[-1] + 0.5)):
                st = round(m.header.stamp.to_sec(), 6)
                if st in want:
                    a = np.frombuffer(m.data, dtype=PT)
                    clouds[st] = np.c_[a["x"], a["y"], a["z"], a["time"]].astype(np.float64)
    return clouds


def track_pose(tr, t, gamma):
    x = np.interp(t, tr["t"], tr["x"])
    y = np.interp(t, tr["t"], tr["y"])
    h = np.interp(t, tr["t"], np.unwrap(np.radians(tr["heading"]))) - gamma
    return x, y, h


def measure(c, cloud, tr_u, o, gamma, hyp, yaw_bias=True):
    t = c["t"]
    P = cloud
    tc = t + P[:, 3] if hyp == "H1" else np.full(len(P), t)
    ex = np.interp(tc, o["t"], o["x"])
    ey = np.interp(tc, o["t"], o["y"])
    eyaw = np.interp(tc, o["t"], np.unwrap(o["yaw"])) - (gamma if yaw_bias else 0.0)
    ce, se = np.cos(eyaw), np.sin(eyaw)
    xs, ys = P[:, 0] + LIDAR_X, P[:, 1]
    mx = ex + ce * xs - se * ys
    my = ey + se * xs + ce * ys
    # NPC pose at the capture time of each point
    tn = t + P[:, 3]
    nx, ny, nh = track_pose(tr_u, tn, gamma)
    cn, sn = np.cos(nh), np.sin(nh)
    dx, dy = mx - nx, my - ny
    lx, ly = cn * dx + sn * dy, -sn * dx + cn * dy
    L, W = c["L"], c["W"]
    z = P[:, 2]
    corr = (np.abs(ly) < W / 2 + 1.5) & (lx > -L / 2 - 6) & (lx < L / 2 + 6)
    if corr.sum() < 10:
        return None
    zg = np.percentile(z[corr], 10)
    box = (np.abs(ly) < W / 2 + 0.3) & (lx > -L / 2 - 2.5) & (lx < L / 2 + 2.5) & \
        (z > zg + 0.3) & (z < zg + 3.5)
    n = int(box.sum())
    if n < 4:
        return {"n": n}
    lxb = lx[box]
    lo, hi = np.min(lxb), np.max(lxb)
    rear = float(np.median(lxb[lxb < lo + 0.10]))
    front = float(np.median(lxb[lxb > hi - 0.10]))
    lat = float(np.median(ly[box]))
    out = {"n": n, "rear_face": rear, "front_face": front, "lat_med": lat,
           "off_rear": rear + L / 2, "off_front": front - L / 2,
           "obs_len": front - rear, "t_pts": float(np.median(P[box, 3])),
           "zg": float(zg)}
    # bumper zone only (0.25-0.9 m above the local ground): the face extreme
    bz = box & (z < zg + 0.9)
    out["n_bumper"] = int(bz.sum())
    if bz.sum() >= 3:
        lz = lx[bz]
        out["off_rear_bumper"] = float(np.min(lz)) + L / 2
        out["off_front_bumper"] = float(np.max(lz)) - L / 2
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", default=["~/bags/lc_gt_2026-09-*.bag"])
    ap.add_argument("--kin-bags", default="~/bags/lc_gt_2026-09-*.bag")
    ap.add_argument("--out", default=os.path.join(gc.RESULTS, "gtcheck_refpoint.json"))
    a = ap.parse_args()
    lanes = bt.load_lanes()
    res = {}
    print("[refpoint] kinematic test", file=sys.stderr)
    res["kinematic"] = kin = kinematic(bt.expand_bags([a.kin_bags]))
    gamma = np.radians(kin["ego"]["gamma_deg"])
    rows = []
    for p in bt.expand_bags(a.bags):
        lab = bt.bag_label(p)
        cands, tr = candidates(p, lanes, gamma)
        print("[refpoint] %s: %d candidate scans" % (lab, len(cands)), file=sys.stderr)
        clouds = read_clouds(p, [c["t"] for c in cands])
        o = bt.load_bag(p)["odom"]
        for c in cands:
            cl = clouds.get(round(c["t"], 6))
            if cl is None:
                continue
            row = dict(c, bag=lab)
            for hyp in ("H1", "H2"):
                m = measure(c, cl, tr[c["uid"]], o, gamma, hyp)
                if m is None:
                    continue
                for k, v in m.items():
                    row["%s_%s" % (hyp, k)] = v
            m = measure(c, cl, tr[c["uid"]], o, gamma, "H1", yaw_bias=False)
            if m and "lat_med" in m:
                row["H1raw_lat_med"] = m["lat_med"]
            rows.append(row)
    keys = sorted(set().union(*[r.keys() for r in rows])) if rows else []
    with open(os.path.join(gc.RESULTS, "gtcheck_refpoint_faces.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    res["lidar"] = summarize(rows)
    bt.write_json(a.out, res)
    print("wrote", a.out, file=sys.stderr)


def summarize(rows):
    out = {}
    for hyp in ("H1", "H2"):
        o = {}
        for side, key in (("ahead", "off_rear"), ("behind", "off_front")):
            for est in ("", "_bumper"):
                k = "%s_%s%s" % (hyp, key, est)
                sel = [r for r in rows if r["side"] == side and r.get("%s_n" % hyp, 0) >= 8
                       and k in r and r[k] == r[k]]
                if not sel:
                    continue
                v = np.array([r[k] for r in sel])
                rel = np.array([r["npc_v"] - r["ego_v"] for r in sel])
                egv = np.array([r["ego_v"] for r in sel])
                rng = np.array([r["range"] for r in sel])
                st = egv < 0.3
                per_npc = {}
                for r, x in zip(sel, v):
                    per_npc.setdefault((r["bag"], r["uid"]), []).append(x)
                oo = {"n_scans": len(sel), "n_npcs": len(per_npc),
                      "offset_m": bt.stats(v),
                      "offset_per_npc_median_m": bt.stats([np.median(x) for x in per_npc.values()]),
                      "offset_ego_stopped_m": bt.stats(v[st]),
                      "offset_ego_moving_m": bt.stats(v[~st]),
                      "corr_with_ego_v": float(np.corrcoef(v, egv)[0, 1]) if len(v) > 3 and np.std(egv) > 0 else None,
                      "corr_with_rel_v": float(np.corrcoef(v, rel)[0, 1]) if len(v) > 3 and np.std(rel) > 0 else None,
                      "by_range": {}, "by_length": {}}
                for lo, hi in ((5, 15), (15, 25), (25, 40)):
                    m = (rng >= lo) & (rng < hi)
                    if m.sum() >= 3:
                        oo["by_range"]["%d-%d" % (lo, hi)] = bt.stats(v[m])
                for lo, hi in ((3.5, 4.7), (4.7, 5.2), (5.2, 7.0), (7.0, 13.0)):
                    m = np.array([lo <= r["L"] < hi for r in sel], bool)
                    if m.sum() >= 3:
                        oo["by_length"]["%.1f-%.1f" % (lo, hi)] = bt.stats(v[m])
                o[side + est] = oo
        sel = [r for r in rows if r["side"] == "side" and r.get("%s_n" % hyp, 0) >= 15]
        if sel:
            ol = np.array([r["%s_obs_len" % hyp] - r["L"] for r in sel])
            mid = np.array([0.5 * (r["%s_rear_face" % hyp] + r["%s_front_face" % hyp]) for r in sel])
            o["side"] = {"n_scans": len(sel), "obs_len_minus_L": bt.stats(ol),
                         "midpoint_offset_m": bt.stats(mid)}
        out[hyp] = o
    sel = [r for r in rows if r["side"] in ("ahead", "behind") and r.get("H1_n", 0) >= 8
           and "H1raw_lat_med" in r and "H1_lat_med" in r]
    sg = np.array([1.0 if r["side"] == "ahead" else -1.0 for r in sel])
    rr = np.array([max(r["range"] + 1.0, 1.0) for r in sel])
    out["lateral_check"] = {
        "n": len(sel),
        "lat_med_yaw_corrected_m": bt.stats([r["H1_lat_med"] for r in sel]),
        "lat_med_raw_odom_yaw_m": bt.stats([r["H1raw_lat_med"] for r in sel]),
        "raw_minus_corrected_per_m_signed_deg": bt.stats(np.degrees(
            np.array([r["H1raw_lat_med"] - r["H1_lat_med"] for r in sel]) * sg / rr))}
    return out


if __name__ == "__main__":
    main()
