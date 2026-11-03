#!/usr/bin/env python3
"""Shared extraction and GT recomputation for the gtcheck_* scripts.

Reuses bagtools (GT /Object_topic, /odom, status, recorded LiDAR targets) and
adds a full-field /smpc/targets extractor (replayed object-mode bags and the
recorded LiDAR-mode topic) plus a faithful Python port of the target-selector
role rules (research/docs/baseline_spec.md section 5, sel:841-1012).

Conventions (docs/PROGRESS_0910.md section 5): GT matched by unique_id, time =
header.stamp, /Object_topic velocity km/h (vehicle frame x = longitudinal).
GT positions are MORAI scenario-local; UTM = local + (302595.0, 4124145.0)
(config/target_selector.yaml object_topic_offset_x/y).
"""
import math
import os
import pickle

import numpy as np

import bagtools as bt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(os.path.dirname(HERE), "results")
CACHE = os.path.join(RESULTS, "gtcheck_cache")
REPLAY_DIR = os.path.join(RESULTS, "gtcheck_replay")
OFF_X, OFF_Y = 302595.0, 4124145.0
KMH = 1.0 / 3.6
# MORAI headings (npc.heading, /Ego_topic, /imu, /odom yaw) are +1.365 deg CCW of
# the UTM grid of the positions/CSVs (grid convergence, measured by
# gtcheck_refpoint.py on straight driving).  Used only for GT "truth" here.
HEADING_BIAS_RAD = math.radians(1.365)

# target_selector.yaml values used by the role rules
LANE_HALF_WIDTH = 2.2
BBOX_MARGIN = 0.2
NEARBY_MARGIN = 1.0
REP_MAX_ABS_D = 1.35
MAX_FRONT = 120.0
MAX_REAR = 120.0
EXT_BEFORE = 100.0
NEARBY_FRONT = 150.0
NEARBY_REAR = 120.0
MAX_NEARBY = 40

VFIELDS = ["valid", "uid", "lane_id", "x", "y", "s", "d", "delta_s", "yaw", "yaw_rate",
           "v_long", "v_lat", "length", "width", "half_s", "half_d", "front_delta_s",
           "rear_delta_s", "track_hits"]
ROLES = ["current_front", "target_front", "target_rear"]


def _v(v):
    g = lambda name: float(getattr(v, name, float("nan")))   # older bags lack some fields
    return (float(v.valid), float(v.unique_id), float(v.lane_id), v.x, v.y, v.s, v.d,
            v.delta_s, g("yaw"), g("yaw_rate"), g("v_long"), g("v_lat"), g("length"),
            g("width"), g("half_s"), g("half_d"), g("front_delta_s"), g("rear_delta_s"),
            g("track_hits"))


def extract_targets(path, topic="/smpc/targets"):
    """Full-field TargetVehicleSet extraction (cached)."""
    os.makedirs(CACHE, exist_ok=True)
    st = os.stat(path)
    cp = os.path.join(CACHE, "tg_%s_%s_%d.pkl" % (os.path.basename(path)[:-4],
                                                  topic.strip("/").replace("/", "_"), st.st_size))
    if os.path.exists(cp) and os.path.getmtime(cp) >= st.st_mtime:
        with open(cp, "rb") as f:
            return pickle.load(f)
    import rosbag
    head, reps, near = [], [], []
    with rosbag.Bag(path) as bag:
        for _, m, t in bag.read_messages(topics=[topic]):
            k = len(head)
            ga = lambda name: float(getattr(m, name, float("nan")))
            head.append((m.header.stamp.to_sec(), t.to_sec(), m.current_lane_id, m.target_lane_id,
                         m.ego_s_current, m.ego_d_current, m.ego_d_target,
                         ga("current_lane_length"), ga("distance_to_lane_end"),
                         ga("lane_change_required"), ga("lane_change_prepare"),
                         ga("lane_change_urgent"), ga("emergency_stop_required"),
                         len(getattr(m, "nearby_vehicles", []))))
            reps.append(_v(m.current_front) + _v(m.target_front) + _v(m.target_rear))
            for v in getattr(m, "nearby_vehicles", []):
                near.append((k,) + _v(v) + (hash(v.role) % 1000,
                                            {"nearby_front": 0, "nearby_rear": 1,
                                             "nearby_overlap": 2}.get(v.role, 9)))
    H = np.asarray(head, float) if head else np.zeros((0, 14))
    out = {kname: H[:, i] for i, kname in enumerate(
        ["t", "bag_t", "cur_lane", "tgt_lane", "ego_s_cur", "ego_d_cur", "ego_d_tgt",
         "cur_len", "dist_to_end", "lc_required", "prepare", "urgent", "emergency", "n_near"])}
    R = np.asarray(reps, float) if reps else np.zeros((0, 3 * len(VFIELDS)))
    nf = len(VFIELDS)
    for j, role in enumerate(ROLES):
        out[role] = {f: R[:, j * nf + i] for i, f in enumerate(VFIELDS)}
    N = np.asarray(near, float) if near else np.zeros((0, 1 + nf + 2))
    out["nearby"] = {"k": N[:, 0].astype(int)}
    for i, f in enumerate(VFIELDS):
        out["nearby"][f] = N[:, 1 + i]
    out["nearby"]["role"] = N[:, -1].astype(int)
    with open(cp, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return out


# --------------------------------------------------------------------------
# GT frames
# --------------------------------------------------------------------------
class GTFrames:
    """Per-/Object_topic-message GT NPC arrays (UTM, SI units)."""

    def __init__(self, d, off=(OFF_X, OFF_Y)):
        g = d["gt"]
        self.t = d["gt_msg_t"]
        self.bag_t = d["gt_msg_bag_t"]
        k = g["k"].astype(int)
        order = np.argsort(k, kind="stable")       # keep npc_list order inside a frame
        self.k = k[order]
        self.uid = g["uid"][order].astype(int)
        self.x = g["x"][order] + off[0]
        self.y = g["y"][order] + off[1]
        self.heading_deg = g["heading"][order]
        self.yaw = np.radians(self.heading_deg)
        self.v = g["vx_kmh"][order] * KMH
        self.v_lat = g["vy_kmh"][order] * KMH
        self.length = np.maximum(g["sx"][order], g["sy"][order])
        self.width = np.minimum(g["sx"][order], g["sy"][order])
        self.start = np.searchsorted(self.k, np.arange(len(self.t)))
        self.end = np.searchsorted(self.k, np.arange(len(self.t)), side="right")
        self.by_stamp = {}
        for i, s in enumerate(self.t):
            self.by_stamp.setdefault(round(s, 6), i)

    def frame(self, i):
        sl = slice(self.start[i], self.end[i])
        return {"uid": self.uid[sl], "x": self.x[sl], "y": self.y[sl], "yaw": self.yaw[sl],
                "v": self.v[sl], "v_lat": self.v_lat[sl], "length": self.length[sl],
                "width": self.width[sl], "heading_deg": self.heading_deg[sl]}

    def index_of(self, stamp):
        return self.by_stamp.get(round(stamp, 6))


def interp_pose(o, t):
    x = np.interp(t, o["t"], o["x"])
    y = np.interp(t, o["t"], o["y"])
    yaw = np.interp(t, o["t"], np.unwrap(o["yaw"]))
    return x, y, yaw


def zoh_pose(o, t, lag=0.0):
    i = np.searchsorted(o["t"], np.asarray(t) - lag, side="right") - 1
    i = np.clip(i, 0, len(o["t"]) - 1)
    return o["x"][i], o["y"][i], o["yaw"][i]


# --------------------------------------------------------------------------
# Lane helpers (bagtools.Lane semantics == lane_path.hpp project/projectExtended)
# --------------------------------------------------------------------------
def footprint(yaw, lane_yaw, length, width):
    dy = np.arctan2(np.sin(yaw - lane_yaw), np.cos(yaw - lane_yaw))
    c, s = np.abs(np.cos(dy)), np.abs(np.sin(dy))
    return 0.5 * (length * c + width * s), 0.5 * (length * s + width * c)


def gt_lane(lanes, x, y, ext=False, max_abs_d=1.75):
    """Physical lane of each point: nearest centre-line with |d| <= max_abs_d.

    Interior only (s strictly inside the CSV), or with the 100 m before-start
    tangent extension when ext=True.  Returns lane (-1 = none), d, s, yaw,
    and per-lane (s, d, yaw, interior)."""
    n = len(x)
    best = np.full(n, np.inf)
    lane = np.full(n, -1)
    dd = np.full(n, np.nan)
    ss = np.full(n, np.nan)
    yy = np.full(n, np.nan)
    per = {}
    for lid, l in lanes.items():
        s, d, yw, _ = l.project(x, y, EXT_BEFORE if ext else 0.0)
        lo = -EXT_BEFORE if ext else 0.0
        interior = (s > lo + 0.05) & (s < l.length - 0.05)
        per[lid] = (s, d, yw, interior)
        ok = interior & (np.abs(d) < best)
        best = np.where(ok, np.abs(d), best)
        lane = np.where(ok, lid, lane)
        dd = np.where(ok, d, dd)
        ss = np.where(ok, s, ss)
        yy = np.where(ok, yw, yy)
    on = best <= max_abs_d
    return np.where(on, lane, -1), dd, ss, yy, per


def roles_from_proj(f, pc, pt, ego_s_c, ego_s_t):
    """Python port of publishTargetsFromCandidates role selection (spec section 5).

    f: frame dict (uid, yaw, length, width) in npc_list order.
    pc / pt: (s, d, yaw) of every NPC on the current lane (project, no extension)
    and on the target lane (projectExtended 100 m).  ego_s_c / ego_s_t: ego
    rear-axle s on the current / target lane.  Returns role -> dict."""
    pcs, pcd, pcy = pc
    pts, ptd, pty = pt
    fcs, fcd = footprint(f["yaw"], pcy, f["length"], f["width"])
    fts, ftd = footprint(f["yaw"], pty, f["length"], f["width"])
    thr = LANE_HALF_WIDTH + BBOX_MARGIN
    clear_c = np.abs(pcd) - fcd
    clear_t = np.abs(ptd) - ftd
    on_c = clear_c <= thr
    on_t = clear_t <= thr
    asg_c = on_c & (~on_t | (clear_c <= clear_t))
    asg_t = on_t & (~on_c | (clear_t < clear_c))
    rep_c = asg_c & (np.abs(pcd) <= REP_MAX_ABS_D)
    rep_t = asg_t & (np.abs(ptd) <= REP_MAX_ABS_D)
    ds_c = pcs - ego_s_c
    ds_t = pts - ego_s_t

    def pick(mask, key, limit):
        # strict '<' against a running minimum in list order
        best, bi = limit, -1
        for i in np.flatnonzero(mask):
            if key[i] < best:
                best, bi = key[i], i
        return bi, best
    i_cf, g_cf = pick(rep_c & (ds_c + fcs > 0.0), np.maximum(0.0, ds_c - fcs), MAX_FRONT)
    i_tf, g_tf = pick(rep_t & (ds_t + fts >= 0.0), np.maximum(0.0, ds_t - fts), MAX_FRONT)
    i_tr, g_tr = pick(rep_t & (ds_t - fts <= 0.0), np.maximum(0.0, -(ds_t + fts)), MAX_REAR)
    out = {}
    for role, i, gap, ds, hs, d in (("current_front", i_cf, g_cf, ds_c, fcs, pcd),
                                    ("target_front", i_tf, g_tf, ds_t, fts, ptd),
                                    ("target_rear", i_tr, g_tr, ds_t, fts, ptd)):
        if i < 0:
            out[role] = {"uid": -1}
        else:
            out[role] = {"uid": int(f["uid"][i]), "i": int(i), "gap": float(gap),
                         "delta_s": float(ds[i]), "half_s": float(hs[i]),
                         "front_delta_s": float(ds[i] + hs[i]),
                         "rear_delta_s": float(ds[i] - hs[i]), "d": float(d[i])}
    out["_aux"] = {"rep_c": rep_c, "rep_t": rep_t, "asg_c": asg_c, "asg_t": asg_t,
                   "on_c": on_c, "on_t": on_t, "ds_c": ds_c, "ds_t": ds_t, "fcs": fcs,
                   "fts": fts, "fcd": fcd, "ftd": ftd, "pcd": pcd, "ptd": ptd,
                   "clear_c": clear_c, "clear_t": clear_t}
    return out


def select_roles(lanes, cur, tgt, ex, ey, f):
    """Role selection from raw positions (projects everything itself)."""
    C, T = lanes[cur], lanes[tgt]
    ecs = C.project(np.array([ex]), np.array([ey]), 0.0)[0][0]
    ets = T.project(np.array([ex]), np.array([ey]), EXT_BEFORE)[0][0]
    pc = C.project(f["x"], f["y"], 0.0)[:3]
    pt = T.project(f["x"], f["y"], EXT_BEFORE)[:3]
    return roles_from_proj(f, pc, pt, ecs, ets)


def nearby_gt(lanes, cur, ex, ey, f, ego_proj_cache=None):
    """Port of makeNearbyVehicle: best lane per NPC (or -1) and its delta_s."""
    n = len(f["x"])
    best = np.full(n, np.inf)
    lane = np.full(n, -1)
    ds_out = np.full(n, np.nan)
    thr = LANE_HALF_WIDTH + BBOX_MARGIN + NEARBY_MARGIN
    for lid, l in lanes.items():
        s, d, yw, _ = l.project(f["x"], f["y"], EXT_BEFORE)
        es, _, _, _ = l.project(np.array([ex]), np.array([ey]), EXT_BEFORE)
        hs, hd = footprint(f["yaw"], yw, f["length"], f["width"])
        clear = np.abs(d) - hd
        ds = s - es[0]
        ok = (clear <= thr) & ~((ds + hs) < -NEARBY_REAR) & ~((ds - hs) > NEARBY_FRONT)
        score = np.maximum(0.0, clear) + 0.01 * np.abs(ds) + 0.001 * abs(lid - cur)
        better = ok & (score < best)
        best = np.where(better, score, best)
        lane = np.where(better, lid, lane)
        ds_out = np.where(better, ds, ds_out)
    return lane, ds_out


def lane_xy(lane, s, d):
    """Point at (s, d) on a lane (inverse projection, straight-segment exact)."""
    i = int(np.clip(np.searchsorted(lane.cum_s, s) - 1, 0, len(lane.seg) - 1))
    v = lane.seg[i] / max(lane.seg_len[i], 1e-9)
    t = s - lane.cum_s[i]
    px, py = lane.P[i] + t * v
    return px - d * v[1], py + d * v[0]


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))
