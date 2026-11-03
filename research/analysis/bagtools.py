#!/usr/bin/env python3
"""Shared offline helpers for the Day-1 SMPC lane-change measurements.

Offline only: reads rosbags with the python rosbag API (always with a topic
filter), never starts ROS nodes.  Measurement rules (docs/PROGRESS_0910.md S5):
  * GT vehicles are matched by unique_id, never by npc_list index.
  * Time axis is header.stamp.  Topics without a header (std_msgs/String,
    std_msgs/Int32, waypoint_maker/State) fall back to bag receive time; in
    these bags header stamps and receive times share the same wall clock
    (offset of a few ms), which is noted where it matters.
  * /Object_topic npc velocity is km/h, /Ego_topic velocity is m/s.

Coordinates: /Object_topic and /Ego_topic positions are MORAI-local, /odom and
the lane CSVs are UTM.  The constant local->UTM offset is estimated per bag
from /Ego_topic vs /odom (both the IONIQ 5 rear axle) and applied to GT NPCs.

Extraction results are cached per bag (pickle) so the measurement scripts can
be re-run quickly.  Cache default: research/results/cache/ (git-ignored).
"""
import argparse
import glob
import math
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(RESEARCH, "results")
DEFAULT_CACHE = os.path.join(RESULTS_DIR, "cache")
DEFAULT_LANE_DIR = os.path.expanduser("~/catkin_ws/src/waypoint_system/data/default")
EXTRACT_VERSION = 3

TOPICS = [
    "/Object_topic", "/Ego_topic", "/odom", "/gps_state", "/path_number",
    "/path_switch_preview", "/smpc/targets", "/smpc/lane_change_status",
    "/smpc/lane_change_supervisor_debug", "/smpc/decision",
]
KMH = 1.0 / 3.6


# --------------------------------------------------------------------------
# CLI helpers
# --------------------------------------------------------------------------
def base_argparser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--bags", nargs="+", default=None,
                   help="bag files or globs (default: ~/bags/lc_gt_*.bag)")
    p.add_argument("--cache", default=DEFAULT_CACHE, help="extraction cache dir")
    p.add_argument("--no-cache", action="store_true", help="ignore/overwrite cache")
    p.add_argument("--lanes", default=DEFAULT_LANE_DIR, help="lane CSV directory")
    p.add_argument("--out", default=RESULTS_DIR, help="results directory")
    p.add_argument("--jobs", type=int, default=6, help="parallel bag extraction")
    return p


def expand_bags(patterns):
    if not patterns:
        patterns = [os.path.expanduser("~/bags/lc_gt_*.bag")]
    out = []
    for p in patterns:
        p = os.path.expanduser(p)
        hits = sorted(glob.glob(p))
        out.extend(hits if hits else ([p] if os.path.exists(p) else []))
    seen, uniq = set(), []
    for b in out:
        if b not in seen:
            uniq.append(b)
            seen.add(b)
    return uniq


def bag_label(path):
    b = os.path.basename(path)
    return b[len("lc_gt_"):-4] if b.startswith("lc_gt_") else b[:-4]


def stats(values):
    """Distribution summary with N/mean/median/p10/p90/min/max."""
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p90": float(np.percentile(a, 90)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


def fmt_stats(s, digits=2):
    if not s or s.get("n", 0) == 0:
        return "n=0"
    f = "%%.%df" % digits
    return ("n=%d mean=" + f + " med=" + f + " p10=" + f + " p90=" + f +
            " [" + f + ", " + f + "]") % (
        s["n"], s["mean"], s["median"], s["p10"], s["p90"], s["min"], s["max"])


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def conv(o):
        if isinstance(o, (np.floating,)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return [conv(x) for x in o.tolist()]
        if isinstance(o, float):
            return None if not math.isfinite(o) else o
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        return o

    import json
    with open(path, "w") as f:
        json.dump(conv(obj), f, indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------
# Bag extraction
# --------------------------------------------------------------------------
def _stamp(msg, bag_t):
    h = getattr(msg, "header", None)
    if h is not None:
        s = h.stamp.to_sec()
        if s > 0:
            return s
    return bag_t.to_sec()


def _veh_tuple(v):
    return (bool(v.valid), int(v.unique_id), int(v.lane_id), float(v.x), float(v.y),
            float(v.s), float(v.d), float(v.delta_s), float(v.v_long),
            float(getattr(v, "half_s", float("nan"))),
            int(getattr(v, "track_hits", -1)))


VEH_FIELDS = ["valid", "uid", "lane_id", "x", "y", "s", "d", "delta_s", "v_long",
              "half_s", "track_hits"]


def extract_bag(path):
    import rosbag  # noqa: imported lazily so --help works without ROS

    gt_t, gt_rows = [], []        # per-message stamps, flattened npc rows
    ego, odom, gps, pathn, prev, status, sup, dec = [], [], [], [], [], [], [], []
    tg_head, tg_reps, tg_near = [], [], []
    with rosbag.Bag(path) as bag:
        for topic, msg, t in bag.read_messages(topics=TOPICS):
            if topic == "/Object_topic":
                st = _stamp(msg, t)
                k = len(gt_t)
                gt_t.append((st, t.to_sec()))
                for n in msg.npc_list:
                    gt_rows.append((k, int(n.unique_id), int(n.type), float(n.position.x),
                                    float(n.position.y), float(n.heading),
                                    float(n.velocity.x), float(n.velocity.y),
                                    float(n.acceleration.x), float(n.size.x),
                                    float(n.size.y)))
            elif topic == "/Ego_topic":
                ego.append((_stamp(msg, t), t.to_sec(), msg.position.x, msg.position.y,
                            msg.heading, msg.velocity.x, msg.velocity.y,
                            msg.acceleration.x))
            elif topic == "/odom":
                q = msg.pose.pose.orientation
                yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                odom.append((_stamp(msg, t), msg.pose.pose.position.x,
                             msg.pose.pose.position.y, yaw,
                             math.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y)))
            elif topic == "/gps_state":
                gps.append((t.to_sec(), int(msg.lane_number), int(msg.current_state)))
            elif topic == "/path_number":
                pathn.append((t.to_sec(), int(msg.data)))
            elif topic == "/path_switch_preview":
                prev.append((_stamp(msg, t), bool(msg.valid), int(msg.mode),
                             int(msg.source_path_number), int(msg.target_path_number),
                             float(msg.blend_duration_sec)))
            elif topic == "/smpc/lane_change_status":
                status.append((_stamp(msg, t), t.to_sec(), bool(msg.active),
                               int(msg.source_lane_id), int(msg.target_lane_id),
                               float(msg.elapsed_sec)))
            elif topic == "/smpc/lane_change_supervisor_debug":
                sup.append((t.to_sec(), msg.data))
            elif topic == "/smpc/decision":
                dec.append((_stamp(msg, t), int(msg.mode), bool(msg.request), msg.reason))
            elif topic == "/smpc/targets":
                k = len(tg_head)
                tg_head.append((_stamp(msg, t), t.to_sec(), msg.current_lane_id,
                                msg.target_lane_id, msg.ego_s_current, msg.ego_d_current,
                                msg.ego_d_target, msg.distance_to_lane_end,
                                bool(msg.lane_change_required), bool(msg.lane_change_prepare),
                                bool(msg.lane_change_urgent)))
                tg_reps.append(_veh_tuple(msg.current_front) + _veh_tuple(msg.target_front) +
                               _veh_tuple(msg.target_rear))
                for v in msg.nearby_vehicles:
                    tg_near.append((k,) + _veh_tuple(v))

    def arr(rows, ncol):
        return np.asarray(rows, float) if rows else np.zeros((0, ncol))

    d = {"path": path, "label": bag_label(path), "version": EXTRACT_VERSION}
    gt_t = arr(gt_t, 2)
    d["gt_msg_t"] = gt_t[:, 0]
    d["gt_msg_bag_t"] = gt_t[:, 1]
    g = arr(gt_rows, 11)
    keys = ["k", "uid", "type", "x", "y", "heading", "vx_kmh", "vy_kmh", "ax", "sx", "sy"]
    d["gt"] = {kname: g[:, i] for i, kname in enumerate(keys)}
    e = arr(ego, 8)
    d["ego"] = {kname: e[:, i] for i, kname in
                enumerate(["t", "bag_t", "x", "y", "heading", "vx", "vy", "ax"])}
    o = arr(odom, 5)
    d["odom"] = {kname: o[:, i] for i, kname in enumerate(["t", "x", "y", "yaw", "v"])}
    gp = arr(gps, 3)
    d["gps"] = {kname: gp[:, i] for i, kname in enumerate(["t", "lane", "state"])}
    pn = arr(pathn, 2)
    d["path_number"] = {"t": pn[:, 0], "data": pn[:, 1]}
    pv = arr(prev, 6)
    d["preview"] = {kname: pv[:, i] for i, kname in
                    enumerate(["t", "valid", "mode", "src", "tgt", "blend_sec"])}
    s = arr(status, 6)
    d["status"] = {kname: s[:, i] for i, kname in
                   enumerate(["t", "bag_t", "active", "src", "tgt", "elapsed"])}
    d["sup"] = {"t": np.asarray([r[0] for r in sup], float), "text": [r[1] for r in sup]}
    d["decision"] = {"t": np.asarray([r[0] for r in dec], float),
                     "mode": np.asarray([r[1] for r in dec], int),
                     "request": np.asarray([r[2] for r in dec], bool),
                     "reason": [r[3] for r in dec]}
    th = arr(tg_head, 11)
    d["targets"] = {kname: th[:, i] for i, kname in enumerate(
        ["t", "bag_t", "cur_lane", "tgt_lane", "ego_s_cur", "ego_d_cur", "ego_d_tgt",
         "dist_to_end", "lc_required", "prepare", "urgent"])}
    tr = arr(tg_reps, 3 * len(VEH_FIELDS))
    nf = len(VEH_FIELDS)
    for j, role in enumerate(["current_front", "target_front", "target_rear"]):
        d["targets"][role] = {f: tr[:, j * nf + i] for i, f in enumerate(VEH_FIELDS)}
    tn = arr(tg_near, 1 + nf)
    d["targets"]["nearby"] = {"k": tn[:, 0]}
    for i, f in enumerate(VEH_FIELDS):
        d["targets"]["nearby"][f] = tn[:, 1 + i]
    return d


def _cache_path(cache_dir, path):
    st = os.stat(path)
    return os.path.join(cache_dir, "%s_%d_v%d.pkl" % (
        os.path.basename(path)[:-4], st.st_size, EXTRACT_VERSION))


def load_bag(path, cache_dir=DEFAULT_CACHE, use_cache=True):
    os.makedirs(cache_dir, exist_ok=True)
    gi = os.path.join(cache_dir, ".gitignore")
    if not os.path.exists(gi):
        with open(gi, "w") as f:
            f.write("*\n")
    cp = _cache_path(cache_dir, path)
    if use_cache and os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)
    d = extract_bag(path)
    with open(cp, "wb") as f:
        pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
    return d


def _load_worker(a):
    path, cache_dir, use_cache = a
    try:
        load_bag(path, cache_dir, use_cache)
        return path, None
    except Exception as exc:  # keep going on a corrupt bag
        return path, repr(exc)


def load_bags(paths, cache_dir=DEFAULT_CACHE, use_cache=True, jobs=6, verbose=True):
    """Extract (in parallel, into the cache) and return list of bag dicts."""
    todo = [p for p in paths if not (use_cache and os.path.exists(_cache_path(cache_dir, p)))]
    errors = {}
    if todo:
        os.makedirs(cache_dir, exist_ok=True)
        if jobs > 1 and len(todo) > 1:
            from multiprocessing import Pool
            with Pool(min(jobs, len(todo))) as pool:
                for i, (p, err) in enumerate(pool.imap_unordered(
                        _load_worker, [(p, cache_dir, use_cache) for p in todo])):
                    if verbose:
                        print("[extract %d/%d] %s %s" % (i + 1, len(todo), bag_label(p),
                                                         err or "ok"), file=sys.stderr)
                    if err:
                        errors[p] = err
        else:
            for p in todo:
                _, err = _load_worker((p, cache_dir, use_cache))
                if err:
                    errors[p] = err
    out = []
    for p in paths:
        if p in errors:
            continue
        out.append(load_bag(p, cache_dir, True))
    return out


# --------------------------------------------------------------------------
# Lanes (same CSV loader / projection semantics as include/lane_path.hpp)
# --------------------------------------------------------------------------
class Lane:
    def __init__(self, path):
        pts = []
        with open(path) as f:
            for line in f:
                c = line.strip().split(",")
                if len(c) < 3:
                    continue
                try:
                    pts.append((float(c[1]), float(c[2])))
                except ValueError:
                    continue
        self.path = path
        self.P = np.asarray(pts, float)
        seg = np.diff(self.P, axis=0)
        self.seg = seg
        self.seg_len = np.hypot(seg[:, 0], seg[:, 1])
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len)])
        self.length = float(self.cum_s[-1])
        from scipy.spatial import cKDTree
        self.tree = cKDTree(self.P)

    def project(self, x, y, before_start_m=0.0, k=3):
        """Vectorised nearest-segment projection. Returns s, d (left +), yaw, dist."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        q = np.stack([x, y], axis=1)
        _, idx = self.tree.query(q, k=k)
        idx = np.atleast_2d(idx)
        if idx.shape[0] != q.shape[0]:
            idx = idx.T
        nseg = len(self.seg)
        cand = np.concatenate([idx - 1, idx], axis=1)
        cand = np.clip(cand, 0, nseg - 1)
        best_d2 = np.full(len(q), np.inf)
        out_s = np.full(len(q), np.nan)
        out_d = np.full(len(q), np.nan)
        out_yaw = np.full(len(q), np.nan)
        for j in range(cand.shape[1]):
            i = cand[:, j]
            v = self.seg[i]
            l2 = np.maximum(np.einsum("ij,ij->i", v, v), 1e-12)
            r = q - self.P[i]
            raw_t = np.einsum("ij,ij->i", r, v) / l2
            lower = np.where(i == 0, -max(0.0, before_start_m) / np.sqrt(l2), 0.0)
            t = np.clip(raw_t, lower, 1.0)
            p = self.P[i] + t[:, None] * v
            e = q - p
            d2 = np.einsum("ij,ij->i", e, e)
            better = d2 < best_d2
            cross = v[:, 0] * e[:, 1] - v[:, 1] * e[:, 0]
            best_d2 = np.where(better, d2, best_d2)
            out_s = np.where(better, self.cum_s[i] + t * self.seg_len[i], out_s)
            out_d = np.where(better, np.where(cross >= 0, 1.0, -1.0) * np.sqrt(d2), out_d)
            out_yaw = np.where(better, np.arctan2(v[:, 1], v[:, 0]), out_yaw)
        return out_s, out_d, out_yaw, np.sqrt(best_d2)


def load_lanes(lane_dir=DEFAULT_LANE_DIR):
    lanes = {}
    for f in sorted(glob.glob(os.path.join(lane_dir, "*.csv"))):
        stem = os.path.basename(f)[:-4]
        try:
            lid = int(stem)
        except ValueError:
            continue
        lanes[lid] = Lane(f)
    return lanes


# --------------------------------------------------------------------------
# Derived per-bag quantities
# --------------------------------------------------------------------------
def utm_offset(d):
    """Constant MORAI-local -> UTM offset from /Ego_topic vs /odom (header stamps)."""
    e, o = d["ego"], d["odom"]
    if len(e["t"]) < 5 or len(o["t"]) < 5:
        return None
    t0, t1 = max(e["t"][0], o["t"][0]), min(e["t"][-1], o["t"][-1])
    m = (e["t"] >= t0) & (e["t"] <= t1)
    ox = np.interp(e["t"][m], o["t"], o["x"])
    oy = np.interp(e["t"][m], o["t"], o["y"])
    dx, dy = ox - e["x"][m], oy - e["y"][m]
    return {"dx": float(np.median(dx)), "dy": float(np.median(dy)),
            "spread_m": float(np.percentile(np.hypot(dx - np.median(dx), dy - np.median(dy)), 95)),
            "n": int(dx.size)}


def gt_tracks(d, off=None):
    """Per-uid GT tracks on header-stamp time axis. Velocity in m/s, positions UTM."""
    if off is None:
        off = utm_offset(d)
    g = d["gt"]
    if len(g["uid"]) == 0:
        return {}
    k = g["k"].astype(int)
    t = d["gt_msg_t"][k]
    tracks = {}
    uids = g["uid"].astype(int)
    order = np.lexsort((t, uids))
    uids_s = uids[order]
    bounds = np.flatnonzero(np.diff(uids_s)) + 1
    for grp in np.split(order, bounds):
        uid = int(uids[grp[0]])
        tt = t[grp]
        # de-duplicate identical stamps (keep last)
        keep = np.concatenate([np.diff(tt) > 1e-6, [True]])
        grp = grp[keep]
        tr = {
            "t": t[grp],
            "x": g["x"][grp] + (off["dx"] if off else 0.0),
            "y": g["y"][grp] + (off["dy"] if off else 0.0),
            "heading": g["heading"][grp],
            "v": g["vx_kmh"][grp] * KMH,
            "v_lat": g["vy_kmh"][grp] * KMH,
            "ax": g["ax"][grp],
            "length": np.maximum(g["sx"][grp], g["sy"][grp]),
            "type": int(g["type"][grp[0]]),
        }
        tracks[uid] = tr
    return tracks


def ego_track(d):
    """Ego on header-stamp axis, UTM from /odom (rear axle), speed from /Ego_topic."""
    o, e = d["odom"], d["ego"]
    v = np.interp(o["t"], e["t"], e["vx"]) if len(e["t"]) else o["v"]
    return {"t": o["t"], "x": o["x"], "y": o["y"], "yaw": o["yaw"], "v": v}


def step_series(t_query, t_src, v_src, default=np.nan):
    """Zero-order hold lookup (value of the last sample at or before t)."""
    t_query = np.asarray(t_query, float)
    if len(t_src) == 0:
        return np.full(t_query.shape, default)
    i = np.searchsorted(t_src, t_query, side="right") - 1
    out = np.asarray(v_src, float)[np.clip(i, 0, len(t_src) - 1)]
    return np.where(i >= 0, out, default)


def status_intervals(d):
    """Active intervals of /smpc/lane_change_status (header stamps).

    A new interval starts on an inactive->active edge, on a (src, tgt) change,
    or when elapsed_sec restarts.  t_end is the first inactive sample after the
    interval (or the last active one if the bag ends while active).
    """
    s = d["status"]
    out = []
    n = len(s["t"])
    if n == 0:
        return out
    act = s["active"] > 0.5
    i = 0
    while i < n:
        if not act[i]:
            i += 1
            continue
        j = i
        while (j + 1 < n and act[j + 1] and s["src"][j + 1] == s["src"][i]
               and s["tgt"][j + 1] == s["tgt"][i]
               and s["elapsed"][j + 1] + 0.05 >= s["elapsed"][j]):
            j += 1
        ended = j + 1 < n and not act[j + 1]
        t_end = s["t"][j + 1] if j + 1 < n else s["t"][j]
        out.append({"t_start": float(s["t"][i]), "t_end": float(t_end),
                    "t_last_active": float(s["t"][j]),
                    "src": int(s["src"][i]), "tgt": int(s["tgt"][i]),
                    "elapsed_first": float(s["elapsed"][i]),
                    "elapsed_last": float(s["elapsed"][j]),
                    "ended": bool(ended), "starts_at_bag_start": i == 0})
        i = j + 1
    return out


def sup_events(d, t0, t1, pattern):
    s = d["sup"]
    return [(float(t), txt) for t, txt in zip(s["t"], s["text"])
            if t0 <= t <= t1 and pattern in txt]


def parse_kv(text):
    out = {}
    for tok in text.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out
