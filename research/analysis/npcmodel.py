#!/usr/bin/env python3
"""GT NPC kinematics shared by M3 and M4.

  * true longitudinal acceleration: GT speed (km/h -> m/s, unique_id tracks) on
    a uniform 20 Hz grid after removing repeated MORAI frames (same pose and
    speed re-published with a new stamp), differentiated with a Savitzky-Golay
    filter (0.55 s window, quadratic).
  * lane assignment at 10 Hz on the lane CSVs (same 100 m before-start tangent
    extension that target_selector uses for nearby / target-lane vehicles).
  * acceleration / deceleration episodes and steady-cruise plateaus.
"""
import numpy as np
from scipy.signal import savgol_filter

import bagtools as bt

GRID_DT = 0.05
LANE_DT = 0.1
EXT_BEFORE_M = 100.0
LANE_ASSIGN_MAX_D = 1.9        # |d| to a lane centre to count as "on" it
EP_ON = 0.3                    # m/s^2 episode threshold
EP_FOOT = 0.1                  # onset = last sample before the episode with |a| <= this
EP_MIN_DUR = 0.8               # s
EP_MIN_DV = 0.8                # m/s
EP_MERGE_GAP = 0.3             # s
STEADY_A = 0.2                 # |a| below this = steady
STEADY_PRE = 1.0               # s of steady before an onset to call it "from cruise"
PLATEAU_A = 0.15
PLATEAU_MIN = 2.0
GAP_INVALID_S = 0.5            # GT stream gap that invalidates the derivative


def unique_frames(tr):
    """Mask dropping MORAI frames that repeat the previous pose and speed."""
    x, y, v = tr["x"], tr["y"], tr["v"]
    rep = np.zeros(len(v), bool)
    # A stopped NPC legitimately repeats its pose; only drop repeats while moving.
    rep[1:] = (x[1:] == x[:-1]) & (y[1:] == y[:-1]) & (v[1:] == v[:-1]) & (np.abs(v[1:]) > 0.05)
    return ~rep


def kinematics(tr):
    """Uniform-grid speed and true acceleration for one GT track."""
    m = unique_frames(tr)
    t, v = tr["t"][m], tr["v"][m]
    if len(t) < 8 or t[-1] - t[0] < 1.0:
        return None
    g = np.arange(t[0], t[-1], GRID_DT)
    vg = np.interp(g, t, v)
    win = 11
    if len(g) < win + 2:
        return None
    a = savgol_filter(vg, win, 2, deriv=1, delta=GRID_DT)
    # Grid samples inside (or within the filter half-window of) a gap in the
    # GT stream are invalid: an NPC leaving and re-entering the ~300 m
    # /Object_topic range keeps its unique_id, and interpolating across the
    # gap would fake an acceleration episode.
    valid = np.ones(len(g), bool)
    traw = tr["t"]
    for i in np.flatnonzero(np.diff(traw) > GAP_INVALID_S):
        valid &= ~((g > traw[i] - 0.3) & (g < traw[i + 1] + 0.3))
    return {"t": g, "v": vg, "a": a, "valid": valid,
            "x": np.interp(g, tr["t"], tr["x"]), "y": np.interp(g, tr["t"], tr["y"])}


def _runs(mask):
    """(start, end_exclusive) index pairs of True runs."""
    if not len(mask):
        return []
    dm = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(dm == 1), np.flatnonzero(dm == -1)))


def episodes(k, sign, t_track0, spawned):
    """Acceleration (sign=+1) or deceleration (sign=-1) episodes on kinematics k."""
    t, v, a = k["t"], k["v"], sign * k["a"]
    valid = (t >= t[0] + 0.5) & k["valid"]      # skip the spawn transient and stream gaps
    runs = _runs((a >= EP_ON) & valid)
    merged = []
    for s, e in runs:
        if (merged and t[s] - t[merged[-1][1] - 1] <= EP_MERGE_GAP and
                valid[merged[-1][1] - 1:s + 1].all()):
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    out = []
    for s, e in merged:
        dur = t[e - 1] - t[s]
        dv = sign * (v[e - 1] - v[s])
        if dur < EP_MIN_DUR or dv < EP_MIN_DV:
            continue
        f = s
        while f > 0 and valid[f - 1] and a[f - 1] > EP_FOOT and t[s] - t[f - 1] <= 1.5:
            f -= 1
        pre = (t >= t[f] - STEADY_PRE) & (t < t[f])
        from_steady = bool(t[f] - t[0] >= STEADY_PRE and pre.any() and
                           np.all(valid[pre]) and np.all(np.abs(k["a"][pre]) < STEADY_A))
        seg = slice(f, e)
        out.append({
            "sign": sign, "t_onset": float(t[f]), "t_on03": float(t[s]),
            "t_end": float(t[e - 1]), "duration": float(t[e - 1] - t[f]),
            "peak": float(np.max(a[s:e])), "mean": float(np.mean(a[seg])),
            "dv": float(sign * (v[e - 1] - v[f])), "v_start": float(v[f]),
            "v_end": float(v[e - 1]),
            "t_since_track_start": float(t[f] - t_track0),
            "spawn_episode": bool(spawned and t[f] - t_track0 <= 1.5),
            "from_steady": from_steady,
        })
    return out


def plateaus(k):
    t, v, a = k["t"], k["v"], k["a"]
    out = []
    for s, e in _runs((np.abs(a) < PLATEAU_A) & k["valid"]):
        if t[e - 1] - t[s] >= PLATEAU_MIN:
            out.append({"t0": float(t[s]), "t1": float(t[e - 1]),
                        "v_mean": float(np.mean(v[s:e]))})
    return out


def lane_assign(lanes, x, y):
    """Per-sample assigned lane (-1 = none), d on it, and s/d on every lane."""
    n = len(x)
    best = np.full(n, np.inf)
    lane = np.full(n, -1)
    per = {}
    for lid, l in lanes.items():
        s, d, _, _ = l.project(x, y, EXT_BEFORE_M)
        interior = (s >= -EXT_BEFORE_M + 0.5) & (s <= l.length - 0.5)
        per[lid] = (s, d, interior)
        ok = interior & (np.abs(d) < best)
        best = np.where(ok, np.abs(d), best)
        lane = np.where(ok, lid, lane)
    lane = np.where(best <= LANE_ASSIGN_MAX_D, lane, -1)
    return lane, per


def lane_changes(tl, lane, per, v_lat_fn=None, persist=1.0):
    """Sustained lane-assignment changes between two lanes that both exist there."""
    out = []
    runs = []
    for s, e in _runs(lane >= 0):
        seg = lane[s:e]
        i = s
        for j in range(s + 1, e + 1):
            if j == e or lane[j] != lane[i]:
                runs.append((i, j, int(lane[i])))
                i = j
    runs = [r for r in runs if tl[r[1] - 1] - tl[r[0]] >= persist]
    for (s0, e0, l0), (s1, e1, l1) in zip(runs[:-1], runs[1:]):
        if l0 == l1 or abs(l0 - l1) != 1 or tl[s1] - tl[e0 - 1] > 2.0:
            continue
        k = s1
        both = per[l0][2][k] and per[l1][2][k]
        d0 = per[l0][1]
        s_from = float(per[l0][0][k])
        # lateral shift on the from-lane over +-3 s (or the available runs)
        ka = max(s0, k - int(3.0 / LANE_DT))
        kb = min(e1 - 1, k + int(3.0 / LANE_DT))
        out.append({"t": float(tl[k]), "from": l0, "to": l1,
                    "both_lanes_exist": bool(both),
                    "d_from_before": float(d0[ka]), "d_from_after": float(d0[kb]),
                    "d_to_after": float(per[l1][1][kb]),
                    "lateral_rate_mps": float(abs(d0[kb] - d0[ka]) / max(tl[kb] - tl[ka], 1e-3)),
                    "s_from": s_from,
                    "v_lat_peak": (float(v_lat_fn(tl[k])) if v_lat_fn else None)})
    return out
