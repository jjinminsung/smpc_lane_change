#!/usr/bin/env python3
"""M4: rear-accel mode detection delay of the baseline observed-accel filter.

Baseline filter (src/smpc_decision_node.cpp updateOneVehicleHistory):
  per unique_id, once per /smpc/targets message (header.stamp):
    first sight / gap > 0.35 s  -> v_prev = v_long, filtered = 0
    1e-3 < dt < 1.0             -> raw = clamp((v_long - v_prev)/dt, +-6)
                                   filtered = 0.35*raw + 0.65*filtered
    always                      -> v_prev = v_long, stamp = header.stamp
  Rear-accel probability table breaks [-0.5, 0.5, 1.0, 1.5] m/s^2.

Part A  /smpc/targets (LiDAR mode, as recorded) and /Object_topic rates.
Part B  object_topic mode: target_selector publishes one /smpc/targets per
        /Object_topic message with v_long = npc.velocity.x / 3.6 and the
        Object_topic header stamp.  The filter is simulated on the raw GT stream
        (repeated MORAI frames and stamp jitter included).  Delay from the true
        onset (M3 episode foot) to filtered > 0.5 and > 1.0 m/s^2, binned by the
        episode's true mean acceleration.  Reference: same GT stream decimated
        to the recorded LiDAR /smpc/targets stamps (~9.6 Hz).
Part C  tracker (/smpc/targets recorded in LiDAR mode): vehicles matched to GT
        by position (<= 3 m), filter run per tracker id exactly as the node;
        delay for GT onsets, false-positive crossings in GT steady cruise.
Part D  probability-table check: P("5 s ahead of constant-velocity by >= 12.5 m")
        per filtered-accel bin, GT event, for target-rear samples.

  python3 research/analysis/m4_mode_detection_delay.py --bags ~/bags/lc_gt_2026-09-28-*.bag
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagtools as bt  # noqa: E402
import npcmodel as nm  # noqa: E402

ALPHA = 0.35
CLAMP = 6.0
GAP_RESET = 0.35
THRESH = (0.5, 1.0)
BREAKS = [-0.5, 0.5, 1.0, 1.5]
ACC_BINS = [(0.3, 0.75), (0.75, 1.25), (1.25, 2.0), (2.0, 10.0)]
MATCH_M = 3.0
STEADY_WIN = 1.0
EVENT_H = 5.0
EVENT_M = 12.5


def filter_sim(t, v):
    """Exact baseline filter on one vehicle's message stream."""
    n = len(t)
    f = np.zeros(n)
    if n == 0:
        return f
    vp, tp, last, fv = v[0], t[0], t[0], 0.0
    for i in range(1, n):
        if t[i] - last > GAP_RESET:
            vp, tp, fv = v[i], t[i], 0.0
            f[i] = 0.0
            last = t[i]
            continue
        dt = t[i] - tp
        if 1e-3 < dt < 1.0:
            raw = min(max((v[i] - vp) / dt, -CLAMP), CLAMP)
            fv = ALPHA * raw + (1.0 - ALPHA) * fv
        vp, tp, last = v[i], t[i], t[i]
        f[i] = fv
    return f


def gt_raw_streams(d):
    """uid -> (stamps, v_long m/s) exactly as /Object_topic delivers them."""
    g = d["gt"]
    out = {}
    if len(g["uid"]) == 0:
        return out
    k = g["k"].astype(int)
    t = d["gt_msg_t"][k]
    uid = g["uid"].astype(int)
    order = np.lexsort((k, uid))
    us = uid[order]
    for grp in np.split(order, np.flatnonzero(np.diff(us)) + 1):
        out[int(uid[grp[0]])] = (t[grp], g["vx_kmh"][grp] * bt.KMH)
    return out


def steady_mask(k, tq):
    """|a_true| < 0.2 over [t-1, t+1] and moving (v > 3 m/s)."""
    a, tg = k["a"], k["t"]
    absa = np.abs(a)
    # rolling max over +-STEADY_WIN on the 20 Hz grid
    w = int(round(STEADY_WIN / nm.GRID_DT))
    pad = np.concatenate([np.full(w, np.inf), absa, np.full(w, np.inf)])
    from numpy.lib.stride_tricks import sliding_window_view
    rmax = sliding_window_view(pad, 2 * w + 1).max(axis=1)
    i = np.clip(np.searchsorted(tg, tq), 0, len(tg) - 1)
    inside = (tq >= tg[0]) & (tq <= tg[-1])
    return inside & k["valid"][i] & (rmax[i] < nm.STEADY_A) & (k["v"][i] > 3.0)


def first_cross(t, f, t0, thr, t_max, sustain=1):
    """Delay from t0 to the first sample with f > thr (held for `sustain` updates)."""
    m = (t >= t0) & (t <= t_max) & (f > thr)
    if sustain > 1 and len(m) >= sustain:
        from numpy.lib.stride_tricks import sliding_window_view
        held = np.zeros(len(m), bool)
        held[:len(m) - sustain + 1] = sliding_window_view(m, sustain).all(axis=1)
        m = held
    i = np.flatnonzero(m)
    return float(t[i[0]] - t0) if len(i) else None


SUSTAIN_N = 3          # "sustained" crossing = 3 consecutive filter updates above the threshold
TAU = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def f_at(t, f, tq):
    """Filter value held at query times (last update at or before tq; nan if none within 0.35 s)."""
    i = np.searchsorted(t, tq, side="right") - 1
    ok = (i >= 0) & (np.abs(t[np.clip(i, 0, len(t) - 1)] - tq) <= GAP_RESET)
    return np.where(ok, f[np.clip(i, 0, len(t) - 1)], np.nan)


def delay_table(rows, key):
    out = {}
    for lo, hi in ACC_BINS + [(0.0, 99.0)]:
        sel = [r for r in rows if lo <= r["true_mean_accel"] < hi]
        name = "all" if hi == 99.0 else "%.2f-%.2f" % (lo, hi)
        ent = {"n": len(sel)}
        for thr in THRESH:
            dl = [r["%s_delay_%.1f" % (key, thr)] for r in sel]
            got = [x for x in dl if x is not None]
            ent["delay_%.1f" % thr] = bt.stats(got)
            ent["frac_detected_%.1f" % thr] = (len(got) / len(sel)) if sel else None
            lag = [r["%s_lag_%.1f" % (key, thr)] for r in sel if r.get("%s_lag_%.1f" % (key, thr)) is not None]
            ent["lag_vs_true_crossing_%.1f" % thr] = bt.stats(lag)
        out[name] = ent
    return out


def bin_index(f):
    return np.searchsorted(np.asarray(BREAKS), f, side="right")


def main():
    ap = bt.base_argparser(__doc__)
    args = ap.parse_args()
    paths = bt.expand_bags(args.bags)
    lanes = bt.load_lanes(args.lanes)
    ds = bt.load_bags(paths, args.cache, not args.no_cache, args.jobs)

    rate_rows = []
    gt_ep_rows, trk_ep_rows = [], []
    fp = {"gt_obj": {"n": 0, ">0.5": 0, ">1.0": 0, "<-0.5": 0, "edges0.5": 0, "edges1.0": 0, "sec": 0.0},
          "gt_lidar_rate": {"n": 0, ">0.5": 0, ">1.0": 0, "<-0.5": 0, "edges0.5": 0, "edges1.0": 0, "sec": 0.0},
          "trk_all": {"n": 0, ">0.5": 0, ">1.0": 0, "<-0.5": 0, "edges0.5": 0, "edges1.0": 0, "sec": 0.0},
          "trk_target_rear": {"n": 0, ">0.5": 0, ">1.0": 0, "<-0.5": 0, "edges0.5": 0, "edges1.0": 0, "sec": 0.0},
          "gt_obj_at_trk_target_rear": {"n": 0, ">0.5": 0, ">1.0": 0, "<-0.5": 0, "edges0.5": 0, "edges1.0": 0, "sec": 0.0}}
    match_dist, vlong_err, vlong_err_rear = [], [], []
    n_gap_excluded = [0]
    table = {"trk_target_rear": {}, "gt_obj_target_rear": {}}
    trk_bins_occ = {"trk_target_rear": np.zeros(5), "gt_obj_target_rear": np.zeros(5)}

    def add_fp(key, fvals, dur):
        f = fp[key]
        f["n"] += len(fvals)
        f[">0.5"] += int(np.sum(fvals > 0.5))
        f[">1.0"] += int(np.sum(fvals > 1.0))
        f["<-0.5"] += int(np.sum(fvals < -0.5))
        f["sec"] += dur

    for d in ds:
        # ---------------- Part A: rates
        tg = d["targets"]
        ot = d["gt_msg_t"]
        rr = {"bag": d["label"]}
        if len(tg["t"]) > 5:
            dt = np.diff(tg["t"])
            rr.update({"targets_n": int(len(tg["t"])), "targets_hz": float(1.0 / np.median(dt)),
                       "targets_dt": bt.stats(dt), "targets_frac_dt_gt_0.35": float(np.mean(dt > GAP_RESET))})
        if len(ot) > 5:
            dt = np.diff(ot)
            rr.update({"object_hz": float(len(ot) / (ot[-1] - ot[0])),
                       "object_dt_median": float(np.median(dt)),
                       "object_dt_p10": float(np.percentile(dt, 10)),
                       "object_dt_p90": float(np.percentile(dt, 90)),
                       "object_frac_dt_le_1ms": float(np.mean(dt <= 1e-3))})
        if len(ot) < 50:
            rate_rows.append(rr)
            continue
        # fraction of moving-NPC frames that repeat the previous state (pose and speed)
        rep_n = mov_n = 0
        for tr_ in bt.gt_tracks(d).values():
            mv = np.abs(tr_["v"][1:]) > 0.05
            rep = (tr_["x"][1:] == tr_["x"][:-1]) & (tr_["y"][1:] == tr_["y"][:-1]) & (tr_["v"][1:] == tr_["v"][:-1])
            rep_n += int(np.sum(rep & mv))
            mov_n += int(np.sum(mv))
        rr["gt_repeated_frame_frac_moving"] = rep_n / max(mov_n, 1)
        rate_rows.append(rr)

        tracks = bt.gt_tracks(d)
        raw = gt_raw_streams(d)
        kin, eps_by_uid = {}, {}
        T0 = float(ot[0])
        eg = bt.ego_track(d)
        for uid, tr in tracks.items():
            k = nm.kinematics(tr)
            if k is None:
                continue
            tl = np.arange(k["t"][0], k["t"][-1], nm.LANE_DT)
            lane, _ = nm.lane_assign(lanes, np.interp(tl, k["t"], k["x"]), np.interp(tl, k["t"], k["y"]))
            if np.sum(lane >= 0) * nm.LANE_DT < 2.0:
                continue
            kin[uid] = k
            ex0 = np.interp(tr["t"][0], eg["t"], eg["x"])
            ey0 = np.interp(tr["t"][0], eg["t"], eg["y"])
            spawned = bool(tr["t"][0] > T0 + 0.5 and np.hypot(tr["x"][0] - ex0, tr["y"][0] - ey0) < 290.0)
            eps_by_uid[uid] = [e for e in nm.episodes(k, +1, tr["t"][0], spawned) if not e["spawn_episode"]]

        # ---------------- Part B: GT filter at object_topic rate (+ decimated reference)
        lid_t = tg["t"] if len(tg["t"]) > 5 else None
        gt_filt = {}
        for uid, k in kin.items():
            t_raw, v_raw = raw[uid]
            f_obj = filter_sim(t_raw, v_raw)
            gt_filt[uid] = (t_raw, f_obj)
            f_lid = t_lid = None
            if lid_t is not None:
                sel = (lid_t >= t_raw[0]) & (lid_t <= t_raw[-1])
                t_lid = lid_t[sel]
                if len(t_lid) > 3:
                    # latest GT message at or before each LiDAR stamp
                    j = np.clip(np.searchsorted(t_raw, t_lid, side="right") - 1, 0, len(t_raw) - 1)
                    f_lid = filter_sim(t_lid, v_raw[j])
            # false positives in steady cruise
            st = steady_mask(k, t_raw)
            add_fp("gt_obj", f_obj[st], float(np.sum(np.diff(t_raw)[st[1:]])) if len(t_raw) > 1 else 0.0)
            for thr in THRESH:
                m = (f_obj > thr) & st
                fp["gt_obj"]["edges%.1f" % thr] += int(np.sum(m[1:] & ~m[:-1]))
            if f_lid is not None:
                stl = steady_mask(k, t_lid)
                add_fp("gt_lidar_rate", f_lid[stl], float(np.sum(np.diff(t_lid)[stl[1:]])))
                for thr in THRESH:
                    m = (f_lid > thr) & stl
                    fp["gt_lidar_rate"]["edges%.1f" % thr] += int(np.sum(m[1:] & ~m[:-1]))
            for e in eps_by_uid.get(uid, []):
                t_on = e["t_onset"]
                if t_on - t_raw[0] < 1.0:
                    continue
                horizon = e["t_end"] + 1.0
                wr = (t_raw >= t_on - 1.0) & (t_raw <= horizon)
                if len(t_raw[wr]) < 2 or np.max(np.diff(t_raw[wr])) > GAP_RESET:
                    n_gap_excluded[0] += 1        # filter reset inside the window
                    continue
                row = {"bag": d["label"], "uid": uid, "true_mean_accel": e["mean"],
                       "true_peak_accel": e["peak"], "duration": e["duration"],
                       "from_steady": e["from_steady"], "v_start": e["v_start"]}
                row["f_tau"] = f_at(t_raw, f_obj, t_on + np.asarray(TAU)).tolist()
                row["f_tau_lid"] = f_at(t_lid, f_lid, t_on + np.asarray(TAU)).tolist() if f_lid is not None else None
                for thr in THRESH:
                    row["objsus_delay_%.1f" % thr] = first_cross(t_raw, f_obj, t_on, thr, horizon, SUSTAIN_N)
                    row["objsus_lag_%.1f" % thr] = None
                    row["obj_delay_%.1f" % thr] = first_cross(t_raw, f_obj, t_on, thr, horizon)
                    tc = first_cross(k["t"], k["a"], t_on, thr, horizon)
                    row["true_cross_%.1f" % thr] = tc
                    row["obj_lag_%.1f" % thr] = (row["obj_delay_%.1f" % thr] - tc
                                                 if (tc is not None and row["obj_delay_%.1f" % thr] is not None) else None)
                    if f_lid is not None:
                        row["lid_delay_%.1f" % thr] = first_cross(t_lid, f_lid, t_on, thr, horizon)
                        row["lid_lag_%.1f" % thr] = (row["lid_delay_%.1f" % thr] - tc
                                                     if (tc is not None and row["lid_delay_%.1f" % thr] is not None) else None)
                    else:
                        row["lid_delay_%.1f" % thr] = None
                        row["lid_lag_%.1f" % thr] = None
                gt_ep_rows.append(row)

        # ---------------- Part C: tracker vehicles in recorded /smpc/targets
        if lid_t is None:
            continue
        nb = tg["nearby"]
        reps = {r: tg[r] for r in ("current_front", "target_front", "target_rear")}
        # per message vehicle list (dedupe by uid within a message)
        per_msg = [dict() for _ in range(len(lid_t))]
        for r, a in reps.items():
            for i in np.flatnonzero(a["valid"] > 0.5):
                u = int(a["uid"][i])
                if u < 0:
                    continue
                ent = per_msg[i].setdefault(u, {"x": a["x"][i], "y": a["y"][i], "v": a["v_long"][i], "roles": set()})
                ent["roles"].add(r)
        for j in range(len(nb["k"])):
            if nb["valid"][j] < 0.5:
                continue
            i, u = int(nb["k"][j]), int(nb["uid"][j])
            if u < 0:
                continue
            per_msg[i].setdefault(u, {"x": nb["x"][j], "y": nb["y"][j], "v": nb["v_long"][j], "roles": set()})
        # GT positions at each targets stamp
        uids = list(kin.keys())
        if not uids:
            continue
        GX = np.full((len(uids), len(lid_t)), np.nan)
        GY = np.full_like(GX, np.nan)
        for a_i, u in enumerate(uids):
            tr = tracks[u]
            cov = (lid_t >= tr["t"][0]) & (lid_t <= tr["t"][-1])
            GX[a_i, cov] = np.interp(lid_t[cov], tr["t"], tr["x"])
            GY[a_i, cov] = np.interp(lid_t[cov], tr["t"], tr["y"])
        # per tracker id streams
        streams = {}
        for i, vehs in enumerate(per_msg):
            for u, ent in vehs.items():
                dd = np.hypot(GX[:, i] - ent["x"], GY[:, i] - ent["y"])
                g = -1
                if np.any(np.isfinite(dd)):
                    gi = int(np.nanargmin(dd))
                    if dd[gi] <= MATCH_M:
                        g = uids[gi]
                        match_dist.append(float(dd[gi]))
                streams.setdefault(u, []).append((lid_t[i], ent["v"], g, "target_rear" in ent["roles"]))
        # filter per tracker id, collect matched samples by GT uid
        by_gt = {}
        for u, rows in streams.items():
            t = np.array([r[0] for r in rows])
            v = np.array([r[1] for r in rows])
            g = np.array([r[2] for r in rows])
            rear = np.array([r[3] for r in rows])
            f = filter_sim(t, v)
            for gu in set(g[g >= 0]):
                m = g == gu
                by_gt.setdefault(int(gu), []).append((t[m], f[m], v[m], rear[m]))
        for gu, chunks in by_gt.items():
            t = np.concatenate([c[0] for c in chunks])
            f = np.concatenate([c[1] for c in chunks])
            v = np.concatenate([c[2] for c in chunks])
            rear = np.concatenate([c[3] for c in chunks])
            o = np.argsort(t, kind="stable")
            t, f, v, rear = t[o], f[o], v[o], rear[o]
            k = kin[gu]
            vg = np.interp(t, k["t"], k["v"])
            vlong_err.extend((v - vg).tolist())
            vlong_err_rear.extend((v[rear] - vg[rear]).tolist())
            st = steady_mask(k, t)
            dts = np.diff(t, prepend=t[0])
            dts = np.where(dts > 0.5, 0.1, dts)
            add_fp("trk_all", f[st], float(np.sum(dts[st])))
            add_fp("trk_target_rear", f[st & rear], float(np.sum(dts[st & rear])))
            t_raw, f_obj = gt_filt[gu]
            fo = f_obj[np.clip(np.searchsorted(t_raw, t, side="right") - 1, 0, len(t_raw) - 1)]
            add_fp("gt_obj_at_trk_target_rear", fo[st & rear], float(np.sum(dts[st & rear])))
            for thr in THRESH:
                for key, ff, mm in (("trk_all", f, st), ("trk_target_rear", f, st & rear),
                                    ("gt_obj_at_trk_target_rear", fo, st & rear)):
                    m = (ff > thr) & mm
                    fp[key]["edges%.1f" % thr] += int(np.sum(m[1:] & ~m[:-1]))
            # Part D: probability table event on target-rear samples (GT event)
            if rear.any():
                tr_t = t[rear]
                ok = tr_t + EVENT_H <= k["t"][-1]
                tq = tr_t[ok]
                if len(tq):
                    cum = np.concatenate([[0.0], np.cumsum(k["v"][1:] * nm.GRID_DT)])
                    s0 = np.interp(tq, k["t"], cum)
                    s5 = np.interp(tq + EVENT_H, k["t"], cum)
                    v0 = np.interp(tq, k["t"], k["v"])
                    ev = (s5 - s0 - v0 * EVENT_H) >= EVENT_M
                    for key, ff in (("trk_target_rear", f[rear][ok]), ("gt_obj_target_rear", fo[rear][ok])):
                        b = bin_index(ff)
                        for bi in range(5):
                            mb = b == bi
                            ent = table[key].setdefault(bi, [0, 0])
                            ent[0] += int(mb.sum())
                            ent[1] += int(ev[mb].sum())
                        trk_bins_occ[key] += np.bincount(b, minlength=5)
            # delay on GT onsets for the tracker
            for e in eps_by_uid.get(gu, []):
                t_on = e["t_onset"]
                w = (t >= t_on - 1.0) & (t <= t_on + 3.0)
                if w.sum() < 20:          # need >= ~2 s of matched samples around onset
                    continue
                horizon = e["t_end"] + 1.0
                row = {"bag": d["label"], "gt_uid": gu, "true_mean_accel": e["mean"],
                       "true_peak_accel": e["peak"], "from_steady": e["from_steady"],
                       "coverage_s": float(w.sum() * 0.104),
                       "was_target_rear_near_onset": bool(rear[w].any())}
                row["f_tau"] = f_at(t, f, t_on + np.asarray(TAU)).tolist()
                pre = (t >= t_on - 1.0) & (t < t_on)
                for thr in THRESH:
                    row["trk_above_%.1f_in_1s_before_onset" % thr] = bool(np.any(f[pre] > thr))
                    row["trksus_delay_%.1f" % thr] = first_cross(t, f, t_on, thr, horizon, SUSTAIN_N)
                    row["trksus_lag_%.1f" % thr] = None
                    row["trk_delay_%.1f" % thr] = first_cross(t, f, t_on, thr, horizon)
                    tc = first_cross(k["t"], k["a"], t_on, thr, horizon)
                    row["trk_lag_%.1f" % thr] = (row["trk_delay_%.1f" % thr] - tc
                                                 if (tc is not None and row["trk_delay_%.1f" % thr] is not None) else None)
                trk_ep_rows.append(row)

    def fp_out(f):
        n = max(f["n"], 1)
        mins = max(f["sec"] / 60.0, 1e-9)
        return {"n_samples": f["n"], "steady_minutes": f["sec"] / 60.0,
                "frac_gt_0.5": f[">0.5"] / n, "frac_gt_1.0": f[">1.0"] / n,
                "frac_lt_-0.5": f["<-0.5"] / n,
                "crossings_per_min_0.5": f["edges0.5"] / mins,
                "crossings_per_min_1.0": f["edges1.0"] / mins}

    obj_hz = [r["object_hz"] for r in rate_rows if "object_hz" in r]

    def curve(rows, key="f_tau"):
        out = {"tau_s": TAU}
        A = np.array([r[key] for r in rows if r.get(key) is not None], float)
        if A.size == 0:
            return out
        for thr in THRESH:
            with np.errstate(invalid="ignore"):
                out["P_above_%.1f" % thr] = [float(np.nanmean(A[:, j] > thr)) for j in range(len(TAU))]
        out["median_filtered"] = [float(np.nanmedian(A[:, j])) for j in range(len(TAU))]
        out["n"] = int(A.shape[0])
        return out
    res = {
        "filter": {"alpha": ALPHA, "clamp": CLAMP, "gap_reset_s": GAP_RESET, "breaks": BREAKS},
        "rates": {
            "targets_hz_lidar_mode_per_bag": bt.stats([r["targets_hz"] for r in rate_rows if "targets_hz" in r]),
            "targets_dt_median_all": bt.stats([r["targets_dt"]["median"] for r in rate_rows if "targets_dt" in r]),
            "targets_frac_dt_gt_0.35": bt.stats([r["targets_frac_dt_gt_0.35"] for r in rate_rows if "targets_frac_dt_gt_0.35" in r]),
            "object_topic_hz_per_bag": bt.stats(obj_hz),
            "object_dt_median": bt.stats([r["object_dt_median"] for r in rate_rows if "object_dt_median" in r]),
            "object_dt_p10": bt.stats([r["object_dt_p10"] for r in rate_rows if "object_dt_p10" in r]),
            "object_dt_p90": bt.stats([r["object_dt_p90"] for r in rate_rows if "object_dt_p90" in r]),
            "object_frac_dt_le_1ms": bt.stats([r["object_frac_dt_le_1ms"] for r in rate_rows if "object_frac_dt_le_1ms" in r]),
            "gt_repeated_frame_frac_moving": bt.stats([r["gt_repeated_frame_frac_moving"] for r in rate_rows if "gt_repeated_frame_frac_moving" in r]),
            "per_bag": rate_rows,
        },
        "object_topic_mode_note": (
            "target_selector objectCallback publishes once per /Object_topic message "
            "(queue 1), v_long = npc.velocity.x/3.6, header = Object_topic header; so the "
            "filter would update at the Object_topic rate. NOTE: Object_topic positions "
            "are MORAI-local, /odom and lane CSVs are UTM (offset ~(302595.0, 4124145.0) m); "
            "baseline objectCallback applies no offset."),
        "gt_object_rate_delay_all_nonspawn": delay_table(gt_ep_rows, "obj"),
        "gt_object_rate_delay_from_steady": delay_table([r for r in gt_ep_rows if r["from_steady"]], "obj"),
        "gt_lidar_rate_delay_all_nonspawn": delay_table(gt_ep_rows, "lid"),
        "gt_lidar_rate_delay_from_steady": delay_table([r for r in gt_ep_rows if r["from_steady"]], "lid"),
        "gt_object_rate_delay_sustained3_from_steady": delay_table([r for r in gt_ep_rows if r["from_steady"]], "objsus"),
        "tracker_delay_all": delay_table(trk_ep_rows, "trk"),
        "tracker_delay_sustained3_all": delay_table(trk_ep_rows, "trksus"),
        "tracker_frac_episodes_already_above_0.5_before_onset": (
            sum(1 for r in trk_ep_rows if r["trk_above_0.5_in_1s_before_onset"]) / len(trk_ep_rows)) if trk_ep_rows else None,
        "tracker_frac_episodes_already_above_1.0_before_onset": (
            sum(1 for r in trk_ep_rows if r["trk_above_1.0_in_1s_before_onset"]) / len(trk_ep_rows)) if trk_ep_rows else None,
        "tracker_delay_target_rear": delay_table([r for r in trk_ep_rows if r["was_target_rear_near_onset"]], "trk"),
        "detection_curve": {
            "gt_object_rate_from_steady": curve([r for r in gt_ep_rows if r["from_steady"]]),
            "gt_lidar_rate_from_steady": curve([r for r in gt_ep_rows if r["from_steady"]], "f_tau_lid"),
            "tracker_all": curve(trk_ep_rows),
            "tracker_from_steady": curve([r for r in trk_ep_rows if r["from_steady"]]),
        },
        "false_positive_steady_cruise": {k: fp_out(v) for k, v in fp.items()},
        "tracker_match": {"match_dist_m": bt.stats(match_dist),
                          "v_long_err_mps_all": bt.stats(vlong_err),
                          "v_long_err_mps_target_rear": bt.stats(vlong_err_rear),
                          "v_long_abs_err_p50_p90_all": [float(np.percentile(np.abs(vlong_err), 50)) if vlong_err else None,
                                                         float(np.percentile(np.abs(vlong_err), 90)) if vlong_err else None]},
        "probability_table_gt_event": {
            key: [{"bin": ["<-0.5", "-0.5..0.5", "0.5..1.0", "1.0..1.5", ">=1.5"][bi],
                   "n": table[key].get(bi, [0, 0])[0],
                   "event_rate": (table[key][bi][1] / table[key][bi][0]) if table[key].get(bi, [0, 0])[0] else None}
                  for bi in range(5)] for key in table},
        "n_gt_episodes": len(gt_ep_rows),
        "n_gt_episodes_excluded_stream_gap": n_gap_excluded[0],
        "n_tracker_episodes": len(trk_ep_rows),
        "gt_episode_rows": gt_ep_rows,
        "tracker_episode_rows": trk_ep_rows,
    }
    out = os.path.join(args.out, "day1_m4.json")
    bt.write_json(out, res)

    R = res["rates"]
    print("targets (LiDAR) Hz per bag:", bt.fmt_stats(R["targets_hz_lidar_mode_per_bag"]))
    print("targets frac dt>0.35:", bt.fmt_stats(R["targets_frac_dt_gt_0.35"], 3))
    print("Object_topic Hz per bag:", bt.fmt_stats(R["object_topic_hz_per_bag"]),
          " dt med", bt.fmt_stats(R["object_dt_median"], 3), " frac dt<=1ms", bt.fmt_stats(R["object_frac_dt_le_1ms"], 3))
    for name in ("gt_object_rate_delay_all_nonspawn", "gt_object_rate_delay_from_steady",
                 "gt_object_rate_delay_sustained3_from_steady",
                 "gt_lidar_rate_delay_from_steady", "tracker_delay_all", "tracker_delay_sustained3_all",
                 "tracker_delay_target_rear"):
        print("==", name)
        for b, ent in res[name].items():
            print("  %-10s n=%-4d d0.5 %s det=%.2f | d1.0 %s det=%s | lag0.5 med %s" % (
                b, ent["n"], bt.fmt_stats(ent["delay_0.5"]), ent["frac_detected_0.5"] or 0,
                bt.fmt_stats(ent["delay_1.0"]),
                "%.2f" % ent["frac_detected_1.0"] if ent["frac_detected_1.0"] is not None else "-",
                "%.2f" % ent["lag_vs_true_crossing_0.5"]["median"] if ent["lag_vs_true_crossing_0.5"].get("n") else "-"))
    print("== false positives in steady cruise")
    for k, v in res["false_positive_steady_cruise"].items():
        print("  %-28s n=%-7d min=%.1f >0.5 %.3f >1.0 %.3f <-0.5 %.3f  cross/min 0.5: %.2f 1.0: %.2f" % (
            k, v["n_samples"], v["steady_minutes"], v["frac_gt_0.5"], v["frac_gt_1.0"], v["frac_lt_-0.5"],
            v["crossings_per_min_0.5"], v["crossings_per_min_1.0"]))
    print("tracker episodes already above 0.5 / 1.0 in the 1 s before onset: %s / %s" % (
        res["tracker_frac_episodes_already_above_0.5_before_onset"], res["tracker_frac_episodes_already_above_1.0_before_onset"]))
    for k, c in res["detection_curve"].items():
        if c.get("n"):
            print("curve %-28s n=%-4d P(f>0.5) at tau %s: %s" % (k, c["n"], TAU, [round(x, 2) for x in c["P_above_0.5"]]))
            print("     %-28s        P(f>1.0)            : %s" % ("", [round(x, 2) for x in c["P_above_1.0"]]))
    print("tracker match dist:", bt.fmt_stats(res["tracker_match"]["match_dist_m"]))
    print("tracker v_long err all:", bt.fmt_stats(res["tracker_match"]["v_long_err_mps_all"]),
          "| target_rear:", bt.fmt_stats(res["tracker_match"]["v_long_err_mps_target_rear"]))
    for key, rows in res["probability_table_gt_event"].items():
        print("table", key, [(r["bin"], r["n"], None if r["event_rate"] is None else round(r["event_rate"], 3)) for r in rows])
    print("wrote", out)


if __name__ == "__main__":
    main()
