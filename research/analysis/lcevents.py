#!/usr/bin/env python3
"""Lane-change event extraction shared by M1/M2/M3.

One event per active interval of /smpc/lane_change_status (supervisor
accepted CHANGE_LEFT and published /path_number).  The ego lateral state is
/odom (IONIQ 5 rear axle, the same point target_selector and the supervisor
use) projected onto the source / target lane CSVs.  Target lane projections use
the same 100 m before-start tangent extension as target_selector.
"""
import numpy as np

import bagtools as bt

EXT_BEFORE_M = 100.0
CENTER_TOL_M = 0.45
SUP_COMPLETE_D_M = 0.75   # supervisor.yaml completion_lateral_d_m


def _first(tt, mask):
    i = np.flatnonzero(mask)
    return float(tt[i[0]]) if len(i) else None


def _first_settled(tt, mask, hold_s):
    """First time mask becomes true and stays true for hold_s."""
    idx = np.flatnonzero(mask)
    for i in idx:
        j = np.searchsorted(tt, tt[i] + hold_s, side="right")
        if j > len(tt):
            break
        if np.all(mask[i:j]) and tt[min(j, len(tt)) - 1] >= tt[i] + hold_s - 0.06:
            return float(tt[i])
    return None


def lc_events(d, lanes):
    eg = bt.ego_track(d)
    if len(eg["t"]) < 10:
        return []
    t_bag0, t_bag1 = float(eg["t"][0]), float(eg["t"][-1])
    tg = d["targets"]
    pv = d["preview"]
    ev = []
    ivs = bt.status_intervals(d)
    for k_iv, iv in enumerate(ivs):
        next_t0 = ivs[k_iv + 1]["t_start"] if k_iv + 1 < len(ivs) else np.inf
        src, tgt = iv["src"], iv["tgt"]
        if src not in lanes or tgt not in lanes:
            continue
        t0, t_end = iv["t_start"], iv["t_end"]
        partial_start = iv["elapsed_first"] > 0.1 or iv["starts_at_bag_start"]
        w0, w1 = t0 - 1.0, min(t_bag1, max(t_end, t0) + 12.0)
        m = (eg["t"] >= w0) & (eg["t"] <= w1)
        tt = eg["t"][m] - t0
        x, y = eg["x"][m], eg["y"][m]
        _, d_src, _, _ = lanes[src].project(x, y, EXT_BEFORE_M)
        s_tgt, d_tgt, _, _ = lanes[tgt].project(x, y, EXT_BEFORE_M)
        s_src, _, _, _ = lanes[src].project(x, y, EXT_BEFORE_M)
        v = eg["v"][m]
        if len(tt) < 3:
            continue
        d_src0 = float(np.interp(0.0, tt, d_src))
        d_tgt0 = float(np.interp(0.0, tt, d_tgt))
        W = abs(d_src0 - d_tgt0)            # lane spacing at the start pose
        v0 = float(np.interp(0.0, tt, v))
        after = tt >= 0.0

        # Supervisor debug text inside the interval
        sup = bt.sup_events(d, t0 - 0.05, t_end + 0.15, "")
        abort_txt = [(t, s) for t, s in sup if s.startswith("abort ")]
        rear_txt = [(t, s) for t, s in sup if s.startswith("rear_abort_confirmed") or
                    s.startswith("rear_abort_emergency")]
        pend_txt = [(t, s) for t, s in sup if s.startswith("rear_abort_pending")]
        motion = [t for t, s in sup if "MOTION_STARTED" in s]
        timeout_txt = [(t, s) for t, s in sup if s.startswith("timeout")]

        # Outcome
        ended = iv["ended"]
        d_tgt_end = float(np.interp(t_end - t0, tt, d_tgt)) if ended else np.nan
        d_src_end = float(np.interp(t_end - t0, tt, d_src)) if ended else np.nan
        if not ended:
            outcome = "censored_bag_end"
        elif abort_txt:
            outcome = "abort_rear" if rear_txt else "abort_other"
        elif sup or len(d["sup"]["t"]) > 0:
            outcome = "complete"
        else:
            # No supervisor debug topic in this bag: infer from pose at deactivation.
            outcome = ("complete_inferred" if abs(d_tgt_end) <= SUP_COMPLETE_D_M + 0.1
                       else "abort_inferred")

        # Preview mode at start (latest preview for this target before t0+0.1)
        mode = None
        blend = None
        if len(pv["t"]):
            j = np.searchsorted(pv["t"], t0 + 0.1) - 1
            while j >= 0 and pv["tgt"][j] != tgt and pv["t"][j] > t0 - 2.0:
                j -= 1
            if j >= 0 and pv["t"][j] > t0 - 2.0:
                mode = int(pv["mode"][j])
                blend = float(pv["blend_sec"][j])
        # targets context at start
        dist_end = None
        if len(tg["t"]):
            j = np.searchsorted(tg["t"], t0) - 1
            if j >= 0:
                dist_end = float(tg["dist_to_end"][j])

        abs_dt = np.abs(d_tgt)
        abs_ds = np.abs(d_src)
        e = {
            "bag": d["label"], "src": src, "tgt": tgt, "pair": "%d->%d" % (src, tgt),
            "t0": t0, "t_end": t_end, "t0_rel_bag": t0 - t_bag0,
            "status_duration": t_end - t0 if ended else None,
            "outcome": outcome, "partial_start": bool(partial_start),
            "v0": v0, "start_type": ("standstill" if v0 < 0.5 else
                                     ("moving" if v0 >= 3.0 else "slow")),
            "preview_mode": {1: "BLEND", 2: "CONNECTOR", 3: "DIRECT"}.get(mode, str(mode)),
            "blend_duration_sec": blend,
            "dist_to_lane_end_at_start": dist_end,
            "lane_spacing_m": W, "d_src0": d_src0, "d_tgt0": d_tgt0,
            "t_motion_started": (motion[0] - t0) if motion else None,
            "t_first_move": _first(tt, after & (v >= 0.2)),
            "t_cross_boundary": _first(tt, after & (abs_dt <= W / 2.0)),
            "t_d075": _first(tt, after & (abs_dt <= SUP_COMPLETE_D_M)),
            "t_center_045": _first(tt, after & (abs_dt <= CENTER_TOL_M)),
            "t_center_045_settled1s": _first_settled(tt, after & (abs_dt <= CENTER_TOL_M), 1.0),
            "t_supervisor_complete": (t_end - t0) if outcome.startswith("complete") else None,
            "d_tgt_end": d_tgt_end, "d_src_end": d_src_end,
            "v_at_center": None,
            "series": {"t": tt, "d_src": d_src, "d_tgt": d_tgt, "v": v, "s_tgt": s_tgt},
            "n_rear_pending": len(pend_txt),
            "timeout_text": [s for _, s in timeout_txt],
        }
        if e["t_center_045"] is not None:
            e["v_at_center"] = float(np.interp(e["t_center_045"], tt, v))
        # Progress fractions (relative to the start offset to the target centre)
        prog = 1.0 - abs_dt / max(abs(d_tgt0), 1e-3)
        for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
            e["t_progress_%02d" % int(frac * 100)] = _first(tt, after & (prog >= frac))

        # Abort details (M2)
        if rear_txt or abort_txt:
            ta, txt = (rear_txt[0] if rear_txt else abort_txt[0])
            kv = bt.parse_kv(txt)
            rel = ta - t0
            ds_a = float(np.interp(rel, tt, d_src))
            dt_a = float(np.interp(rel, tt, d_tgt))
            # post-abort window ends at the next accepted attempt (a retry
            # moves the ego laterally again) or 12 s after deactivation.
            post = (tt >= rel) & (tt < next_t0 - t0)
            peak_i = None
            if post.any():
                pi = np.flatnonzero(post)
                peak_i = pi[np.argmax(abs_ds[pi])]
            peak = float(abs_ds[peak_i]) if peak_i is not None else None
            t_peak = float(tt[peak_i]) - rel if peak_i is not None else None
            departed = peak is not None and peak > CENTER_TOL_M
            t_ret = t_ret_settled = None
            if departed:
                after_peak = post & (tt >= tt[peak_i])
                r1 = _first(tt, after_peak & (abs_ds <= CENTER_TOL_M))
                r2 = _first_settled(tt, after_peak & (abs_ds <= CENTER_TOL_M), 1.0)
                t_ret = None if r1 is None else r1 - rel
                t_ret_settled = None if r2 is None else r2 - rel
            v_post = v[post] if post.any() else np.zeros(0)
            if not departed:
                ret_outcome = "never_departed_0.45m"
            elif t_ret is not None:
                ret_outcome = "returned"
            elif len(v_post) and np.all(v_post[-min(len(v_post), 30):] < 0.2):
                ret_outcome = "stopped_between_lanes"
            else:
                ret_outcome = "not_returned_in_window"
            e["abort"] = {
                "kind": txt.split()[0], "t_abort_rel": rel,
                "text": txt, "abort_line": abort_txt[0][1] if abort_txt else None,
                "gap": kv.get("gap"), "closing": kv.get("closing"), "ttc": kv.get("ttc"),
                "required": kv.get("required"), "remaining": kv.get("remaining"),
                "escapable": kv.get("escapable"), "launch_grace": kv.get("launch_grace"),
                "rear_id": kv.get("rear_id"),
                "d_src_at_abort": ds_a, "d_tgt_at_abort": dt_a,
                "lateral_progress_m": abs(ds_a - d_src0) if np.isfinite(ds_a) else None,
                "lateral_progress_frac": abs(ds_a - d_src0) / max(W, 1e-3),
                "abs_d_src_at_abort": abs(ds_a),
                "v_at_abort": float(np.interp(rel, tt, v)),
                "peak_abs_d_src_after_abort": peak,
                "t_peak_after_abort": t_peak,
                "peak_frac_of_lane": (peak / max(W, 1e-3)) if peak is not None else None,
                "t_return_045": t_ret,
                "t_return_045_settled1s": t_ret_settled,
                "return_outcome": ret_outcome,
                "post_window_s": float(min(tt[-1], next_t0 - t0) - rel),
            }
        ev.append(e)
    return ev


def strip_series(e):
    return {k: v for k, v in e.items() if k != "series"}
