#!/usr/bin/env python3
"""M1: lane-change duration and lateral progress vs time.

Start  t0 = rising edge of /smpc/lane_change_status.active (header stamp); this
          is the same instant the supervisor publishes /path_number (checked).
Ends   (a) ego rear-axle |d| to target-lane CSV centre <= 0.45 m (first time),
       (b) same, held for 1 s,
       (c) supervisor completion (status falling edge, no abort),
       plus boundary crossing |d_tgt| <= W/2 (W = lane spacing at the start pose)
       and progress fractions (1 - |d_tgt|/|d_tgt(t0)|).
Population for the duration = completed changes whose start lies inside the bag.
Control = every accepted attempt (aborted / censored ones listed by outcome).

  python3 research/analysis/m1_lc_duration.py --bags ~/bags/lc_gt_2026-09-28-*.bag
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagtools as bt  # noqa: E402
import lcevents  # noqa: E402

METRICS = ["t_cross_boundary", "t_d075", "t_center_045", "t_center_045_settled1s",
           "t_supervisor_complete", "t_motion_started", "t_first_move",
           "t_progress_10", "t_progress_25", "t_progress_50", "t_progress_75",
           "t_progress_90"]
GRID = np.round(np.arange(0.0, 8.01, 0.25), 2)


def summarize(evs):
    out = {"n": len(evs)}
    for m in METRICS:
        out[m] = bt.stats([e[m] for e in evs if e.get(m) is not None])
    out["n_never_center_045_within_window"] = sum(1 for e in evs if e["t_center_045"] is None)
    out["v0"] = bt.stats([e["v0"] for e in evs])
    out["lane_spacing_m"] = bt.stats([e["lane_spacing_m"] for e in evs])
    return out


def progress_curve(evs):
    rows = []
    for e in evs:
        s = e["series"]
        tt, dt = s["t"], np.abs(s["d_tgt"])
        p = 1.0 - dt / max(abs(e["d_tgt0"]), 1e-3)
        rows.append(np.interp(GRID, tt, p, left=np.nan, right=np.nan))
    if not rows:
        return {}
    a = np.vstack(rows)
    return {"t": GRID.tolist(),
            "median": np.nanmedian(a, axis=0).tolist(),
            "p10": np.nanpercentile(a, 10, axis=0).tolist(),
            "p90": np.nanpercentile(a, 90, axis=0).tolist(),
            "n": int(a.shape[0])}


def main():
    ap = bt.base_argparser(__doc__)
    args = ap.parse_args()
    paths = bt.expand_bags(args.bags)
    lanes = bt.load_lanes(args.lanes)
    ds = bt.load_bags(paths, args.cache, not args.no_cache, args.jobs)
    events = []
    pn_check = []
    for d in ds:
        evs = lcevents.lc_events(d, lanes)
        # /path_number publish (bag time) vs status rising edge (header stamp)
        pn = d["path_number"]
        for e in evs:
            if len(pn["t"]):
                k = np.argmin(np.abs(pn["t"] - e["t0"]))
                if pn["data"][k] == e["tgt"] and abs(pn["t"][k] - e["t0"]) < 1.0:
                    pn_check.append(pn["t"][k] - e["t0"])
        events.extend(evs)

    full = [e for e in events if not e["partial_start"]]
    done = [e for e in full if e["outcome"].startswith("complete")]
    groups = {
        "completed_all": done,
        "completed_moving": [e for e in done if e["start_type"] == "moving"],
        "completed_standstill": [e for e in done if e["start_type"] == "standstill"],
        "completed_slow_0.5_3mps": [e for e in done if e["start_type"] == "slow"],
        "completed_BLEND": [e for e in done if e["preview_mode"] == "BLEND"],
        "completed_CONNECTOR": [e for e in done if e["preview_mode"] == "CONNECTOR"],
    }
    for pair in ("0->1", "1->2", "2->3"):
        groups["completed_" + pair] = [e for e in done if e["pair"] == pair]
        groups["completed_moving_" + pair] = [e for e in done if e["pair"] == pair and
                                              e["start_type"] == "moving"]
    dates = sorted(set(e["bag"][:10] for e in done))
    by_date = {dt: summarize([e for e in done if e["bag"].startswith(dt)]) for dt in dates}

    outcome_counts = {}
    for e in events:
        k = e["outcome"] + ("|partial_start" if e["partial_start"] else "")
        outcome_counts[k] = outcome_counts.get(k, 0) + 1

    res = {
        "definition": {
            "t0": "rising edge of /smpc/lane_change_status.active (header.stamp)",
            "lateral_point": "/odom rear axle projected on lane CSVs (same as target_selector/supervisor)",
            "center_tol_m": lcevents.CENTER_TOL_M,
            "boundary": "|d_tgt| <= W/2, W = |d_src - d_tgt| at t0",
            "supervisor_complete": "status falling edge without abort (|d|<=0.75 m x5, >=2 s)",
        },
        "n_bags": len(ds),
        "bags": [d["label"] for d in ds],
        "n_attempts": len(events),
        "outcome_counts": outcome_counts,
        "path_number_minus_status_t0_s": bt.stats(pn_check),
        "summary": {k: summarize(v) for k, v in groups.items()},
        "control_all_attempts_full_start": summarize(full),
        "by_date_completed": by_date,
        "progress_curve": {
            "completed_moving": progress_curve(groups["completed_moving"]),
            "completed_standstill": progress_curve(groups["completed_standstill"]),
        },
        "events": [lcevents.strip_series(e) for e in events],
    }
    for e in res["events"]:
        e.pop("abort", None)
    out = os.path.join(args.out, "day1_m1.json")
    bt.write_json(out, res)

    print("bags=%d attempts=%d outcomes=%s" % (len(ds), len(events), outcome_counts))
    print("path_number - status t0: %s" % bt.fmt_stats(res["path_number_minus_status_t0_s"], 3))
    for k in ["completed_all", "completed_moving", "completed_standstill",
              "completed_slow_0.5_3mps", "completed_BLEND", "completed_CONNECTOR",
              "completed_0->1", "completed_1->2", "completed_2->3"]:
        s = res["summary"][k]
        print("%-26s n=%-3d center045 %s" % (k, s["n"], bt.fmt_stats(s["t_center_045"])))
        print("%-26s       supv_done %s" % ("", bt.fmt_stats(s["t_supervisor_complete"])))
        print("%-26s       cross     %s" % ("", bt.fmt_stats(s["t_cross_boundary"])))
    for dt, s in by_date.items():
        print("date %s n=%d center045 %s | supv %s" % (dt, s["n"], bt.fmt_stats(s["t_center_045"]),
                                                     bt.fmt_stats(s["t_supervisor_complete"])))
    print("wrote", out)


if __name__ == "__main__":
    main()
