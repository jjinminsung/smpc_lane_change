#!/usr/bin/env python3
"""M2: rear-abort (return) events.

For each rear_abort_confirmed / rear_abort_emergency on
/smpc/lane_change_supervisor_debug (std_msgs/String -> bag receive time; the
bags' header stamps and receive times share the wall clock within ~10 ms):
  * lateral progress at the abort moment (|d_src(t_abort) - d_src(t0)| and as a
    fraction of the lane spacing, rear-axle /odom on the source-lane CSV),
  * time from abort until |d_src| <= 0.45 m (first, and held 1 s).
Control group: all accepted attempts (how many aborted, at which progress).

  python3 research/analysis/m2_abort_return.py --bags ~/bags/lc_gt_2026-09-28-*.bag
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagtools as bt  # noqa: E402
import lcevents  # noqa: E402


def main():
    ap = bt.base_argparser(__doc__)
    args = ap.parse_args()
    paths = bt.expand_bags(args.bags)
    lanes = bt.load_lanes(args.lanes)
    ds = bt.load_bags(paths, args.cache, not args.no_cache, args.jobs)
    rows, n_attempts, inferred = [], 0, []
    for d in ds:
        for e in lcevents.lc_events(d, lanes):
            n_attempts += 1
            if e["outcome"] == "abort_inferred":
                inferred.append(lcevents.strip_series(e))
            if "abort" not in e:
                continue
            a = e["abort"]
            s = e["series"]
            # lateral trace around the abort for the record (0.25 s grid)
            g = np.round(np.arange(-1.0, 6.01, 0.25), 2) + a["t_abort_rel"]
            trace = np.interp(g, s["t"], np.abs(s["d_src"]), left=np.nan, right=np.nan)
            rows.append({
                "bag": e["bag"], "pair": e["pair"], "start_type": e["start_type"],
                "preview_mode": e["preview_mode"], "v0": e["v0"],
                "kind": a["kind"], "t_abort_after_start": a["t_abort_rel"],
                "t_motion_started": e["t_motion_started"],
                "v_at_abort": a["v_at_abort"],
                "lateral_progress_m": a["lateral_progress_m"],
                "lateral_progress_frac": a["lateral_progress_frac"],
                "abs_d_src_at_abort": a["abs_d_src_at_abort"],
                "peak_abs_d_src_after_abort": a["peak_abs_d_src_after_abort"],
                "t_peak_after_abort": a["t_peak_after_abort"],
                "peak_frac_of_lane": a["peak_frac_of_lane"],
                "t_return_045": a["t_return_045"],
                "t_return_045_settled1s": a["t_return_045_settled1s"],
                "return_outcome": a["return_outcome"],
                "post_window_s": a["post_window_s"],
                "rear_id": a["rear_id"], "gap": a["gap"], "closing": a["closing"],
                "ttc": a["ttc"], "required": a["required"], "remaining": a["remaining"],
                "escapable": a["escapable"], "launch_grace": a["launch_grace"],
                "n_rear_pending": e["n_rear_pending"],
                "abort_line": a["abort_line"],
                "abs_d_src_trace_rel_abort": {"t": (g - a["t_abort_rel"]).tolist(),
                                              "abs_d_src": trace.tolist()},
            })
    moved = [r for r in rows if r["return_outcome"] != "never_departed_0.45m"]
    res = {
        "n_bags": len(ds), "n_attempts": n_attempts, "n_rear_aborts": len(rows),
        "n_rear_aborts_with_lateral_departure_gt_045": len(moved),
        "abort_kind_counts": {k: sum(1 for r in rows if r["kind"] == k)
                              for k in sorted(set(r["kind"] for r in rows))},
        "lateral_progress_m_all_aborts": bt.stats([r["lateral_progress_m"] for r in rows]),
        "t_abort_after_start_all": bt.stats([r["t_abort_after_start"] for r in rows]),
        "t_return_045_departed": bt.stats([r["t_return_045"] for r in moved]),
        "return_outcome_counts": {k: sum(1 for r in rows if r["return_outcome"] == k)
                                  for k in sorted(set(r["return_outcome"] for r in rows))},
        "peak_abs_d_src_after_abort_all": bt.stats([r["peak_abs_d_src_after_abort"] for r in rows]),
        "aborts_without_supervisor_debug_inferred": inferred,
        "events": rows,
    }
    out = os.path.join(args.out, "day1_m2.json")
    bt.write_json(out, res)
    print("bags=%d attempts=%d rear_aborts=%d (departed >0.45 m: %d) kinds=%s" % (
        len(ds), n_attempts, len(rows), len(moved), res["abort_kind_counts"]))
    print("%-20s %-5s %-10s %-9s %5s %5s %6s %5s %6s %6s %6s %-22s %s" % (
        "bag", "pair", "start", "kind", "t_ab", "v_ab", "prog_m", "frac", "peak", "t_pk",
        "t_ret", "return", "gap/closing/ttc"))
    for r in rows:
        f = lambda x: "%.2f" % x if x is not None else "-"  # noqa: E731
        print("%-20s %-5s %-10s %-9s %5.2f %5.2f %6.2f %5.2f %6s %6s %6s %-22s %s/%s/%s" % (
            r["bag"], r["pair"], r["start_type"], r["kind"].replace("rear_abort_", ""),
            r["t_abort_after_start"], r["v_at_abort"], r["lateral_progress_m"],
            r["lateral_progress_frac"], f(r["peak_abs_d_src_after_abort"]),
            f(r["t_peak_after_abort"]), f(r["t_return_045"]), r["return_outcome"],
            r["gap"], r["closing"], r["ttc"]))
    print("inferred aborts (no supervisor debug):", len(inferred))
    print("wrote", out)


if __name__ == "__main__":
    main()
