#!/usr/bin/env python3
"""Merge day1_m1..m4.json into day1_measurements.json (+ .md short tables).

  python3 research/analysis/day1_report.py            # after the m1..m4 scripts
  python3 research/analysis/run_day1.py --bags ...     # runs everything
"""
import argparse
import datetime
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagtools as bt  # noqa: E402


def load(out, name):
    with open(os.path.join(out, name)) as f:
        return json.load(f)


def s3(s, keys=("n", "mean", "median", "p10", "p90"), d=2):
    if not s or not s.get("n"):
        return {"n": 0}
    return {k: (round(s[k], d) if isinstance(s[k], float) else s[k]) for k in keys}


def cell(s, k="median", d=2):
    if not s or not s.get("n"):
        return "-"
    return ("%%.%df" % d) % s[k]


def row_stats(s, d=2):
    """median (p10-p90), mean"""
    if not s or not s.get("n"):
        return "-"
    f = "%%.%df" % d
    return (f + " (" + f + "-" + f + "), mean " + f) % (s["median"], s["p10"], s["p90"], s["mean"])


def num(x, d=2):
    return "-" if x is None else ("%%.%df" % d) % x


def g(dct, *keys):
    """Nested get that returns None for missing keys (small bag subsets)."""
    for k in keys:
        if not isinstance(dct, dict) or k not in dct:
            return None
        dct = dct[k]
    return dct


def git_rev(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=bt.RESULTS_DIR)
    args = ap.parse_args()
    m1, m2, m3, m4 = (load(args.out, "day1_m%d.json" % i) for i in (1, 2, 3, 4))
    S = m1["summary"]
    repo = os.path.dirname(bt.RESEARCH)

    # ------------------------------------------------------------- JSON
    sim = {
        "lane_change": {
            "moving_start_BLEND": {k: s3(S["completed_moving"][k]) for k in (
                "t_cross_boundary", "t_d075", "t_center_045", "t_supervisor_complete")},
            "standstill_start_CONNECTOR": {k: s3(S["completed_standstill"][k]) for k in (
                "t_motion_started", "t_cross_boundary", "t_d075", "t_center_045", "t_supervisor_complete")},
            "progress_curve": m1["progress_curve"],
            "lane_spacing_m": s3(S["completed_all"]["lane_spacing_m"]),
        },
        "abort_return": {
            "n_rear_aborts": m2["n_rear_aborts"],
            "return_outcome_counts": m2["return_outcome_counts"],
            "departed_events": [{k: e[k] for k in (
                "bag", "pair", "start_type", "v_at_abort", "t_abort_after_start",
                "lateral_progress_m", "peak_abs_d_src_after_abort", "t_peak_after_abort",
                "t_return_045", "return_outcome")} for e in m2["events"]
                if e["return_outcome"] != "never_departed_0.45m"],
        },
        "npc": m3["parametric_for_sim"],
        "npc_lane_change_rate_per_npc_min_mid_lane": (
            m3["lane_changes"]["kind_counts"]["mid_lane"] / m3["highway_npc_minutes"]),
        "mode_detection": {
            "update_rate_hz": {"object_topic_mode": s3(m4["rates"]["object_topic_hz_per_bag"]),
                               "lidar_targets_recorded": s3(m4["rates"]["targets_hz_lidar_mode_per_bag"])},
            "gt_object_rate_from_steady": m4["gt_object_rate_delay_from_steady"],
            "gt_lidar_rate_from_steady": m4["gt_lidar_rate_delay_from_steady"],
            "detection_curve": m4["detection_curve"],
            "false_positive_steady_cruise": m4["false_positive_steady_cruise"],
            "probability_table_gt_event": m4["probability_table_gt_event"],
        },
    }
    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "baseline_repo": repo, "baseline_commit": git_rev(repo),
        "bags": m1["bags"], "n_bags": m1["n_bags"],
        "lane_csv_dir": bt.DEFAULT_LANE_DIR,
        "time_axis": "header.stamp (String/Int32 topics: bag receive time)",
        "ego_lateral_point": "/odom rear axle (same as target_selector / supervisor)",
        "gt_frame": "Object_topic/Ego_topic MORAI-local; +(302594.99, 4124145.00) m -> UTM (per-bag estimate, p95 spread < 0.25 m)",
        "object_topic_range_m": 300,
        "detail_files": ["day1_m1.json", "day1_m2.json", "day1_m3.json", "day1_m4.json",
                         "day1_m3_tracks.csv", "day1_m3_episodes.csv", "day1_m3_relevant_pairs.csv",
                         "day1_m3_profiles.csv"],
    }
    combined = {
        "meta": meta,
        "sim_params": sim,
        "M1_lc_duration": {k: m1[k] for k in ("definition", "n_attempts", "outcome_counts",
                                               "path_number_minus_status_t0_s", "summary",
                                               "control_all_attempts_full_start", "by_date_completed",
                                               "progress_curve")},
        "M2_abort_return": m2,
        "M3_npc_longitudinal": {k: m3[k] for k in m3 if k not in ("episodes_sample_note",)},
        "M4_mode_detection": {k: m4[k] for k in m4 if k not in ("gt_episode_rows", "tracker_episode_rows")},
    }
    combined["M3_npc_longitudinal"]["lane_changes"] = {
        k: v for k, v in m3["lane_changes"].items() if k != "events"}
    bt.write_json(os.path.join(args.out, "day1_measurements.json"), combined)

    # ------------------------------------------------------------- Markdown
    L = []
    w = L.append
    w("# Day-1 measurements (baseline %s, %d bags)" % (meta["baseline_commit"], meta["n_bags"]))
    w("")
    w("Generated %s by `research/analysis/run_day1.py`. Bags: %d `lc_gt_*.bag` (%s .. %s). "
      "Time axis = header.stamp. Ego lateral = /odom rear axle on lane CSVs "
      "(`waypoint_system/data/default`, matches /smpc/targets ego_d to p99 < 0.13 m). "
      "Full numbers: `day1_measurements.json`." % (meta["generated"], meta["n_bags"],
                                                   min(meta["bags"]), max(meta["bags"])))
    w("")
    w("## M1 Lane-change duration (from status rising edge = /path_number publish)")
    w("")
    w("Population: completed changes with the start inside the bag. Times in s after t0: median (p10-p90), mean.")
    w("")
    w("| population | N | cross boundary (abs(d_tgt) <= W/2) | abs(d_tgt) <= 0.45 m | supervisor complete |")
    w("|---|---|---|---|---|")
    for key, name in [("completed_all", "all completed"),
                      ("completed_moving", "moving start (v0>=3, BLEND)"),
                      ("completed_standstill", "standstill start (CONNECTOR)"),
                      ("completed_moving_0->1", "moving 0->1"),
                      ("completed_moving_1->2", "moving 1->2"),
                      ("completed_moving_2->3", "moving 2->3")]:
        s = S[key]
        w("| %s | %d | %s | %s | %s |" % (name, s["n"], row_stats(s["t_cross_boundary"]),
                                           row_stats(s["t_center_045"]),
                                           row_stats(s["t_supervisor_complete"])))
    mv, al = S["completed_moving"]["t_center_045"], S["completed_all"]["t_center_045"]
    w("")
    if mv.get("n") and al.get("n"):
        w("Doc value 4.16 s: matches only the pooled mean of two different populations (all completed: "
          "mean %.2f, supervisor mean %.2f). For a moving-start change (BLEND) the 0.45 m time is median "
          "%.2f / p90 %.2f s, so 2.0 + 0.55 + %.2f + 1.5 = %.2f s vs the 8.0 s horizon (doc: 8.21 s)."
          % (al["mean"], S["completed_all"]["t_supervisor_complete"]["mean"], mv["median"], mv["p90"],
             mv["median"], 2.0 + 0.55 + mv["median"] + 1.5))
    else:
        w("Doc value 4.16 s: not enough completed changes in this bag subset.")
    oc = m1["outcome_counts"]
    w("")
    w("Control (all %d accepted attempts): %s. Standstill starts: motion begins %s s (median) after t0."
      % (m1["n_attempts"], ", ".join("%s %d" % kv for kv in sorted(oc.items())),
         num(g(S, "completed_standstill", "t_motion_started", "median"))))
    w("")
    pm, ps = m1["progress_curve"]["completed_moving"], m1["progress_curve"]["completed_standstill"]
    tgrid = [x / 2.0 for x in range(13)]
    w("Lateral progress 1 - abs(d_tgt)/abs(d_tgt(t0)) (median):")
    w("")
    w("| t [s] | " + " | ".join("%.1f" % t for t in tgrid) + " |")
    w("|---|" + "---|" * len(tgrid))
    for name, pc in (("moving", pm), ("standstill", ps)):
        vals = dict(zip(pc.get("t", []), pc.get("median", [])))
        w("| %s | " % name + " | ".join(num(vals.get(t)) for t in tgrid) + " |")
    w("")
    w("By date (completed, 0.45 m median / supervisor median): " + "; ".join(
        "%s n=%d %s/%s" % (dt[5:], s["n"], cell(s["t_center_045"]), cell(s["t_supervisor_complete"]))
        for dt, s in m1["by_date_completed"].items()))
    w("")
    w("## M2 Rear-abort return")
    w("")
    w("%d rear aborts (%s) in %d attempts; %d never left the source lane (abs(d_src) stayed <= 0.45 m)."
      % (m2["n_rear_aborts"], ", ".join("%s %d" % (k.replace("rear_abort_", ""), v)
                                        for k, v in m2["abort_kind_counts"].items()),
         m2["n_attempts"], m2["return_outcome_counts"].get("never_departed_0.45m", 0)))
    nd = [e for e in m2["events"] if e["return_outcome"] == "never_departed_0.45m"]
    w("Never-departed aborts: %d standstill / %d moving, abort %.2f s (median) after t0, lateral progress <= %.2f m."
      % (sum(1 for e in nd if e["start_type"] == "standstill"), sum(1 for e in nd if e["start_type"] != "standstill"),
         sorted(e["t_abort_after_start"] for e in nd)[len(nd) // 2] if nd else float("nan"),
         max(e["lateral_progress_m"] for e in nd) if nd else float("nan")))
    w("")
    w("Aborts where the ego left the source lane (after the abort the reference path is re-blended from the "
      "target lane, so a moving ego first swings into the target lane):")
    w("")
    w("| bag | pair | v@abort | t abort after start | progress@abort m (frac) | peak abs(d_src) after abort (t) | back <= 0.45 m after | outcome |")
    w("|---|---|---|---|---|---|---|---|")
    for e in m2["events"]:
        if e["return_outcome"] == "never_departed_0.45m":
            continue
        w("| %s | %s | %.1f | %.2f | %.2f (%.2f) | %.2f m (%.2f s) | %s | %s |" % (
            e["bag"], e["pair"], e["v_at_abort"], e["t_abort_after_start"],
            e["lateral_progress_m"], e["lateral_progress_frac"],
            e["peak_abs_d_src_after_abort"], e["t_peak_after_abort"],
            "%.2f s" % e["t_return_045"] if e["t_return_045"] is not None else "never",
            e["return_outcome"]))
    w("")
    w("## M3 NPC longitudinal behaviour (GT, unique_id)")
    w("")
    A, R = m3["all_highway"], m3["relevant"]
    w("%d highway NPC tracks (%.0f NPC-min), %d spawned in view. /Object_topic only lists NPCs within ~300 m "
      "of the ego. MORAI acceleration.x vs our derivative: median corr %.2f."
      % (m3["n_highway_tracks"], m3["highway_npc_minutes"], m3["n_spawned_in_bag"],
         m3["morai_ax_vs_derivative_corr"]["median"]))
    w("")
    w("| quantity | all highway NPCs | relevant rear (target lane, -120..0 m) | relevant front (0..+100 m) |")
    w("|---|---|---|---|")
    fa, fr, ff = A["time_fraction"], R["time_fraction_rear"], R["time_fraction_front"]
    for k in ("a>0.3", "a>0.5", "a>1.0", "a<-0.3"):
        w("| time fraction %s m/s^2 | %.2f | %.2f | %.2f |" % (k, fa[k], fr[k], ff[k]))
    pr, pf = R["pairs"]["rear_all"], R["pairs"]["front_all"]
    w("| (window, NPC) pairs / unique NPCs | - | %s / %s | %s / %s |" % (
        pr.get("n_pairs", 0), pr.get("n_unique_npcs", 0), pf.get("n_pairs", 0), pf.get("n_unique_npcs", 0)))
    w("| P(accel episode overlaps window) | - | %s | %s |" % (num(pr.get("P_accel_episode_overlaps")), num(pf.get("P_accel_episode_overlaps"))))
    w("| ... of which spawn-acceleration episode | - | %s | %s |" % (
        num(pr.get("frac_accel_eps_that_are_spawn_episode")), num(pf.get("frac_accel_eps_that_are_spawn_episode"))))
    w("| P(accel onset inside window) | - | %s | %s |" % (num(pr.get("P_accel_onset_inside_window")), num(pf.get("P_accel_onset_inside_window"))))
    w("| accel episode mean a (median, p10-p90) | %s (%s-%s) | %s (%s-%s) | %s (%s-%s) |" % (
        cell(A["accel_nonspawn"]["mean_accel"]), cell(A["accel_nonspawn"]["mean_accel"], "p10"),
        cell(A["accel_nonspawn"]["mean_accel"], "p90"),
        cell(pr.get("accel_mean")), cell(pr.get("accel_mean"), "p10"), cell(pr.get("accel_mean"), "p90"),
        cell(pf.get("accel_mean")), cell(pf.get("accel_mean"), "p10"), cell(pf.get("accel_mean"), "p90")))
    w("")
    w("Rear P(accelerating) by lane pair: " + "; ".join(
        "%s %.2f (n=%d)" % (k[5:], R["pairs"][k]["P_accel_episode_overlaps"], R["pairs"][k]["n_pairs"])
        for k in ("rear_0->1", "rear_1->2", "rear_2->3") if R["pairs"][k].get("n_pairs")) +
      "; lc_attempt %s, lane_end_wait %s." % (num(g(R, "pairs", "rear_lc_attempt", "P_accel_episode_overlaps")),
                                               num(g(R, "pairs", "rear_lane_end_wait", "P_accel_episode_overlaps"))))
    w("")
    P = m3["parametric_for_sim"]
    w("Parametric summary (all highway NPCs; median [p10, p90]):")
    w("")
    w("| parameter | value |")
    w("|---|---|")
    w("| spawn speed | median %s m/s; %s%% at 13.89 m/s, %s%% at 0 m/s |" % (
        num(A["spawn_speed"].get("median")), num(100 * (A["spawn_speed_frac_at_13.89"] or 0), 0),
        num(100 * (A["spawn_frac_at_0mps"] or 0), 0)))
    for name, s in (("spawn acceleration", P["spawn_accel_mps2"]),
                    ("spawn acceleration duration [s]", P["spawn_accel_duration_s"]),
                    ("speed at end of spawn acceleration", P["desired_speed_after_spawn_mps"]),
                    ("cruise plateau speed (abs(a)<0.15 for >=2 s)", P["cruise_speed_plateaus_mps"]),
                    ("non-spawn accel episode mean a", P["accel_mean_mps2"]),
                    ("accel episode duration [s]", P["accel_duration_s"]),
                    ("accel episode dv [m/s]", P["accel_dv_mps"]),
                    ("inter-onset interval of accel episodes [s]", P["accel_inter_onset_s"]),
                    ("decel episode mean abs(a)", P["decel_mean_mps2"]),
                    ("decel episode duration [s]", P["decel_duration_s"])):
        w("| %s | %s [%s, %s] (n=%d) |" % (name, cell(s), cell(s, "p10"), cell(s, "p90"), s.get("n", 0)))
    w("| accel / decel episode rate | %.2f / %.2f per NPC-min |" % (
        P["accel_episode_rate_per_npc_min"], P["decel_episode_rate_per_npc_min"]))
    h = P["accel_mean_hist"]
    w("| accel mean-a histogram %s | %s |" % (h["bins"], [round(x, 3) for x in h.get("frac", [])]))
    w("")
    lc = m3["lane_changes"]
    w("NPC lane changes: yes. %d mid-lane changes by %d/%d NPCs (%.2f per NPC-min; %s), plus %d lane-end merges "
      "and %d in the upstream extension zone. %d mid-lane changes happened while the NPC was relevant to a "
      "decision window, %d of them into the ego's target lane. Lateral rate median %.2f m/s."
      % (lc["kind_counts"]["mid_lane"], lc["n_npcs_with_mid_lane_change"], lc["n_highway_npcs"],
         sim["npc_lane_change_rate_per_npc_min_mid_lane"],
         ", ".join("%s %d" % kv for kv in lc["mid_lane_directions"].items() if kv[1]),
         lc["kind_counts"]["lane_end_merge"], lc["kind_counts"]["upstream_extension_zone"],
         lc["mid_lane_in_relevant_window"], lc["mid_lane_into_target_lane_in_window"],
         lc["mid_lane_lateral_rate_mps"].get("median") or float("nan")))
    w("")
    w("## M4 Mode detection delay (filter a = 0.35 raw + 0.65 a, raw = clamp(dv/dt, +-6))")
    w("")
    rt = m4["rates"]
    w("Update rate: recorded /smpc/targets (LiDAR mode) %.2f Hz median per bag (gaps > 0.35 s: %.2f%%); "
      "object_topic mode = one /smpc/targets per /Object_topic message = %.1f Hz (dt median %.3f s, "
      "%.1f%% of dt <= 1 ms and skipped by the 1e-3 guard; %.0f%% of moving-NPC frames repeat the previous state)."
      % (rt["targets_hz_lidar_mode_per_bag"]["median"], 100 * rt["targets_frac_dt_gt_0.35"]["mean"],
         rt["object_topic_hz_per_bag"]["median"], rt["object_dt_median"]["median"],
         100 * rt["object_frac_dt_le_1ms"]["mean"], 100 * rt["gt_repeated_frame_frac_moving"]["median"]))
    w("")
    w("Delay [s] from true onset (foot of the GT acceleration rise) to filtered > threshold, onsets from "
      "steady (abs(a) < 0.2 for 1 s before). median (p10-p90), detected fraction:")
    w("")
    w("| true mean accel | N | GT @Object_topic >0.5 | >1.0 | GT @9.4 Hz >0.5 | >1.0 | tracker >0.5 (first) | >0.5 (3 consecutive) |")
    w("|---|---|---|---|---|---|---|---|")
    G, GL, TR, TS = (m4["gt_object_rate_delay_from_steady"], m4["gt_lidar_rate_delay_from_steady"],
                     m4["tracker_delay_all"], m4["tracker_delay_sustained3_all"])

    def dc(ent, thr):
        s = ent["delay_%s" % thr]
        if not s.get("n"):
            return "-"
        return "%.2f (%.2f-%.2f) %.0f%%" % (s["median"], s["p10"], s["p90"], 100 * ent["frac_detected_%s" % thr])

    for b in ("0.30-0.75", "0.75-1.25", "1.25-2.00", "all"):
        w("| %s | %d / trk %d | %s | %s | %s | %s | %s | %s |" % (
            b, G[b]["n"], TR[b]["n"], dc(G[b], "0.5"), dc(G[b], "1.0"), dc(GL[b], "0.5"), dc(GL[b], "1.0"),
            dc(TR[b], "0.5"), dc(TS[b], "0.5")))
    w("")
    w("Filter lag behind the true acceleration crossing the same threshold (median): GT@Object_topic %.2f s, "
      "GT@9.4 Hz %.2f s. The rest of the delay is the GT acceleration ramp itself."
      % (G["all"]["lag_vs_true_crossing_0.5"].get("median", float("nan")),
         GL["all"]["lag_vs_true_crossing_0.5"].get("median", float("nan"))))
    w("")
    fp = m4["false_positive_steady_cruise"]
    w("False positives in GT steady cruise (abs(a_true) < 0.2 over +-1 s, v > 3 m/s):")
    w("")
    w("| input | steady min | frac >0.5 | frac >1.0 | frac <-0.5 | crossings/min >0.5 | >1.0 |")
    w("|---|---|---|---|---|---|---|")
    for k, name in (("gt_obj", "GT @Object_topic"), ("gt_lidar_rate", "GT @9.4 Hz"),
                    ("trk_all", "tracker, all matched"), ("trk_target_rear", "tracker, target_rear role"),
                    ("gt_obj_at_trk_target_rear", "GT @Object_topic, same target_rear samples")):
        f = fp[k]
        w("| %s | %.1f | %.3f | %.3f | %.3f | %.1f | %.1f |" % (
            name, f["steady_minutes"], f["frac_gt_0.5"], f["frac_gt_1.0"], f["frac_lt_-0.5"],
            f["crossings_per_min_0.5"], f["crossings_per_min_1.0"]))
    tm = m4["tracker_match"]
    w("")
    w("Tracker vs GT: match distance median %.2f m; v_long error median %.2f m/s (p10 %.2f, p90 %.2f); "
      "%.0f%% of tracker onsets already had filtered > 0.5 in the 1 s before the true onset."
      % (tm["match_dist_m"].get("median", float("nan")), tm["v_long_err_mps_all"].get("median", float("nan")),
         tm["v_long_err_mps_all"].get("p10", float("nan")), tm["v_long_err_mps_all"].get("p90", float("nan")),
         100 * (m4["tracker_frac_episodes_already_above_0.5_before_onset"] or float("nan"))))
    w("")
    w("Rear-accel probability table check (target_rear samples; event = GT 5 s displacement >= 12.5 m above "
      "constant velocity, the yaml's event definition):")
    w("")
    tb = m4["probability_table_gt_event"]
    w("| filtered accel bin | " + " | ".join(r["bin"] for r in tb["trk_target_rear"]) + " |")
    w("|---|" + "---|" * 5)
    w("| tracker filter: event rate (n) | " + " | ".join(
        "%.3f (%d)" % (r["event_rate"], r["n"]) if r["event_rate"] is not None else "-" for r in tb["trk_target_rear"]) + " |")
    w("| GT@Object_topic filter: event rate (n) | " + " | ".join(
        "%.3f (%d)" % (r["event_rate"], r["n"]) if r["event_rate"] is not None else "-" for r in tb["gt_obj_target_rear"]) + " |")
    w("| yaml table (tracker fit, 95% upper) | 0.11 | 0.15 | 0.28 | 0.30 | 0.31 |")
    w("")
    w("## Caveats")
    w("")
    for c in CAVEATS:
        w("- " + c)
    w("")
    with open(os.path.join(args.out, "day1_measurements.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote", os.path.join(args.out, "day1_measurements.json"))
    print("wrote", os.path.join(args.out, "day1_measurements.md"))


CAVEATS = [
    "Bags span 2026-09-04..09-28; src/ is identical between 4fc908c (09-22) and the baseline dd1cf54 (09-28 13:48, "
    "after the last bag), only config changed. Only 3 completed changes are in the 09-28 bags, so pooled numbers "
    "mix code versions (see M1 by-date row). Lane-change timing is dominated by waypoint_system's 4.0 s blend "
    "(path_switch_preview.blend_duration_sec), which is outside this repo.",
    "Ego lateral offset uses the /odom rear axle, like target_selector and the supervisor; the vehicle centre is "
    "~1.5 m ahead and crosses the boundary slightly earlier.",
    "/Object_topic and /Ego_topic positions are MORAI-local; /odom and the lane CSVs are UTM. Baseline "
    "target_selector objectCallback uses npc.position unchanged, so target_source:=object_topic will not put any "
    "NPC on a lane CSV unless the ~(302594.99, 4124145.00) m offset is added.",
    "/Object_topic only contains NPCs within ~300 m of the ego; NPCs that leave and re-enter keep their "
    "unique_id, and the gap is masked out of the acceleration estimate.",
    "MORAI NPCs are strongly non-stationary: they spawn near the start of lanes 1-3 at 13.89 m/s or 0 m/s and "
    "accelerate at ~1.45 m/s^2 for ~8 s. Most target-rear accelerations in 0->1 windows are this spawn "
    "acceleration, already running before the window opens (onset inside the window only ~7%).",
    "Lane-end wait windows: status inactive, lane < 3, ego v < 0.5 m/s, distance_to_lane_end <= 60 m. LC attempt "
    "windows include the 2 s before approval.",
    "The tracker delay is a lower bound polluted by noise: in GT steady cruise the tracker filter exceeds 0.5 m/s^2 "
    "~30% of the time, so 'first crossing after onset' is often noise. Use the sustained (3 consecutive) column "
    "and the detection curve in the JSON.",
    "String topics (supervisor debug, /path_number) carry no header; bag receive time is used. It agrees with the "
    "status header stamp to < 1 ms (median), max 35 ms.",
    "M2 has only 6 aborts with lateral departure. The full swing into the target lane (peak 3.7-3.9 m) is seen in "
    "the three 09-10 cases; the 09-17 case peaks at 1.0 m and the 09-28 case (abort at 53% progress) ends stopped "
    "straddling the lanes for the rest of the bag. waypoint_system (which re-blends the path on the abort "
    "/path_number) is outside this repo and may differ between these dates.",
    "M4 onset = foot (a <= 0.1) of a centred 0.55 s Savitzky-Golay derivative of GT speed; for a sharp step this "
    "foot can lead the true step by up to ~0.2 s, so delays from onset are slight upper bounds. The filter's own lag "
    "is the 'lag behind the true crossing' number.",
    "Object_topic-mode rate assumes target_selector keeps up with 49 Hz (subscriber queue 1; brute-force projection "
    "of ~35 NPCs onto all lane CSVs per message) - not measured here.",
]


if __name__ == "__main__":
    main()
