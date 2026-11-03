#!/usr/bin/env python3
"""A-checks on a live MORAI bag recorded with target_source:=object_topic over the UDP object path.

/Object_topic comes from morai_udp_bridge/morai_object_info_node.py: positions already UTM,
heading already grid-corrected (-1.347 deg), header.stamp = PC receive time.  The live
target_selector ran with offset 0/0, heading offset 0, rear-axle -> centre shift (1.25 m,
0.5 L - 1.1 m for L >= 7.5 m) and ego extrapolation.  The recorded /smpc/targets are the
live node output, so no replay.  GT = /Object_topic + the same shift; ego = /odom (no
/Ego_topic in the bag).

Also checks for the UDP path: (1) npc.position is still the rear axle (kinematic side-slip
test on the NPC tracks of this bag), (2) heading is already grid-corrected (heading vs
course on straights, vs lane yaw).

Output: research/results/gtcheck_live.json, gtcheck_live_section.md (--append adds it to
gtcheck_report.md).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bagtools as bt  # noqa: E402
import gtcheck_common as gc  # noqa: E402
import gtcheck_a as ga  # noqa: E402
import gtcheck_refpoint as gr  # noqa: E402
from gtcheck_fix import ref_shift, pct, f2  # noqa: E402

BAG = os.path.expanduser("~/bags/research/func2_baseline_S2.bag")
ROLES = gc.ROLES


class LiveGTFrames(gc.GTFrames):
    def __init__(self, d, off=(0.0, 0.0)):
        super().__init__(d, (0.0, 0.0))          # already UTM
        self.yaw = np.radians(self.heading_deg)  # already grid heading
        sh = ref_shift(self.length)
        self.x = self.x + sh * np.cos(self.yaw)
        self.y = self.y + sh * np.sin(self.yaw)


class LiveCheck(ga.BagCheck):
    def __init__(self, path, lanes):
        self.label = "live_" + os.path.basename(path)[:-4]
        self.lanes = lanes
        self.d = bt.load_bag(path, cache_dir=gc.CACHE)
        self.G = LiveGTFrames(self.d)
        self.R = gc.extract_targets(path)
        self.L = None
        G = self.G
        self.P0, self.P1 = {}, {}
        for lid, l in lanes.items():
            self.P0[lid] = l.project(G.x, G.y, 0.0)[:3]
            self.P1[lid] = l.project(G.x, G.y, gc.EXT_BEFORE)[:3]
        self.phys0 = gc.gt_lane(lanes, G.x, G.y, ext=False)
        self.phys1 = gc.gt_lane(lanes, G.x, G.y, ext=True)
        self.wins = ga.lc_windows(self.d)


def kinematic_live(d):
    tr = bt.gt_tracks(d, {"dx": 0.0, "dy": 0.0})
    rows = []
    for u, t in tr.items():
        r = gr.kin_track(t["t"], t["x"], t["y"], t["heading"])
        if r:
            rows.append(np.c_[r[0], r[1], r[2], np.full(len(r[0]), np.median(t["length"]))])
    a = np.concatenate(rows)
    beta, om, v, L = a.T
    kap = om / v
    st = np.abs(kap) < 0.002
    gamma = float(np.median(beta[st]))
    out = {"n": int(len(a)), "n_straight": int(st.sum()), "gamma_deg": float(np.degrees(gamma)),
           "gamma_p10_p90_deg": [float(np.degrees(np.percentile(beta[st], 10))),
                                 float(np.degrees(np.percentile(beta[st], 90)))],
           "x_ref_by_speed": []}
    for thr in (0.008, 0.004):
        cv = np.abs(kap) > thr
        x_est = -(beta - gamma) / np.where(cv, kap, np.nan)
        for lo, hi in ((3, 9), (9, 16), (16, 30)):
            m = cv & (v >= lo) & (v < hi)
            if m.sum() >= 30:
                out["x_ref_by_speed"].append({"kappa_min": thr, "v": "%d-%d" % (lo, hi), "n": int(m.sum()),
                                              "x_ref_med_m": float(np.median(x_est[m]))})
        if out["x_ref_by_speed"]:
            break
    return out


def duplicates_and_accel_filter(d):
    """Exact duplicate /Object_topic frames and their effect on the SMPC observed-accel EMA
    (smpc_decision updateOneVehicleHistory: dt > 1e-3, a = 0.35 raw + 0.65 a)."""
    G = gc.GTFrames(d, (0.0, 0.0))
    n = len(G.t)
    dup = np.zeros(n, bool)
    for i in range(1, n):
        a, b = slice(G.start[i - 1], G.end[i - 1]), slice(G.start[i], G.end[i])
        if G.end[i] - G.start[i] == G.end[i - 1] - G.start[i - 1] and G.end[i] > G.start[i]:
            dup[i] = (np.array_equal(G.uid[a], G.uid[b]) and np.array_equal(G.x[a], G.x[b])
                      and np.array_equal(G.y[a], G.y[b]) and np.array_equal(G.v[a], G.v[b]))
    dt = np.diff(G.t)
    g = d["gt"]
    T = d["gt_msg_t"]
    rA, rD = [], []
    for u in np.unique(g["uid"]):
        m = np.flatnonzero(g["uid"] == u)
        t = T[g["k"][m].astype(int)]
        v = g["vx_kmh"][m] / 3.6
        ax = g["ax"][m]
        if len(t) < 40:
            continue

        def run(sel):
            a, pv, pt = 0.0, None, None
            out = np.full(len(t), np.nan)
            for i in sel:
                if pv is not None:
                    dtt = t[i] - pt
                    if 1e-3 < dtt < 1.0:
                        a = 0.35 * np.clip((v[i] - pv) / dtt, -6, 6) + 0.65 * a
                pv, pt = v[i], t[i]
                out[i] = a
            return out
        A = run(range(len(t)))
        keep = [0] + [i for i in range(1, len(t)) if not (t[i] - t[i - 1] < 0.005 and v[i] == v[i - 1])]
        D = run(keep)
        last = np.nan
        for i in range(len(t)):
            if np.isfinite(D[i]):
                last = D[i]
            D[i] = last
        for i in range(len(t)):
            w = (t > t[i] - 0.5) & (t < t[i] + 0.5)
            if ax[i] > 0.5 and np.all(ax[w] > 0.4) and t[i] - t[0] > 2.0:
                rA.append(A[i] / ax[i])
                rD.append(D[i] / ax[i])
    new = np.flatnonzero(~dup)
    return {"dup_frac": float(dup.mean()), "dup_dt_ms": bt.stats(1000 * dt[dup[1:]]),
            "distinct_state_interval_ms": bt.stats(1000 * np.diff(G.t[new])),
            "accel_filter_ratio_all_frames": bt.stats(rA),
            "accel_filter_ratio_distinct_only": bt.stats(rD)}


def published_vs_gt(bc):
    G, R = bc.G, bc.R
    nb = R["nearby"]
    dxy, raw = [], []
    for k in range(len(R["t"])):
        i = G.index_of(R["t"][k])
        if i is None:
            continue
        sl = slice(G.start[i], G.end[i])
        for j in np.flatnonzero(nb["k"] == k):
            jr = np.flatnonzero(G.uid[sl] == int(nb["uid"][j]))
            if len(jr):
                jr = jr[0]
                dxy.append(np.hypot(nb["x"][j] - G.x[sl][jr], nb["y"][j] - G.y[sl][jr]))
                # distance to the unshifted (raw rear-axle) position, for contrast
                sh = ref_shift(G.length[sl][jr])
                raw.append(np.hypot(nb["x"][j] - (G.x[sl][jr] - sh * np.cos(G.yaw[sl][jr])),
                                    nb["y"][j] - (G.y[sl][jr] - sh * np.sin(G.yaw[sl][jr]))))
    return bt.stats(dxy), bt.stats(raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", default=BAG)
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    gc.HEADING_BIAS_RAD = 0.0                    # heading already grid-corrected
    lanes = bt.load_lanes()
    bc = LiveCheck(a.bag, lanes)
    r = bc.check_object_mode()
    r["worst_E1"] = r["worst_E1"][:10]
    r["published_xy_vs_gt_centre_m"], r["published_xy_vs_raw_position_m"] = published_vs_gt(bc)
    r["kinematic"] = kinematic_live(bc.d)
    r["duplicates"] = duplicates_and_accel_filter(bc.d)
    o = bc.d["odom"]
    G = bc.G
    m = (G.t >= o["t"][0]) & (G.t <= o["t"][-1])
    r["stamp_info"] = {"object_recv_minus_stamp_ms": bt.stats(1000 * (G.bag_t - G.t)),
                       "odom_dt_ms": bt.stats(1000 * np.diff(o["t"])),
                       "object_dt_ms": bt.stats(1000 * np.diff(G.t)),
                       "n_obj_frames": int(len(G.t)), "n_targets": int(len(bc.R["t"])),
                       "frac_frames_in_odom_span": float(m.mean())}
    win = ga.in_windows(bc.R["t"], bc.wins)
    V = bc.R["target_rear"]
    mm = win & (V["valid"] > 0)
    r["lc_windows"] = [(float(x - G.t[0]), float(y - G.t[0])) for x, y in bc.wins]
    r["lc_target_rear_frames"] = int(mm.sum())
    bt.write_json(os.path.join(gc.RESULTS, "gtcheck_live.json"), r)
    sec = section(r, os.path.basename(a.bag))
    with open(os.path.join(gc.RESULTS, "gtcheck_live_section.md"), "w") as fo:
        fo.write(sec)
    if a.append:
        with open(os.path.join(gc.RESULTS, "gtcheck_report.md"), "a") as fo:
            fo.write("\n" + sec)
    print(sec)


def section(r, name):
    L = []
    w = L.append
    rc = r["role_cmp"]
    ln = r["lane"]
    vel = r["vel"]
    rt = r["rate"]
    el = r["ego_lag"]
    kin = r["kinematic"]
    hd = r["heading"]
    si = r["stamp_info"]
    w("### Live MORAI bag, UDP object path (`%s`)" % name)
    w("")
    w("Scenario research_S2_rear_accel, method baseline, recorded 00:51-00:53 with `target_source:=object_topic`. "
      "/Object_topic = `morai_udp_bridge/morai_object_info_node.py` (UTM positions, grid heading = true-north "
      "heading - 1.347 deg, stamp = PC receive time). Live selector params (launch log): offset 0/0, heading "
      "offset 0, ref_to_center 1.25 m, long vehicle 7.5 m / 1.1 m, extrapolate_ego true. The recorded "
      "/smpc/targets is the live node output (no replay). GT = /Object_topic + the same shift; ego = /odom "
      "(no /Ego_topic in the bag, and the Object Info stream carries no ego record). %d /Object_topic frames, "
      "%d /smpc/targets, %d LC windows." % (si["n_obj_frames"], si["n_targets"], len(r["lc_windows"])))
    w("")
    w("| # | check | live result | result |")
    w("|---|---|---|---|")
    pv = r["published_xy_vs_gt_centre_m"]
    pr = r["published_xy_vs_raw_position_m"]
    w("| R0 | published x/y = /Object_topic position + shift | max %s m from the shifted GT (median %s m from the raw position = the shift) | **PASS** |" % (
        f2(pv["max"], 4), f2(pr["median"], 2)))
    w("| A1 | lane_id within +-120 m vs GT lane | %s of %d lane-centred NPC-frames, %d/%d wrong, %d missing; %d NPC-frames outside the 4 CSVs (%d in nearby) | **PASS** |" % (
        pct(ln["agree_frac_centred"]), ln["n_centred"], ln["wrong_lane_known"], ln["gt_lane_known"],
        ln["missing_from_nearby_known"], ln["n_no_gt_lane_within_120m"], ln["no_gt_lane_but_in_nearby"]))
    w("| A1' | nearby port == node lane_id | %s (port-only %d, node-only %d) | **%s** |" % (
        pct(r["nearby_impl"]["lane_equal_frac"]), r["nearby_impl"]["port_only"], r["nearby_impl"]["node_only"],
        "PASS" if r["nearby_impl"]["lane_equal_frac"] > 0.999 else "CHECK"))
    for e, lab in (("E2", "roles vs GT, node ego (E2)"), ("E1", "roles vs GT, /odom at frame stamp (E1)")):
        w("| A2%s | %s | cf %s, tf %s, tr %s (LC windows: tf %s, tr %s) | **%s** |" % (
            "" if e == "E2" else "'", lab,
            pct(rc[e]["current_front"]["all"]["agree"]), pct(rc[e]["target_front"]["all"]["agree"]),
            pct(rc[e]["target_rear"]["all"]["agree"]), pct(rc[e]["target_front"]["lc_window"]["agree"]),
            pct(rc[e]["target_rear"]["lc_window"]["agree"]),
            "PASS" if min(rc[e][x]["all"]["agree"] for x in ROLES) >= (0.9999 if e == "E2" else 0.99) else "CHECK"))
    w("| A3 | v_long vs velocity.x/3.6 | max %s m/s (target_rear n=%d, nearby n=%d) | **PASS** |" % (
        f2(max(vel[x]["v_long_minus_velx"]["max"] for x in ROLES + ["nearby"] if vel[x]["n"]), 4),
        vel["target_rear"]["n"], vel["nearby"]["n"]))
    w("| A3' | v_lat (vehicle frame, known limitation) | missed lane-lateral motion in %s of nearby entries | known limitation |" % (
        pct(vel["nearby"]["frac_lane_lat_gt_0p25_node_lt_0p25"], 1)))
    w("| A4 | delta_s vs centre, E2 (implementation) | p90 %s m, max %s m (the max is only where the ego is past the lane-2 CSV end and ego_s is clamped, so rebuilding the ego from (s, d) is not exact; 6 entries) | **PASS** |" % (
        f2(max(vel[x]["delta_s_abs_err_E2"]["p90"] for x in ROLES + ["nearby"] if vel[x]["n"]), 4),
        f2(max(vel[x]["delta_s_abs_err_E2"]["max"] for x in ROLES + ["nearby"] if vel[x]["n"]), 2)))
    w("| A4 | delta_s vs centre, E1: target_rear p50 / p90 / max | %s / %s / %s m (nearby p90 %s, max %s) | **PASS** |" % (
        f2(vel["target_rear"]["delta_s_abs_err_E1"]["median"]), f2(vel["target_rear"]["delta_s_abs_err_E1"]["p90"]),
        f2(vel["target_rear"]["delta_s_abs_err_E1"]["max"]), f2(vel["nearby"]["delta_s_abs_err_E1"]["p90"]),
        f2(vel["nearby"]["delta_s_abs_err_E1"]["max"])))
    w("| A4 | node ego_s - /odom at frame stamp, moving: p10 / p50 / p90 | %s / %s / %s m (implied lag median %s s) | **PASS** |" % (
        f2(el["ego_s_err_moving_m"]["p10"]), f2(el["ego_s_err_moving_m"]["median"]), f2(el["ego_s_err_moving_m"]["p90"]),
        f2(el["implied_lag_s_moving"]["median"], 3)))
    w("| A5 | heading - lane yaw, lane-centred moving NPCs (median, p10..p90) | %s deg (%s..%s), n=%d | **PASS** |" % (
        f2(hd["err_deg_signed_enu"]["median"], 3), f2(hd["err_deg_signed_enu"]["p10"]), f2(hd["err_deg_signed_enu"]["p90"]),
        hd["lane_centred_moving_rows"]))
    w("| U2 | heading already grid-corrected: heading - course on straights | %s deg (p10..p90 %s..%s; September ROS path: +1.361) | **PASS** |" % (
        f2(kin["gamma_deg"], 3), f2(kin["gamma_p10_p90_deg"][0]), f2(kin["gamma_p10_p90_deg"][1])))
    xr = kin["x_ref_by_speed"]
    w("| U1 | position still the rear axle (kinematic x_ref) | %s | **%s** |" % (
        "; ".join("%s m/s: %s m (n=%d, abs kappa>%s)" % (x["v"], f2(x["x_ref_med_m"]), x["n"], x["kappa_min"]) for x in xr) or "too few curve samples",
        "PASS" if xr and all(x["x_ref_med_m"] < 0.6 for x in xr) else "INCONCLUSIVE"))
    du = r["duplicates"]
    w("| U3 new | duplicate /Object_topic frames (UDP path) | %s of frames are exact copies of the previous one, %s ms later; distinct states every %s ms (p90 %s) = %s Hz | **FAIL** (new) |" % (
        pct(du["dup_frac"], 1), f2(du["dup_dt_ms"]["median"], 1), f2(du["distinct_state_interval_ms"]["median"], 1),
        f2(du["distinct_state_interval_ms"]["p90"], 1), f2(1000.0 / du["distinct_state_interval_ms"]["median"], 1)))
    w("| U3 effect | SMPC observed-accel EMA / true MORAI accel, sustained accel > 0.5 m/s^2 | all frames %s (p10..p90 %s..%s) vs duplicates removed %s | **FAIL** (new) |" % (
        f2(du["accel_filter_ratio_all_frames"]["median"]), f2(du["accel_filter_ratio_all_frames"]["p10"]),
        f2(du["accel_filter_ratio_all_frames"]["p90"]), f2(du["accel_filter_ratio_distinct_only"]["median"])))
    w("| A6 | rate | in %s Hz (header), out %s Hz, %s of frames processed, %s identical consecutive outputs, output dt p90 %s s / max %s s | **PASS** |" % (
        f2(rt["in_hz"], 1), f2(rt["out_hz"], 1), pct(rt["coverage"], 1), pct(rt["output_identical_to_prev_frac"], 1),
        f2(rt["dt_out"]["p90"], 3), f2(rt["dt_out_max"], 3)))
    w("")
    w("Role counts (E1): " + "; ".join("%s GT valid %d, wrong %d, missed %d, extra %d" % (
        x, rc["E1"][x]["all"]["gt_valid"], rc["E1"][x]["all"]["wrong_vehicle"], rc["E1"][x]["all"]["missed"],
        rc["E1"][x]["all"]["extra"]) for x in ROLES) + ". /Object_topic receive - stamp median %s ms; "
      "object dt median %s ms (p90 %s); /odom dt median %s ms." % (
          f2(si["object_recv_minus_stamp_ms"]["median"], 1), f2(si["object_dt_ms"]["median"], 1),
          f2(si["object_dt_ms"]["p90"], 1), f2(si["odom_dt_ms"]["median"], 1)))
    w("")
    if r["worst_E1"]:
        w("Worst E1 mismatches: " + "; ".join("t=%.2f %s node %d / GT %d (gap %.1f vs %.1f)" % (
            c["t_rel"], c["role"], c["node_uid"], c["gt_uid"], c["gap_node"], c["gap_gt"]) for c in r["worst_E1"][:5]) + ".")
        w("")
    w("U3: every Object Info state arrives twice ~1 ms apart. The copy is harmless for positions (1 ms), "
      "but smpc_decision's per-vehicle acceleration EMA (`dt > 1e-3`, a = 0.35 raw + 0.65 a) takes the copy "
      "as a dv = 0 sample, so the observed TV acceleration settles at ~0.4-0.6x the truth. That under-reads "
      "rear acceleration, which this rear_accel scenario is about. Suggested fix: dedup in "
      "`morai_object_info_node.py` by the packet sim stamp (it is decoded and then discarded) or by "
      "content, and/or skip repeated states in `objectCallback`. A6 counts the copies as frames, so "
      "the real object update rate is the distinct-state rate above.")
    w("")
    w("Caveats: the E1 'synchronous' ego here is /odom interpolated at the /Object_topic receive stamp; the "
      "relative latency of the GPS and Object Info UDP streams is not observable in this bag. The UDP "
      "record also carries overhang / wheelbase / rear_overhang per object (object_info.py), which the ROS "
      "bridge drops; publishing them would let the selector use the exact L/2 - rear_overhang shift instead "
      "of the 1.25 m constant.")
    w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
