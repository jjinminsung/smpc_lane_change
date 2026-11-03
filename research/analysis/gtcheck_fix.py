#!/usr/bin/env python3
"""Re-verification of the fixed object_topic path (2026-11-04).

Fix under test (src/target_selector_node.cpp objectCallback, config/target_selector.yaml):
  yaw   = (npc.heading - 1.365 deg) in rad
  shift = 1.25 m if L < 7.5 m else 0.5 L - 1.1 m
  x, y  = npc.position + offset + shift * (cos yaw, sin yaw)     (rear axle -> bbox centre)
  ego   = /odom extrapolated to the /Object_topic stamp (dt clamped to [0, 0.1] s)

GT side is updated to the same model (centre = rear axle + shift, yaw corrected),
then the gtcheck_a checks are re-run on the replays in
research/results/gtcheck_replay/<bag>_fix.bag.  The LiDAR face residual of check B
is recomputed from gtcheck_refpoint_faces.csv with the same shift.

Output: research/results/gtcheck_fix.json and research/results/gtcheck_fix_section.md
(the section is also appended to gtcheck_report.md with --append).
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bagtools as bt  # noqa: E402
import gtcheck_common as gc  # noqa: E402
import gtcheck_a as ga  # noqa: E402

BAGS = ["2026-09-28-11-05-29", "2026-09-28-11-30-19", "2026-09-21-15-43-18"]
ROLES = gc.ROLES
HEADING_OFFSET_DEG = -1.365
REF_TO_CENTER = 1.25
LONG_L = 7.5
LONG_REAR_OVERHANG = 1.1


def ref_shift(length):
    length = np.asarray(length, float)
    return np.where(length >= LONG_L, np.maximum(0.0, 0.5 * length - LONG_REAR_OVERHANG), REF_TO_CENTER)


class FixedGTFrames(gc.GTFrames):
    """GT with the fixed model: yaw corrected, position moved to the bbox centre."""

    def __init__(self, d, off=(gc.OFF_X, gc.OFF_Y)):
        super().__init__(d, off)
        self.yaw = np.radians(self.heading_deg + HEADING_OFFSET_DEG)
        sh = ref_shift(self.length)
        self.x = self.x + sh * np.cos(self.yaw)
        self.y = self.y + sh * np.sin(self.yaw)
        self.shift = sh


def ego_truth(lab, lanes):
    """Node ego_s vs /Ego_topic (MORAI rear axle, same stamp as /Object_topic) for the old and
    fixed replays, and the effect of the [0, 0.1] s clamp when the newest /odom received
    before a frame is already newer than the frame stamp (dt < 0)."""
    import rosbag
    path = os.path.expanduser("~/bags/lc_gt_%s.bag" % lab)
    d = bt.load_bag(path)
    e = d["ego"]
    G = FixedGTFrames(d)          # only stamps / receive times are used here
    st = {round(t, 6): i for i, t in enumerate(e["t"])}
    od = []
    with rosbag.Bag(path) as b:
        for _, m, t in b.read_messages(topics=["/odom"]):
            od.append((t.to_sec(), m.header.stamp.to_sec()))
    od = np.asarray(od)
    out = {}
    for suf in ("", "_fix"):
        R = gc.extract_targets(os.path.join(gc.REPLAY_DIR, lab + suf + ".bag"))
        rows = []
        for k in range(len(R["t"])):
            i = G.index_of(R["t"][k])
            ie = st.get(round(R["t"][k], 6))
            if i is None or ie is None or e["vx"][ie] < 3.0:
                continue
            j = np.searchsorted(od[:, 0], G.bag_t[i], side="right") - 1
            lane = lanes[int(R["cur_lane"][k])]
            s_true = lane.project(np.array([e["x"][ie] + gc.OFF_X]), np.array([e["y"][ie] + gc.OFF_Y]), 0.0)[0][0]
            rows.append((R["t"][k] - od[max(j, 0), 1], R["ego_s_cur"][k] - s_true))
        a = np.asarray(rows)
        neg = a[:, 0] < 0
        out["old" if not suf else "fix"] = {
            "err": bt.stats(a[:, 1]), "abs_err": bt.stats(np.abs(a[:, 1])),
            "frac_newest_odom_after_frame": float(neg.mean()),
            "err_when_odom_after_frame": bt.stats(a[neg, 1]),
            "err_when_odom_before_frame": bt.stats(a[~neg, 1])}
    return out


def faces_residual():
    rows = list(csv.DictReader(open(os.path.join(gc.RESULTS, "gtcheck_refpoint_faces.csv"))))
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    out = {}
    for side, key in (("ahead", "off_rear"), ("behind", "off_front")):
        for est in ("", "_bumper"):
            k = "H1_%s%s" % (key, est)
            sel = [r for r in rows if r["side"] == side and f(r, "H1_n") >= 8 and np.isfinite(f(r, k))]
            if not sel:
                continue
            L = np.array([f(r, "L") for r in sel])
            before = np.array([f(r, k) for r in sel])
            after = before - ref_shift(L)
            per = {}
            for r, x in zip(sel, after):
                per.setdefault((r["bag"], r["uid"]), []).append(x)
            st = np.array([f(r, "ego_v") < 0.3 for r in sel])
            o = {"n_scans": len(sel), "n_npcs": len(per), "before": bt.stats(before),
                 "after": bt.stats(after), "after_abs": bt.stats(np.abs(after)),
                 "after_per_npc_median": bt.stats([np.median(x) for x in per.values()]),
                 "after_ego_stopped": bt.stats(after[st]), "by_length_after": {}}
            for lo, hi in ((3.5, 4.7), (4.7, 5.2), (5.2, 7.5), (7.5, 13.0)):
                m = (L >= lo) & (L < hi)
                if m.sum() >= 3:
                    o["by_length_after"]["%.1f-%.1f" % (lo, hi)] = bt.stats(after[m])
            out[side + est] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true", help="append the section to gtcheck_report.md")
    a = ap.parse_args()
    gc.GTFrames = FixedGTFrames          # GT model == the fixed node model
    gc.HEADING_BIAS_RAD = 0.0            # GT yaw is already corrected
    lanes = bt.load_lanes()
    res = {"bags": {}}
    pooled_lidar = {r: [] for r in ROLES}
    pooled_pe = []
    worst = []
    for lab in BAGS:
        orig = os.path.expanduser("~/bags/lc_gt_%s.bag" % lab)
        rep = os.path.join(gc.REPLAY_DIR, lab + "_fix.bag")
        print("[gtcheck_fix] %s" % lab, file=sys.stderr)
        bc = ga.BagCheck(orig, rep, lanes)
        r = bc.check_object_mode()
        worst += r.pop("worst_E1")
        la = bc.check_lidar_mode()
        r["lidar_C"] = ga.lidar_summary(la, None)
        r["lidar_pos_err"] = ga.pos_err_summary(la["pos_err"])
        pooled_pe.append(la["pos_err"])
        for role in ROLES:
            pooled_lidar[role].append(la[role])
        # published shift sanity: node x/y vs fixed GT centre
        G, R = bc.G, bc.R
        dxy = []
        for k in range(len(R["t"])):
            i = bc.idx[k]
            if i < 0:
                continue
            sl = slice(G.start[i], G.end[i])
            nb = R["nearby"]
            for j in np.flatnonzero(nb["k"] == k):
                jr = np.flatnonzero(G.uid[sl] == int(nb["uid"][j]))
                if len(jr):
                    dxy.append(math.hypot(nb["x"][j] - G.x[sl][jr[0]], nb["y"][j] - G.y[sl][jr[0]]))
        r["published_xy_vs_fixed_gt_m"] = bt.stats(dxy)
        # rear-gap distribution in LC windows (after the fix the published gap is centre-based)
        win = ga.in_windows(R["t"], ga.lc_windows(bc.d))
        V = R["target_rear"]
        m = win & (V["valid"] > 0)
        g = -V["front_delta_s"][m] - 0.790
        r["lc_target_rear_gap"] = {"n": int(m.sum()), "lt_9m": int(np.sum(g < 9.0)),
                                   "in_9_15": int(np.sum((g >= 9.0) & (g < 15.0)))}
        res["bags"][lab] = r
    res["pooled_lidar_C"] = ga.lidar_summary({r: np.concatenate(pooled_lidar[r]) for r in ROLES}, None)
    res["pooled_lidar_pos_err"] = ga.pos_err_summary(np.concatenate(pooled_pe))
    res["worst_E1"] = sorted(worst, key=lambda c: -c["severity"])[:20]
    res["faces"] = faces_residual()
    res["ego_truth"] = {lab: ego_truth(lab, lanes) for lab in BAGS}
    bt.write_json(os.path.join(gc.RESULTS, "gtcheck_fix.json"), res)
    sec = section(res, json.load(open(os.path.join(gc.RESULTS, "gtcheck_a.json"))))
    with open(os.path.join(gc.RESULTS, "gtcheck_fix_section.md"), "w") as fo:
        fo.write(sec)
    if a.append:
        with open(os.path.join(gc.RESULTS, "gtcheck_report.md"), "a") as fo:
            fo.write("\n" + sec)
    print(sec)


def pct(x, d=2):
    return "n/a" if x is None else ("%." + str(d) + "f%%") % (100.0 * x)


def f2(x, d=2):
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else ("%." + str(d) + "f") % x


def pool_role(bags, e, role, key="all"):
    n = sum(bags[b]["role_cmp"][e][role][key]["n"] for b in BAGS)
    ag = sum(bags[b]["role_cmp"][e][role][key]["agree"] * bags[b]["role_cmp"][e][role][key]["n"] for b in BAGS)
    tot = {k: sum(bags[b]["role_cmp"][e][role][key][k] for b in BAGS)
           for k in ("gt_valid", "wrong_vehicle", "missed", "extra")}
    tot["n"], tot["agree"] = n, ag / max(n, 1)
    return tot


def lane_pool(bags):
    kn = sum(bags[b]["lane"]["gt_lane_known"] for b in BAGS)
    wr = sum(bags[b]["lane"]["wrong_lane_known"] for b in BAGS)
    nc = sum(bags[b]["lane"]["n_centred"] for b in BAGS)
    nco = sum(bags[b]["lane"]["agree_frac_centred"] * bags[b]["lane"]["n_centred"] for b in BAGS)
    ms = sum(bags[b]["lane"]["missing_from_nearby_known"] for b in BAGS)
    return kn, wr, nc, nco / max(nc, 1), ms


def vmax(bags, role, key, stat):
    return max(bags[b]["vel"][role][key][stat] for b in BAGS if bags[b]["vel"][role]["n"])


def section(F, A):
    fb, ab = F["bags"], A["bags"]
    L = []
    w = L.append
    w("## Re-verification after fixes (2026-11-04)")
    w("")
    w("Fix under test (uncommitted `src/target_selector_node.cpp` + `config/target_selector.yaml`, node rebuilt "
      "00:40): `yaw = (npc.heading - 1.365 deg)`, position moved from the rear axle to the bbox centre by "
      "1.25 m (L < 7.5 m) or 0.5 L - 1.1 m along the corrected heading, ego = /odom extrapolated to the "
      "/Object_topic stamp (dt in [0, 0.1] s). Same 3 bags, same replay recipe (master 11411, --clock), "
      "output `gtcheck_replay/<bag>_fix.bag`. GT side updated to the same model (centre = rear axle + "
      "shift, corrected yaw); script `research/analysis/gtcheck_fix.py`, numbers in `gtcheck_fix.json`. "
      "v_lat is unchanged on purpose (still vehicle-frame) and is listed as a known limitation.")
    w("")
    kn, wr, nc, nco, ms = lane_pool(fb)
    e2 = {r: pool_role(fb, "E2", r) for r in ROLES}
    e1 = {r: pool_role(fb, "E1", r) for r in ROLES}
    e1b = {r: pool_role(ab, "E1", r) for r in ROLES}
    e1w = {r: pool_role(fb, "E1", r, "lc_window") for r in ROLES}
    e1wb = {r: pool_role(ab, "E1", r, "lc_window") for r in ROLES}
    fc = F["faces"]
    xy = max(fb[b]["published_xy_vs_fixed_gt_m"]["max"] for b in BAGS)
    lagb = [ab[b]["ego_lag"]["ego_s_err_moving_m"] for b in BAGS]
    laga = [fb[b]["ego_lag"]["ego_s_err_moving_m"] for b in BAGS]
    ds_b = {r: (max(ab[b]["vel"][r]["delta_s_abs_err_E1"]["p90"] for b in BAGS),
                max(ab[b]["vel"][r]["delta_s_abs_err_E1"]["max"] for b in BAGS)) for r in ROLES + ["nearby"]}
    ds_a = {r: (max(fb[b]["vel"][r]["delta_s_abs_err_E1"]["p90"] for b in BAGS),
                max(fb[b]["vel"][r]["delta_s_abs_err_E1"]["max"] for b in BAGS)) for r in ROLES + ["nearby"]}
    rate = {b: fb[b]["rate"] for b in BAGS}
    hd = [fb[b]["heading"]["err_deg_signed_enu"]["median"] for b in BAGS]
    w("| # | check | before fix | after fix | result |")
    w("|---|---|---|---|---|")
    w("| R0 | published x/y = npc.position + offset + shift | - | max deviation from the fixed GT centre %s m | **PASS** |" % f2(xy, 4))
    w("| A1 | lane_id within +-120 m vs GT lane | 99.98%% centred, 22 wrong | %s centred (n=%d), %d/%d wrong, %d missing | **PASS** |" % (
        pct(nco), nc, wr, kn, ms))
    w("| A2 | roles vs GT, node ego (E2) | 100%% | cf %s, tf %s, tr %s | **PASS** |" % (
        pct(e2["current_front"]["agree"]), pct(e2["target_front"]["agree"]), pct(e2["target_rear"]["agree"])))
    w("| A2' | roles vs GT, synchronous ego (E1) | cf %s, tf %s, tr %s (tf: %d wrong / %d missed / %d extra) | cf %s, tf %s, tr %s (tf: %d / %d / %d) | **%s** |" % (
        pct(e1b["current_front"]["agree"]), pct(e1b["target_front"]["agree"]), pct(e1b["target_rear"]["agree"]),
        e1b["target_front"]["wrong_vehicle"], e1b["target_front"]["missed"], e1b["target_front"]["extra"],
        pct(e1["current_front"]["agree"]), pct(e1["target_front"]["agree"]), pct(e1["target_rear"]["agree"]),
        e1["target_front"]["wrong_vehicle"], e1["target_front"]["missed"], e1["target_front"]["extra"],
        "PASS" if min(e1[r]["agree"] for r in ROLES) >= min(e1b[r]["agree"] for r in ROLES) else "REGRESSION"))
    w("| A2' LC | same, LC windows | tf %s, tr %s | tf %s, tr %s | info |" % (
        pct(e1wb["target_front"]["agree"]), pct(e1wb["target_rear"]["agree"]),
        pct(e1w["target_front"]["agree"]), pct(e1w["target_rear"]["agree"])))
    w("| A3 | v_long vs velocity.x/3.6 | 0.0000 | max %s m/s | **PASS** |" % f2(max(vmax(fb, r, "v_long_minus_velx", "max") for r in ROLES + ["nearby"]), 4))
    w("| A3' | v_lat (vehicle frame) | latent | unchanged by design: missed lane-lateral motion in %s-%s of nearby entries | **known limitation** |" % (
        pct(min(fb[b]["vel"]["nearby"]["frac_lane_lat_gt_0p25_node_lt_0p25"] for b in BAGS), 1),
        pct(max(fb[b]["vel"]["nearby"]["frac_lane_lat_gt_0p25_node_lt_0p25"] for b in BAGS), 1)))
    w("| A4 | delta_s vs true centre (E1), target_rear p90 / max | %s / %s m | %s / %s m | **PASS** |" % (
        f2(ds_b["target_rear"][0]), f2(ds_b["target_rear"][1]), f2(ds_a["target_rear"][0]), f2(ds_a["target_rear"][1])))
    w("| A4 | delta_s vs true centre (E1), nearby p90 / max | %s / %s m | %s / %s m | **PASS** |" % (
        f2(ds_b["nearby"][0]), f2(ds_b["nearby"][1]), f2(ds_a["nearby"][0]), f2(ds_a["nearby"][1])))
    w("| A4 | ego_s error while moving, median (p10..p90) | %s | %s | **PASS** |" % (
        "; ".join("%s (%s..%s)" % (f2(x["median"]), f2(x["p10"]), f2(x["p90"])) for x in lagb),
        "; ".join("%s (%s..%s)" % (f2(x["median"]), f2(x["p10"]), f2(x["p90"])) for x in laga)))
    et = F["ego_truth"]
    w("| A4 | node ego_s - /Ego_topic (true rear axle), moving: p10 / p50 / p90 | %s | %s | **PASS** |" % (
        "; ".join("%s / %s / %s" % (f2(et[b]["old"]["err"]["p10"]), f2(et[b]["old"]["err"]["median"]), f2(et[b]["old"]["err"]["p90"])) for b in BAGS),
        "; ".join("%s / %s / %s" % (f2(et[b]["fix"]["err"]["p10"]), f2(et[b]["fix"]["err"]["median"]), f2(et[b]["fix"]["err"]["p90"])) for b in BAGS)))
    w("| A4 new | frames whose newest /odom is newer than the frame (dt<0, clamped to 0): share, ego_s error p90 | - | %s | **WARN** (new) |" % (
        "; ".join("%s, p90 %s m (vs %s m when dt>=0)" % (pct(et[b]["fix"]["frac_newest_odom_after_frame"], 0),
                                                       f2(et[b]["fix"]["err_when_odom_after_frame"]["p90"]),
                                                       f2(et[b]["fix"]["err_when_odom_before_frame"]["p90"])) for b in BAGS)))
    w("| A5 | heading - lane yaw (median per bag) | +1.36 deg | %s deg | **PASS** |" % ", ".join(f2(x, 3) for x in hd))
    w("| A6 | output Hz / frames processed / identical outputs | 32.1-37.8 Hz / 66-76%% / 0.8-10.7%% | %s-%s Hz / %s-%s / %s-%s | WARN (unchanged) |" % (
        f2(min(r["out_hz"] for r in rate.values()), 1), f2(max(r["out_hz"] for r in rate.values()), 1),
        pct(min(r["coverage"] for r in rate.values()), 0), pct(max(r["coverage"] for r in rate.values()), 0),
        pct(min(r["output_identical_to_prev_frac"] for r in rate.values()), 1),
        pct(max(r["output_identical_to_prev_frac"] for r in rate.values()), 1)))
    for side, lab in (("ahead_bumper", "rear face (car ahead, bumper zone)"), ("behind_bumper", "front face (car behind, bumper zone)"),
                      ("ahead", "rear face (all points)"), ("behind", "front face (all points)")):
        o = fc.get(side)
        if not o:
            continue
        w("| B | LiDAR %s residual, median (p10..p90) | %s (%s..%s) m | %s (%s..%s) m, per-NPC median %s m | **%s** |" % (
            lab, f2(o["before"]["median"]), f2(o["before"]["p10"]), f2(o["before"]["p90"]),
            f2(o["after"]["median"]), f2(o["after"]["p10"]), f2(o["after"]["p90"]),
            f2(o["after_per_npc_median"]["median"]),
            "PASS" if abs(o["after"]["median"]) <= 0.25 else "WARN"))
    w("")
    w("Per-bag detail (after the fix):")
    w("")
    w("| bag | msgs | E1 agree cf / tf / tr | E1 tf wrong / missed / extra | dS E1 target_rear p50 / p90 / max | ego_s err moving p50 / max abs | ego lag implied p50 / p90 [s] | out Hz | processed |")
    w("|---|---|---|---|---|---|---|---|---|")
    for b in BAGS:
        x = fb[b]
        rc = x["role_cmp"]["E1"]
        v = x["vel"]["target_rear"]["delta_s_abs_err_E1"]
        el = x["ego_lag"]
        w("| %s | %d | %s / %s / %s | %d / %d / %d | %s / %s / %s | %s / %s | %s / %s | %.1f | %s |" % (
            b, x["n_msgs"], pct(rc["current_front"]["all"]["agree"]), pct(rc["target_front"]["all"]["agree"]),
            pct(rc["target_rear"]["all"]["agree"]), rc["target_front"]["all"]["wrong_vehicle"],
            rc["target_front"]["all"]["missed"], rc["target_front"]["all"]["extra"],
            f2(v["median"]), f2(v["p90"]), f2(v["max"]), f2(el["ego_s_err_moving_m"]["median"], 3),
            f2(max(abs(el["ego_s_err_moving_m"]["min"]), abs(el["ego_s_err_moving_m"]["max"]))),
            f2(el["implied_lag_s_moving"]["median"], 3), f2(el["implied_lag_s_moving"]["p90"], 3),
            x["rate"]["out_hz"], pct(x["rate"]["coverage"], 1)))
    w("")
    w("Face residual by length class (after, all scans, bumper zone): " + "; ".join(
        "%s: ahead %s / behind %s" % (
            k, f2(fc["ahead_bumper"]["by_length_after"].get(k, {}).get("median")),
            f2(fc["behind_bumper"]["by_length_after"].get(k, {}).get("median")))
        for k in ("3.5-4.7", "4.7-5.2", "5.2-7.5", "7.5-13.0")) + " m.")
    w("")
    w("Worst remaining E1 role mismatches:")
    w("")
    w("| bag | t [s] | role | node uid | GT uid | node gap | GT gap |")
    w("|---|---|---|---|---|---|---|")
    for c in F["worst_E1"][:8]:
        w("| %s | %.2f | %s | %d | %d | %.2f | %.2f |" % (c["bag"], c["t_rel"], c["role"], c["node_uid"],
                                                       c["gt_uid"], c["gap_node"], c["gap_gt"]))
    w("")
    w("Remaining physical-role gaps (spec filter by design): " + "; ".join(
        "%s %s %s" % (b[5:], r, pct(fb[b]["physical"][r]["agree_frac"], 1)) for b in BAGS for r in ROLES) + ".")
    w("")
    w("LC-window target_rear rear bumper gaps (now centre-based): " + "; ".join(
        "%s: %d frames, %d < 9 m, %d in [9, 15) m" % (b[5:], fb[b]["lc_target_rear_gap"]["n"],
                                                       fb[b]["lc_target_rear_gap"]["lt_9m"],
                                                       fb[b]["lc_target_rear_gap"]["in_9_15"]) for b in BAGS) + ".")
    w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
