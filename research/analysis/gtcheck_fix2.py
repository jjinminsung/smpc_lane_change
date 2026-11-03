#!/usr/bin/env python3
"""Re-verification 2 (2026-11-04): duplicate-frame drop + backward ego extrapolation.

Node changes under test: object_topic_drop_duplicate_frames (skip a frame whose
(unique_id, position.x/y, velocity.x) list equals the previous one) and the ego
extrapolation clamp [-0.1, +0.1] s.  Replays (gtcheck_replay2.py, master 11411):
  * live UDP bag func2_baseline_S2 with the research_morai settings (offsets 0, heading 0)
    -> gtcheck_replay/func2_baseline_S2_fix2.bag
  * the 3 September ROS-path bags with the yaml settings -> gtcheck_replay/<bag>_fix2.bag

Reports (a) duplicates in the new /smpc/targets, (b) smpc_decision-style observed-accel
EMA / true MORAI accel for target_rear, (c) ego error and edge flips, (d) A1/A2/A2'/A4.

Output: research/results/gtcheck_fix2.json, gtcheck_fix2_section.md (--append adds it
to gtcheck_report.md).
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
import gtcheck_fix as gf  # noqa: E402
import gtcheck_live as gl  # noqa: E402
from gtcheck_fix import pct, f2  # noqa: E402

SEPT = ["2026-09-28-11-05-29", "2026-09-28-11-30-19", "2026-09-21-15-43-18"]
ROLES = gc.ROLES
LIVE_REC = gl.BAG
LIVE_FIX2 = os.path.join(gc.REPLAY_DIR, "func2_baseline_S2_fix2.bag")


def dup_flags(G):
    n = len(G.t)
    dup = np.zeros(n, bool)
    for i in range(1, n):
        a, b = slice(G.start[i - 1], G.end[i - 1]), slice(G.start[i], G.end[i])
        if G.end[i] - G.start[i] == G.end[i - 1] - G.start[i - 1] and G.end[i] > G.start[i]:
            dup[i] = (np.array_equal(G.uid[a], G.uid[b]) and np.array_equal(G.x[a], G.x[b])
                      and np.array_equal(G.y[a], G.y[b]) and np.array_equal(G.v[a], G.v[b]))
    return dup


def output_dups(R, G, dup):
    idx = np.array([G.index_of(t) if G.index_of(t) is not None else -1 for t in R["t"]])
    dt = np.diff(R["t"])
    src_dup = np.array([dup[i] if i >= 0 else False for i in idx])
    # an output whose source frame equals the last frame that WAS published
    same_as_prev_pub = np.zeros(len(idx), bool)
    for k in range(1, len(idx)):
        i0, i1 = idx[k - 1], idx[k]
        if i0 < 0 or i1 < 0:
            continue
        a, b = slice(G.start[i0], G.end[i0]), slice(G.start[i1], G.end[i1])
        same_as_prev_pub[k] = (G.end[i1] - G.start[i1] == G.end[i0] - G.start[i0] and
                               np.array_equal(G.uid[a], G.uid[b]) and np.array_equal(G.x[a], G.x[b]) and
                               np.array_equal(G.y[a], G.y[b]) and np.array_equal(G.v[a], G.v[b]))
    # input frames the node did not publish: true duplicates vs distinct states
    have = set(np.round(R["t"], 6))
    gin = (G.t >= R["t"][0]) & (G.t <= R["t"][-1])
    missing = np.array([round(t, 6) not in have for t in G.t]) & gin
    return {"n_out": int(len(R["t"])), "out_dt_lt_5ms_frac": float(np.mean(dt < 0.005)),
            "out_source_frame_is_duplicate_frac": float(src_dup.mean()),
            "out_same_content_as_prev_output_frac": float(same_as_prev_pub[1:].mean()),
            "input_dup_frac": float(dup[gin].mean()),
            "input_frames_not_published": int(missing.sum()),
            "not_published_that_were_duplicates": int(np.sum(missing & dup)),
            "not_published_distinct": int(np.sum(missing & ~dup)),
            "out_dt_ms": bt.stats(1000 * dt), "out_dt_max_s": float(dt.max())}


def accel_ratio(R, d, G):
    """smpc_decision updateOneVehicleHistory on an /smpc/targets stream (per uid, all roles +
    nearby, reset after a 0.35 s gap), sampled where the uid is target_rear; divided by the
    true MORAI acceleration.x of that uid in the same /Object_topic frame, during sustained
    acceleration (true > 0.5 within +-0.5 s, track older than 2 s)."""
    g = d["gt"]
    T = d["gt_msg_t"]
    # true accel lookup and sustained mask per (frame, uid)
    k_all = g["k"].astype(int)
    true = {}
    for u in np.unique(g["uid"]):
        m = np.flatnonzero(g["uid"] == u)
        t = T[k_all[m]]
        ax = g["ax"][m]
        for j, i in enumerate(k_all[m]):
            w = (t > t[j] - 0.5) & (t < t[j] + 0.5)
            sus = ax[j] > 0.5 and np.all(ax[w] > 0.4) and t[j] - t[0] > 2.0
            true[(int(i), int(u))] = (ax[j], sus)
    nb = R["nearby"]
    hist = {}
    out = []
    order = {}
    for k in range(len(R["t"])):
        ents = [(R[r]["uid"][k], R[r]["v_long"][k]) for r in ROLES if R[r]["valid"][k] > 0]
        m = np.flatnonzero(nb["k"] == k)
        ents += list(zip(nb["uid"][m], nb["v_long"][m]))
        st = R["t"][k]
        done = set()
        for u, v in ents:
            u = int(u)
            if u < 0 or u in done:
                continue
            done.add(u)
            h = hist.get(u)
            if h is None or st - h[2] > 0.35:
                hist[u] = [v, 0.0, st]
                continue
            dt = st - h[2]
            if 1e-3 < dt < 1.0:
                h[1] = 0.35 * np.clip((v - h[0]) / dt, -6.0, 6.0) + 0.65 * h[1]
            h[0], h[2] = v, st
        if R["target_rear"]["valid"][k] > 0:
            u = int(R["target_rear"]["uid"][k])
            i = G.index_of(st)
            if i is None or u not in hist:
                continue
            tv = true.get((int(i), u))
            if tv and tv[1]:
                out.append(hist[u][1] / tv[0])
    return bt.stats(out)


def ego_split(R, d, G, odom_recv):
    """Node ego_s vs /odom at the frame stamp, split by the sign of dt between the frame
    stamp and the newest /odom received before the frame (what the node holds)."""
    o = d["odom"]
    rows = []
    lanes = bt.load_lanes()
    for k in range(len(R["t"])):
        i = G.index_of(R["t"][k])
        if i is None:
            continue
        ox, oy, _ = gc.interp_pose(o, R["t"][k])
        v = float(np.interp(R["t"][k], o["t"], o["v"]))
        if v < 3.0:
            continue
        lane = lanes[int(R["cur_lane"][k])]
        s1 = lane.project(np.array([ox]), np.array([oy]), 0.0)[0][0]
        j = np.searchsorted(odom_recv[:, 0], G.bag_t[i], side="right") - 1
        rows.append((R["t"][k] - odom_recv[max(j, 0), 1], R["ego_s_cur"][k] - s1))
    a = np.asarray(rows)
    neg = a[:, 0] < 0
    return {"err": bt.stats(a[:, 1]), "abs_err": bt.stats(np.abs(a[:, 1])),
            "frac_odom_after_frame": float(neg.mean()),
            "err_when_odom_after_frame": bt.stats(a[neg, 1]),
            "err_when_odom_before_frame": bt.stats(a[~neg, 1])}


def odom_recv_times(path):
    import rosbag
    rows = []
    with rosbag.Bag(path) as b:
        for _, m, t in b.read_messages(topics=["/odom"]):
            rows.append((t.to_sec(), m.header.stamp.to_sec()))
    return np.asarray(rows)


class LiveFix2(gl.LiveCheck):
    def __init__(self, path, replay, lanes):
        super().__init__(path, lanes)
        self.R = gc.extract_targets(replay)


def frozen_spans():
    """Spans where /Object_topic keeps the same (uid, x, y, velocity.x) list at distinct
    stamps (> 5 ms apart): the drop rule publishes nothing there."""
    import glob
    out = {"bags": {}, "frames": 0, "same_sig_distinct": 0}
    paths = sorted(glob.glob(os.path.expanduser("~/bags/lc_gt_2026-09-*.bag"))) + [LIVE_REC]
    for p in paths:
        d = bt.load_bag(p, cache_dir=gc.CACHE if p == LIVE_REC else bt.DEFAULT_CACHE)
        g, T = d["gt"], d["gt_msg_t"]
        n = len(T)
        if n < 2:
            continue
        k = g["k"].astype(int)
        st = np.searchsorted(k, np.arange(n))
        en = np.searchsorted(k, np.arange(n), side="right")
        same = np.zeros(n, bool)
        for i in range(1, n):
            a, b = slice(st[i - 1], en[i - 1]), slice(st[i], en[i])
            if en[i] - st[i] == en[i - 1] - st[i - 1] and en[i] > st[i]:
                same[i] = (np.array_equal(g["uid"][a], g["uid"][b]) and np.array_equal(g["x"][a], g["x"][b]) and
                           np.array_equal(g["y"][a], g["y"][b]) and np.array_equal(g["vx_kmh"][a], g["vx_kmh"][b]))
        dt = np.r_[np.inf, np.diff(T)]
        longest, t0 = 0.0, None
        for i in range(n):
            if same[i]:
                t0 = T[i - 1] if t0 is None else t0
                longest = max(longest, T[i] - t0)
            else:
                t0 = None
        sd = int(np.sum(same & (dt > 0.005)))
        out["frames"] += n
        out["same_sig_distinct"] += sd
        out["bags"][bt.bag_label(p)] = {"frames": n, "same_sig_distinct": sd, "longest_span_s": float(longest)}
    spans = sorted(out["bags"].items(), key=lambda kv: -kv[1]["longest_span_s"])
    out["top"] = spans[:5]
    out["n_bags_span_gt_0p35"] = int(sum(1 for _, v in out["bags"].items() if v["longest_span_s"] > 0.35))
    out["n_bags"] = len(out["bags"])
    return out


def summarize_check(bc):
    r = bc.check_object_mode()
    keep = {k: r[k] for k in ("n_msgs", "rate", "role_cmp", "lane", "nearby_impl", "vel", "heading",
                              "ego_lag", "physical", "rear_for_gap")}
    keep["worst_E1"] = r["worst_E1"][:8]
    return keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    lanes = bt.load_lanes()
    res = {}
    # ---------------- live UDP bag
    gc.HEADING_BIAS_RAD = 0.0
    live = LiveFix2(LIVE_REC, LIVE_FIX2, lanes)
    L = summarize_check(live)
    Graw = gc.GTFrames(live.d, (0.0, 0.0))
    dup = dup_flags(Graw)
    L["dups_new"] = output_dups(live.R, Graw, dup)
    L["dups_recorded_live"] = output_dups(gc.extract_targets(LIVE_REC), Graw, dup)
    L["accel_ratio_new"] = accel_ratio(live.R, live.d, Graw)
    L["accel_ratio_recorded_live"] = accel_ratio(gc.extract_targets(LIVE_REC), live.d, Graw)
    orv = odom_recv_times(LIVE_REC)
    L["ego_new"] = ego_split(live.R, live.d, Graw, orv)
    L["ego_recorded_live"] = ego_split(gc.extract_targets(LIVE_REC), live.d, Graw, orv)
    L["recorded_live_role_cmp"] = gl.LiveCheck(LIVE_REC, lanes).check_object_mode()["role_cmp"]
    res["live"] = L
    # ---------------- September ROS-path bags
    gc.GTFrames = gf.FixedGTFrames
    res["sept"] = {}
    for lab in SEPT:
        orig = os.path.expanduser("~/bags/lc_gt_%s.bag" % lab)
        bc = ga.BagCheck(orig, os.path.join(gc.REPLAY_DIR, lab + "_fix2.bag"), lanes)
        S = summarize_check(bc)
        dupS = dup_flags(bc.G)
        S["dups_new"] = output_dups(bc.R, bc.G, dupS)
        S["accel_ratio_new"] = accel_ratio(bc.R, bc.d, bc.G)
        S["accel_ratio_fix1"] = accel_ratio(gc.extract_targets(os.path.join(gc.REPLAY_DIR, lab + "_fix.bag")),
                                            bc.d, bc.G)
        orv = odom_recv_times(orig)
        S["ego_new"] = ego_truth_suffix(lab, lanes, "_fix2", orv, bc)
        S["ego_fix1"] = ego_truth_suffix(lab, lanes, "_fix", orv, bc)
        res["sept"][lab] = S
    res["frozen"] = frozen_spans()
    res["fix1"] = __import__("json").load(open(os.path.join(gc.RESULTS, "gtcheck_fix.json")))["bags"]
    bt.write_json(os.path.join(gc.RESULTS, "gtcheck_fix2.json"), res)
    sec = section(res)
    with open(os.path.join(gc.RESULTS, "gtcheck_fix2_section.md"), "w") as fo:
        fo.write(sec)
    if a.append:
        with open(os.path.join(gc.RESULTS, "gtcheck_report.md"), "a") as fo:
            fo.write("\n" + sec)
    print(sec)


def ego_truth_suffix(lab, lanes, suf, odom_recv, bc):
    """Node ego_s vs /Ego_topic (exact rear axle at the frame stamp), split by dt sign."""
    e = bc.d["ego"]
    G = bc.G
    st = {round(t, 6): i for i, t in enumerate(e["t"])}
    R = gc.extract_targets(os.path.join(gc.REPLAY_DIR, lab + suf + ".bag"))
    rows = []
    for k in range(len(R["t"])):
        i = G.index_of(R["t"][k])
        ie = st.get(round(R["t"][k], 6))
        if i is None or ie is None or e["vx"][ie] < 3.0:
            continue
        j = np.searchsorted(odom_recv[:, 0], G.bag_t[i], side="right") - 1
        lane = lanes[int(R["cur_lane"][k])]
        s_true = lane.project(np.array([e["x"][ie] + gc.OFF_X]), np.array([e["y"][ie] + gc.OFF_Y]), 0.0)[0][0]
        rows.append((R["t"][k] - odom_recv[max(j, 0), 1], R["ego_s_cur"][k] - s_true))
    a = np.asarray(rows)
    neg = a[:, 0] < 0
    return {"err": bt.stats(a[:, 1]), "abs_err": bt.stats(np.abs(a[:, 1])),
            "frac_odom_after_frame": float(neg.mean()),
            "err_when_odom_after_frame": bt.stats(a[neg, 1]),
            "err_when_odom_before_frame": bt.stats(a[~neg, 1])}


def _agree(rc, role, key="all"):
    return rc["E1"][role][key]["agree"]


def _counts(rc, role):
    x = rc["E1"][role]["all"]
    return x["wrong_vehicle"] + x["missed"] + x["extra"]


def section(res):
    L = []
    w = L.append
    lv = res["live"]
    w("## Re-verification 2 (2026-11-04): duplicate-frame drop + backward ego extrapolation")
    w("")
    w("Node rebuilt 01:03 with `object_topic_drop_duplicate_frames: true` (objectCallback returns without "
      "publishing when the (unique_id, position.x/y, velocity.x) list equals the previous frame's) and the ego "
      "extrapolation clamp widened to [-0.1, +0.1] s. Replays with `gtcheck_replay2.py` (master 11411, "
      "--clock): the live UDP bag `func2_baseline_S2` with the research_morai settings (offset 0/0, heading "
      "offset 0; inputs /Object_topic /odom /gps_state /smpc/lane_change_status) -> "
      "`gtcheck_replay/func2_baseline_S2_fix2.bag`, and the 3 September ROS-path bags with the yaml settings "
      "-> `<bag>_fix2.bag`. 'Before' = the live node recording in the bag (previous build: no drop, clamp "
      "[0, 0.1]) and the re-verification-1 replays `<bag>_fix.bag`. Script `research/analysis/gtcheck_fix2.py`, "
      "numbers `gtcheck_fix2.json`.")
    w("")
    dn, do = lv["dups_new"], lv["dups_recorded_live"]
    an, ao = lv["accel_ratio_new"], lv["accel_ratio_recorded_live"]
    en, eo = lv["ego_new"], lv["ego_recorded_live"]
    rcn, rco = lv["role_cmp"], lv["recorded_live_role_cmp"]
    ln = lv["lane"]
    vel = lv["vel"]
    w("### Live UDP bag (func2_baseline_S2)")
    w("")
    w("| # | check | before (live recording) | after (rebuilt node) | result |")
    w("|---|---|---|---|---|")
    w("| (a) | outputs whose source frame is a duplicate of the previous frame | %s | %s | **%s** |" % (
        pct(do["out_source_frame_is_duplicate_frac"], 1), pct(dn["out_source_frame_is_duplicate_frac"], 1),
        "PASS" if dn["out_source_frame_is_duplicate_frac"] < 0.01 else "FAIL"))
    w("| (a) | consecutive outputs < 5 ms apart / same NPC content as previous output | %s / %s | %s / %s | **%s** |" % (
        pct(do["out_dt_lt_5ms_frac"], 1), pct(do["out_same_content_as_prev_output_frac"], 1),
        pct(dn["out_dt_lt_5ms_frac"], 1), pct(dn["out_same_content_as_prev_output_frac"], 1),
        "PASS" if dn["out_same_content_as_prev_output_frac"] < 0.01 else "FAIL"))
    w("| (a) | input frames not published: duplicates / distinct states | %d / %d | %d / %d | **%s** |" % (
        do["not_published_that_were_duplicates"], do["not_published_distinct"],
        dn["not_published_that_were_duplicates"], dn["not_published_distinct"],
        "PASS" if dn["not_published_distinct"] <= max(5, do["not_published_distinct"] + 5) else "CHECK"))
    w("| (a) | output count / rate / max output gap | %d / %s Hz / %s s | %d / %s Hz / %s s | **PASS** |" % (
        do["n_out"], f2(1000.0 / do["out_dt_ms"]["mean"], 1), f2(do["out_dt_max_s"], 3),
        dn["n_out"], f2(1000.0 / dn["out_dt_ms"]["mean"], 1), f2(dn["out_dt_max_s"], 3)))
    w("| (b) | observed-accel EMA / true MORAI accel, target_rear, sustained accel > 0.5: median (p10..p90), n | %s (%s..%s), n=%d | %s (%s..%s), n=%d | **%s** |" % (
        f2(ao["median"]), f2(ao["p10"]), f2(ao["p90"]), ao["n"], f2(an["median"]), f2(an["p10"]), f2(an["p90"]), an["n"],
        "PASS" if an.get("median", 0) >= 0.85 else "FAIL"))
    w("| (c) | node ego_s - /odom at frame stamp, moving: p10 / p50 / p90 / max abs | %s / %s / %s / %s m | %s / %s / %s / %s m | **PASS** |" % (
        f2(eo["err"]["p10"]), f2(eo["err"]["median"]), f2(eo["err"]["p90"]), f2(eo["abs_err"]["max"]),
        f2(en["err"]["p10"]), f2(en["err"]["median"]), f2(en["err"]["p90"]), f2(en["abs_err"]["max"])))
    w("| (c) | frames whose newest /odom is newer than the frame: share, error p10..p90 | %s, %s..%s m | %s, %s..%s m | **PASS** |" % (
        pct(eo["frac_odom_after_frame"], 0), f2(eo["err_when_odom_after_frame"].get("p10")), f2(eo["err_when_odom_after_frame"].get("p90")),
        pct(en["frac_odom_after_frame"], 0), f2(en["err_when_odom_after_frame"].get("p10")), f2(en["err_when_odom_after_frame"].get("p90"))))
    w("| (c) | E1 edge flips (role mismatches cf+tf+tr) | %d | %d | **%s** |" % (
        sum(_counts(rco, r) for r in ROLES), sum(_counts(rcn, r) for r in ROLES),
        "PASS" if sum(_counts(rcn, r) for r in ROLES) <= sum(_counts(rco, r) for r in ROLES) + 2 else "CHECK"))
    w("| (d) | A2 roles, node's own ego (E2) | 100%% | cf %s, tf %s, tr %s | **%s** |" % (
        pct(rcn["E2"]["current_front"]["all"]["agree"]), pct(rcn["E2"]["target_front"]["all"]["agree"]),
        pct(rcn["E2"]["target_rear"]["all"]["agree"]),
        "PASS" if min(rcn["E2"][r]["all"]["agree"] for r in ROLES) > 0.9999 else "CHECK"))
    w("| (d) | A2' roles, synchronous ego (E1) | cf %s, tf %s, tr %s | cf %s, tf %s, tr %s | **%s** |" % (
        pct(_agree(rco, "current_front")), pct(_agree(rco, "target_front")), pct(_agree(rco, "target_rear")),
        pct(_agree(rcn, "current_front")), pct(_agree(rcn, "target_front")), pct(_agree(rcn, "target_rear")),
        "PASS" if min(_agree(rcn, r) for r in ROLES) >= 0.999 else "CHECK"))
    w("| (d) | A1 lane_id, lane-centred NPC-frames | 100.00%% (15289) | %s (%d), %d wrong, %d missing | **PASS** |" % (
        pct(ln["agree_frac_centred"]), ln["n_centred"], ln["wrong_lane_known"], ln["missing_from_nearby_known"]))
    w("| (d) | A4 delta_s vs centre (E1), target_rear p50 / p90 / max | 0.00 / 0.03 / 0.42 m | %s / %s / %s m | **PASS** |" % (
        f2(vel["target_rear"]["delta_s_abs_err_E1"]["median"]), f2(vel["target_rear"]["delta_s_abs_err_E1"]["p90"]),
        f2(vel["target_rear"]["delta_s_abs_err_E1"]["max"])))
    w("| (d) | A3 v_long vs velocity.x/3.6 | 0.0000 | %s m/s | **PASS** |" % f2(max(
        vel[x]["v_long_minus_velx"]["max"] for x in ROLES + ["nearby"] if vel[x]["n"]), 4))
    w("")
    w("### September ROS-path bags (/Ego_topic = exact ego truth)")
    w("")
    w("| bag | input dup frames | outputs from dup frames | distinct frames not published | accel EMA ratio fix1 -> fix2 (median, n) | ego_s - /Ego_topic p10 / p50 / p90, fix1 -> fix2 | share odom newer than frame; its error p90, fix1 -> fix2 | E1 agree cf / tf / tr (fix2) | E1 mismatches fix1 -> fix2 | A1 centred | A4 dS E1 target_rear p90 / max | max output gap |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for lab, S in res["sept"].items():
        F1 = res["fix1"][lab]
        d2 = S["dups_new"]
        e1, e2 = S["ego_fix1"], S["ego_new"]
        rc = S["role_cmp"]
        mis1 = sum(F1["role_cmp"]["E1"][r]["all"]["wrong_vehicle"] + F1["role_cmp"]["E1"][r]["all"]["missed"] +
                   F1["role_cmp"]["E1"][r]["all"]["extra"] for r in ROLES)
        mis2 = sum(_counts(rc, r) for r in ROLES)
        w("| %s | %s | %s | %d | %s (%d) -> %s (%d) | %s / %s / %s -> %s / %s / %s | %s, %s -> %s, %s m | %s / %s / %s | %d -> %d | %s | %s / %s m | %s s |" % (
            lab, pct(d2["input_dup_frac"], 1), pct(d2["out_source_frame_is_duplicate_frac"], 1), d2["not_published_distinct"],
            f2(S["accel_ratio_fix1"].get("median")), S["accel_ratio_fix1"]["n"], f2(S["accel_ratio_new"].get("median")),
            S["accel_ratio_new"]["n"],
            f2(e1["err"]["p10"]), f2(e1["err"]["median"]), f2(e1["err"]["p90"]),
            f2(e2["err"]["p10"]), f2(e2["err"]["median"]), f2(e2["err"]["p90"]),
            pct(e1["frac_odom_after_frame"], 0), f2(e1["err_when_odom_after_frame"].get("p90")),
            pct(e2["frac_odom_after_frame"], 0), f2(e2["err_when_odom_after_frame"].get("p90")),
            pct(_agree(rc, "current_front")), pct(_agree(rc, "target_front")), pct(_agree(rc, "target_rear")),
            mis1, mis2, pct(S["lane"]["agree_frac_centred"]),
            f2(S["vel"]["target_rear"]["delta_s_abs_err_E1"]["p90"]), f2(S["vel"]["target_rear"]["delta_s_abs_err_E1"]["max"]),
            f2(d2["out_dt_max_s"], 3)))
    w("")
    fz = res["frozen"]
    w("")
    w("New caveat (dedup rule): the rule drops any frame equal to the previous one, not only the 1 ms UDP copy. "
      "When MORAI stalls, /Object_topic repeats one state with fresh stamps while the ego is frozen too (NPC "
      "velocity.x non-zero but positions constant). Across the %d bags, %d of %d frames (%s) are such distinct-time "
      "repeats; %d bags have a stall longer than 0.35 s. Longest: %s. During a stall the node now publishes "
      "nothing, so consumers see /smpc/targets go stale (0.35/0.5 s freshness gates; SMPC vehicle histories "
      "reset after 0.35 s, re-arming the 0.8 s new-rear-track guard on resume). The max output gap in the "
      "fix2 replays is in the table above. Suggested guard: drop a repeat only when it arrives within ~5 ms "
      "of the previous frame (the UDP copy is 1.0-1.5 ms), or republish at least every 0.1 s." % (
          fz["n_bags"], fz["same_sig_distinct"], fz["frames"], pct(fz["same_sig_distinct"] / max(fz["frames"], 1), 2),
          fz["n_bags_span_gt_0p35"], "; ".join("%s %.1f s" % (k, v["longest_span_s"]) for k, v in fz["top"][:4])))
    w("")
    worst = [c for S in res["sept"].values() for c in S["worst_E1"]] + lv["worst_E1"]
    if worst:
        w("Remaining E1 mismatches (largest): " + "; ".join(
            "%s t=%.2f %s node %d / GT %d (gap %.1f vs %.1f)" % (c["bag"][5:], c["t_rel"], c["role"], c["node_uid"],
                                                                c["gt_uid"], c["gap_node"], c["gap_gt"])
            for c in sorted(worst, key=lambda c: -c["severity"])[:6]) + ".")
        w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
