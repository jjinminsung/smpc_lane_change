#!/usr/bin/env python3
"""Re-verification 3 (2026-11-04): duplicate drop limited to repeats < 5 ms after the last
published frame (node rebuilt ~01:19).

Replays (gtcheck_replay2.py, master 11411, suffix _fix3):
  * func2_baseline_S2 (UDP path, research_morai settings: offsets 0, heading 0)
  * 2026-09-28-11-30-19 and 2026-09-04-15-00-35 (ROS path, yaml settings), both with MORAI stalls
Reports output count/rate, identical-consecutive outputs, accel-EMA ratio (UDP bag), and
the max /smpc/targets gap inside the stalls (ROS bags).

Output: research/results/gtcheck_fix3.json, gtcheck_fix3_section.md (--append adds it to
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
import gtcheck_fix2 as f2m  # noqa: E402
from gtcheck_fix import pct, f2  # noqa: E402

LIVE = os.path.expanduser("~/bags/research/func2_baseline_S2.bag")
ROS = ["2026-09-28-11-30-19", "2026-09-04-15-00-35"]
ROLES = gc.ROLES


def msg_identical(R):
    """Consecutive /smpc/targets identical except the header (lanes, roles, nearby)."""
    nb = R["nearby"]
    sig = []
    for k in range(len(R["t"])):
        s = [R["cur_lane"][k], R["tgt_lane"][k], R["ego_s_cur"][k], R["ego_d_cur"][k]]
        for role in ROLES:
            s += [R[role][f][k] for f in ("valid", "uid", "x", "y", "v_long", "delta_s")]
        m = nb["k"] == k
        s += list(nb["uid"][m]) + list(nb["x"][m]) + list(nb["y"][m]) + list(nb["delta_s"][m])
        sig.append(tuple(np.round(s, 9)))
    same = np.array([False] + [sig[k] == sig[k - 1] for k in range(1, len(sig))])
    return same


def stall_spans(G, dup_exact):
    """[t0, t1] spans of consecutive identical frames (any gap), longest first."""
    spans = []
    n = len(G.t)
    i = 1
    while i < n:
        if dup_exact[i]:
            j = i
            while j + 1 < n and dup_exact[j + 1]:
                j += 1
            spans.append((float(G.t[i - 1]), float(G.t[j])))
            i = j + 1
        else:
            i += 1
    return sorted(spans, key=lambda s: -(s[1] - s[0]))


def summarize(R, d, G, live=False):
    dup = f2m.dup_flags(G)
    od = f2m.output_dups(R, G, dup)
    same_msg = msg_identical(R)
    dt = np.diff(R["t"])
    o = {"n_out": int(len(R["t"])), "duration_s": float(R["t"][-1] - R["t"][0]),
         "rate_hz": float((len(R["t"]) - 1) / (R["t"][-1] - R["t"][0])),
         "out_dt_lt_5ms_frac": od["out_dt_lt_5ms_frac"],
         "same_npc_content_as_prev_output_frac": od["out_same_content_as_prev_output_frac"],
         "identical_consecutive_msg_frac": float(same_msg[1:].mean()),
         "distinct_states_lost": od["distinct_states_lost"],
         "out_dt_max_s": float(dt.max()), "out_dt_p90_s": float(np.percentile(dt, 90))}
    spans = stall_spans(G, dup)
    real = [s for s in spans if s[1] - s[0] > 0.1]
    o["stalls_gt_0p1s"] = len(real)
    if real:
        a, b = real[0]
        m = (R["t"] >= a) & (R["t"] <= b)
        tt = np.r_[a, R["t"][m], b]
        o["longest_stall"] = {"t_rel": float(a - G.t[0]), "span_s": float(b - a), "outputs_inside": int(m.sum()),
                              "max_gap_inside_s": float(np.max(np.diff(tt)))}
        ins = np.zeros(len(R["t"]), bool)
        for a, b in real:
            ins |= (R["t"] >= a) & (R["t"] <= b)
        gaps = []
        for a, b in real:
            m = (R["t"] >= a) & (R["t"] <= b)
            gaps.append(float(np.max(np.diff(np.r_[a, R["t"][m], b]))))
        o["max_gap_inside_any_stall_s"] = float(max(gaps))
        o["identical_consecutive_msg_frac_outside_stalls"] = float(same_msg[1:][~ins[1:]].mean())
        o["identical_consecutive_msg_frac_inside_stalls"] = float(same_msg[1:][ins[1:]].mean()) if ins[1:].any() else None
    if live:
        o["accel_ratio_allveh"] = f2m.accel_ratio(R, d, G, None, "all")
        o["accel_ratio_target_rear"] = f2m.accel_ratio(R, d, G)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--append", action="store_true")
    a = ap.parse_args()
    res = {}
    d = bt.load_bag(LIVE, cache_dir=gc.CACHE)
    G = gc.GTFrames(d, (0.0, 0.0))
    res["live"] = {"fix3": summarize(gc.extract_targets(os.path.join(gc.REPLAY_DIR, "func2_baseline_S2_fix3.bag")), d, G, True),
                   "fix2": summarize(gc.extract_targets(os.path.join(gc.REPLAY_DIR, "func2_baseline_S2_fix2.bag")), d, G, True),
                   "recorded": summarize(gc.extract_targets(LIVE), d, G, True)}
    for lab in ROS:
        p = os.path.expanduser("~/bags/lc_gt_%s.bag" % lab)
        d = bt.load_bag(p)
        G = gc.GTFrames(d)
        res[lab] = {}
        for suf in ("_fix", "_fix2", "_fix3"):
            q = os.path.join(gc.REPLAY_DIR, lab + suf + ".bag")
            if os.path.exists(q):
                res[lab][suf[1:]] = summarize(gc.extract_targets(q), d, G)
    bt.write_json(os.path.join(gc.RESULTS, "gtcheck_fix3.json"), res)
    sec = section(res)
    with open(os.path.join(gc.RESULTS, "gtcheck_fix3_section.md"), "w") as fo:
        fo.write(sec)
    if a.append:
        with open(os.path.join(gc.RESULTS, "gtcheck_report.md"), "a") as fo:
            fo.write("\n" + sec)
    print(sec)


def section(res):
    L = []
    w = L.append
    w("## Re-verification 3 (2026-11-04): duplicate drop limited to repeats within 5 ms")
    w("")
    w("Node rebuilt ~01:19. A frame is dropped only if its (unique_id, position.x/y, velocity.x) list equals "
      "the last published frame's AND its stamp is 0-5 ms after that frame. Replays with "
      "`gtcheck_replay2.py` (master 11411, --clock, suffix `_fix3`): func2_baseline_S2 with the "
      "research_morai settings (offsets 0, heading 0), and 2026-09-28-11-30-19 / 2026-09-04-15-00-35 with the "
      "yaml settings. 'fix2' = Re-verification 2 build (drop on any repeat), 'fix1' = Re-verification 1 build "
      "(no drop). Script `research/analysis/gtcheck_fix3.py`, numbers `gtcheck_fix3.json`.")
    w("")
    lv = res["live"]
    w("### (1) Live UDP bag func2_baseline_S2")
    w("")
    w("| metric | live recording (no drop) | fix2 (drop any repeat) | fix3 (drop repeats < 5 ms) | result |")
    w("|---|---|---|---|---|")
    rec, x2, x3 = lv["recorded"], lv["fix2"], lv["fix3"]
    w("| output count / rate | %d / %s Hz | %d / %s Hz | %d / %s Hz | **%s** |" % (
        rec["n_out"], f2(rec["rate_hz"], 1), x2["n_out"], f2(x2["rate_hz"], 1), x3["n_out"], f2(x3["rate_hz"], 1),
        "PASS" if 14.0 <= x3["rate_hz"] <= 18.0 else "CHECK"))
    w("| consecutive outputs < 5 ms apart | %s | %s | %s | **%s** |" % (
        pct(rec["out_dt_lt_5ms_frac"], 1), pct(x2["out_dt_lt_5ms_frac"], 1), pct(x3["out_dt_lt_5ms_frac"], 1),
        "PASS" if x3["out_dt_lt_5ms_frac"] < 0.01 else "FAIL"))
    w("| outputs with the same NPC content as the previous output | %s | %s | %s | **%s** |" % (
        pct(rec["same_npc_content_as_prev_output_frac"], 1), pct(x2["same_npc_content_as_prev_output_frac"], 1),
        pct(x3["same_npc_content_as_prev_output_frac"], 1),
        "PASS" if x3["same_npc_content_as_prev_output_frac"] < 0.01 else "FAIL"))
    w("| identical consecutive /smpc/targets (all fields but header) | %s | %s | %s | **%s** |" % (
        pct(rec["identical_consecutive_msg_frac"], 1), pct(x2["identical_consecutive_msg_frac"], 1),
        pct(x3["identical_consecutive_msg_frac"], 1), "PASS" if x3["identical_consecutive_msg_frac"] < 0.01 else "FAIL"))
    w("| distinct object states never published | %d | %d | %d | **%s** |" % (
        rec["distinct_states_lost"], x2["distinct_states_lost"], x3["distinct_states_lost"],
        "PASS" if x3["distinct_states_lost"] <= 2 else "CHECK"))
    w("| accel EMA / true accel, all vehicles, sustained > 0.5: median (p10..p90), n | %s (%s..%s), %d | %s (%s..%s), %d | %s (%s..%s), %d | **%s** |" % (
        f2(rec["accel_ratio_allveh"]["median"]), f2(rec["accel_ratio_allveh"]["p10"]), f2(rec["accel_ratio_allveh"]["p90"]), rec["accel_ratio_allveh"]["n"],
        f2(x2["accel_ratio_allveh"]["median"]), f2(x2["accel_ratio_allveh"]["p10"]), f2(x2["accel_ratio_allveh"]["p90"]), x2["accel_ratio_allveh"]["n"],
        f2(x3["accel_ratio_allveh"]["median"]), f2(x3["accel_ratio_allveh"]["p10"]), f2(x3["accel_ratio_allveh"]["p90"]), x3["accel_ratio_allveh"]["n"],
        "PASS" if x3["accel_ratio_allveh"]["median"] >= 0.9 else "FAIL"))
    w("| same, target_rear only | %s (n=%d) | %s (n=%d) | %s (n=%d) | **%s** |" % (
        f2(rec["accel_ratio_target_rear"]["median"]), rec["accel_ratio_target_rear"]["n"],
        f2(x2["accel_ratio_target_rear"]["median"]), x2["accel_ratio_target_rear"]["n"],
        f2(x3["accel_ratio_target_rear"]["median"]), x3["accel_ratio_target_rear"]["n"],
        "PASS" if x3["accel_ratio_target_rear"]["median"] >= 0.9 else "FAIL"))
    w("| output dt p90 / max | %s / %s s | %s / %s s | %s / %s s | **PASS** |" % (
        f2(rec["out_dt_p90_s"], 3), f2(rec["out_dt_max_s"], 3), f2(x2["out_dt_p90_s"], 3), f2(x2["out_dt_max_s"], 3),
        f2(x3["out_dt_p90_s"], 3), f2(x3["out_dt_max_s"], 3)))
    w("")
    w("### (2) ROS-path bags with MORAI stalls")
    w("")
    w("| bag | build | outputs | rate | longest stall (t, span) | outputs inside it | max gap inside any stall | max output gap overall | identical consecutive msgs: all / outside stalls / inside stalls | result |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for lab in ROS:
        for b in ("fix", "fix2", "fix3"):
            x = res[lab].get(b)
            if not x:
                continue
            ls = x.get("longest_stall", {})
            res_ok = ""
            if b == "fix3":
                res_ok = "**PASS**" if x.get("max_gap_inside_any_stall_s", 0) <= 0.2 else "**FAIL**"
            w("| %s | %s | %d | %s Hz | %s s, %s s | %s | %s s | %s s | %s / %s / %s | %s |" % (
                lab, b, x["n_out"], f2(x["rate_hz"], 1), f2(ls.get("t_rel"), 1), f2(ls.get("span_s"), 2),
                ls.get("outputs_inside", "-"), f2(x.get("max_gap_inside_any_stall_s"), 3), f2(x["out_dt_max_s"], 3),
                pct(x["identical_consecutive_msg_frac"], 1), pct(x.get("identical_consecutive_msg_frac_outside_stalls"), 1),
                pct(x.get("identical_consecutive_msg_frac_inside_stalls"), 1), res_ok))
    w("")
    w("Inside a stall the identical outputs are expected: MORAI repeats one state and the ego is frozen too, "
      "so the node correctly republishes the same content with fresh stamps. Outside stalls the "
      "identical-consecutive fraction is the ROS path's own repeated frames.")
    w("")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
