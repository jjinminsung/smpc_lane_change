#!/usr/bin/env python3
"""M3: NPC longitudinal behaviour from MORAI GT (/Object_topic, unique_id tracks).

Population (a) all highway NPCs: tracks assigned to one of lane CSVs 0..3 for
>= 2 s.  Population (b) relevant: NPCs in the ego's target lane (GT lane
assignment) within 120 m behind / 100 m ahead of the ego (target-lane CSV s)
during a decision window:
  * LC attempt window  = [status rising edge - 2 s, status falling edge]
  * lane-end wait      = status inactive, lane < 3, ego v < 0.5 m/s,
                         distance_to_lane_end <= 60 m  (target = lane + 1).
Outputs per-track summaries, episode lists, a parametric summary for the
simulator, and NPC lane-change detections.

  python3 research/analysis/m3_npc_longitudinal.py --bags ~/bags/lc_gt_2026-09-28-*.bag
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bagtools as bt  # noqa: E402
import lcevents  # noqa: E402
import npcmodel as nm  # noqa: E402

REAR_RANGE = 120.0
FRONT_RANGE = 100.0
WIN_PRE = 2.0
WAIT_V = 0.5
WAIT_DIST = 60.0
MIN_HWY_S = 2.0
LC_END_ZONE_M = 40.0            # NPC lane change this close to the from-lane CSV end = lane-end merge
LC_UPSTREAM_S_M = 5.0           # s < this = inside the 100 m tangent extension (geometry unreliable)
SPAWN_MAX_DIST = 290.0          # Object_topic range is ~300 m around the ego
ACCEL_BINS = [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 4.0]


def wait_windows(d, eg):
    tg = d["targets"]
    if len(tg["t"]) == 0:
        return []
    st = d["status"]
    active = bt.step_series(tg["t"], st["t"], st["active"], 0.0) > 0.5
    v = np.interp(tg["t"], eg["t"], eg["v"])
    m = (~active) & (tg["cur_lane"] < 3) & (v < WAIT_V) & (tg["dist_to_end"] <= WAIT_DIST)
    out = []
    for s, e in nm._runs(m):
        t0, t1 = tg["t"][s], tg["t"][e - 1]
        if out and t0 - out[-1]["t1"] < 1.0 and out[-1]["tgt"] == int(tg["cur_lane"][s]) + 1:
            out[-1]["t1"] = t1
            continue
        out.append({"kind": "lane_end_wait", "t0": float(t0), "t1": float(t1),
                    "tgt": int(tg["cur_lane"][s]) + 1, "src": int(tg["cur_lane"][s])})
    return [w for w in out if w["t1"] - w["t0"] >= 0.5]


def hist(values, bins):
    a = np.asarray([x for x in values if x is not None and np.isfinite(x)])
    if a.size == 0:
        return {"bins": bins, "counts": [0] * (len(bins) - 1), "n": 0}
    c, _ = np.histogram(np.clip(a, bins[0], bins[-1] - 1e-9), bins=bins)
    return {"bins": bins, "counts": c.tolist(), "frac": (c / a.size).tolist(), "n": int(a.size)}


def main():
    ap = bt.base_argparser(__doc__)
    args = ap.parse_args()
    paths = bt.expand_bags(args.bags)
    lanes = bt.load_lanes(args.lanes)
    ds = bt.load_bags(paths, args.cache, not args.no_cache, args.jobs)

    track_rows, all_eps, plateau_rows, lc_rows = [], [], [], []
    profile_rows = []
    pair_rows, windows_all = [], []
    frac_all = {"n": 0, "a>0.3": 0, "a>0.5": 0, "a>1.0": 0, "a<-0.3": 0, "a<-1.0": 0}
    frac_rel = {r: dict(frac_all) for r in ("rear", "front")}
    ax_corr = []

    for d in ds:
        eg = bt.ego_track(d)
        if len(eg["t"]) < 10:
            continue
        T0 = float(d["gt_msg_t"][0]) if len(d["gt_msg_t"]) else float(eg["t"][0])
        T1 = float(d["gt_msg_t"][-1]) if len(d["gt_msg_t"]) else float(eg["t"][-1])
        tracks = bt.gt_tracks(d)
        evs = lcevents.lc_events(d, lanes)
        wins = [{"kind": "lc_attempt", "t0": e["t0"] - WIN_PRE, "t1": e["t_end"],
                 "tgt": e["tgt"], "src": e["src"], "outcome": e["outcome"],
                 "lc_t0": e["t0"]} for e in evs]
        wins += wait_windows(d, eg)
        for w in wins:
            w["bag"] = d["label"]
        windows_all.extend(wins)

        kin = {}
        for uid, tr in tracks.items():
            k = nm.kinematics(tr)
            if k is None:
                continue
            tl = np.arange(k["t"][0], k["t"][-1], nm.LANE_DT)
            xl, yl = np.interp(tl, k["t"], k["x"]), np.interp(tl, k["t"], k["y"])
            lane, per = nm.lane_assign(lanes, xl, yl)
            hwy_s = float(np.sum(lane >= 0) * nm.LANE_DT)
            k["tl"], k["lane"], k["per"] = tl, lane, per
            kin[uid] = k
            if hwy_s < MIN_HWY_S:
                continue
            # /Object_topic only lists NPCs within ~300 m of the ego, so a first
            # appearance at the range edge is a range entry, not a spawn.
            ex0 = np.interp(tr["t"][0], eg["t"], eg["x"])
            ey0 = np.interp(tr["t"][0], eg["t"], eg["y"])
            dist_first = float(np.hypot(tr["x"][0] - ex0, tr["y"][0] - ey0))
            ex1 = np.interp(tr["t"][-1], eg["t"], eg["x"])
            ey1 = np.interp(tr["t"][-1], eg["t"], eg["y"])
            dist_last = float(np.hypot(tr["x"][-1] - ex1, tr["y"][-1] - ey1))
            spawned = bool(tr["t"][0] > T0 + 0.5 and dist_first < SPAWN_MAX_DIST)
            eps = nm.episodes(k, +1, tr["t"][0], spawned) + nm.episodes(k, -1, tr["t"][0], spawned)
            pls = nm.plateaus(k)
            # MORAI acceleration.x vs our derivative (sanity)
            axg = np.interp(k["t"], tr["t"], tr["ax"])
            ok = k["t"] > k["t"][0] + 0.5
            if ok.sum() > 50 and np.std(axg[ok]) > 0.05:
                ax_corr.append(float(np.corrcoef(axg[ok], k["a"][ok])[0, 1]))
            on = lane >= 0
            lanes_seen = sorted(set(int(x) for x in lane[on]))
            # lateral speed for lane changes
            lcs = nm.lane_changes(tl, lane, per)
            for c in lcs:
                # MORAI velocity.y is body-frame and stays ~0 during NPC lane
                # changes, so lateral speed is taken from the CSV d instead.
                c.pop("v_lat_peak", None)
                L0 = lanes[c["from"]].length
                if c["s_from"] >= L0 - LC_END_ZONE_M:
                    c["kind"] = "lane_end_merge"
                elif c["s_from"] < LC_UPSTREAM_S_M:
                    c["kind"] = "upstream_extension_zone"
                else:
                    c["kind"] = "mid_lane"
                c["v"] = float(np.interp(c["t"], k["t"], k["v"]))
                c.update({"bag": d["label"], "uid": uid})
                # was it in the ego's target lane near the ego at the time?
                lc_rows.append(c)
            row = {
                "bag": d["label"], "uid": uid, "type": tr["type"],
                "t_first_rel_bag": float(tr["t"][0] - T0),
                "duration": float(tr["t"][-1] - tr["t"][0]),
                "spawned_in_bag": spawned, "highway_s": hwy_s,
                "dist_to_ego_first": dist_first, "dist_to_ego_last": dist_last,
                "ended_in_bag": bool(tr["t"][-1] < T1 - 0.5),
                "despawned_near": bool(tr["t"][-1] < T1 - 0.5 and dist_last < SPAWN_MAX_DIST),
                "lanes_seen": lanes_seen,
                "first_lane": int(lane[on][0]) if on.any() else -1,
                "first_s": float(per[int(lane[on][0])][0][np.flatnonzero(on)[0]]) if on.any() else None,
                "v_first": float(np.median(tr["v"][:5])),
                "v_last": float(np.median(tr["v"][-5:])),
                "v_max": float(np.max(k["v"])), "v_min": float(np.min(k["v"])),
                "v_mean": float(np.mean(k["v"])),
                "final_cruise_v": pls[-1]["v_mean"] if pls else None,
                "n_accel_ep": sum(1 for e in eps if e["sign"] > 0),
                "n_decel_ep": sum(1 for e in eps if e["sign"] < 0),
                "n_lane_changes": sum(1 for c in lcs if c["both_lanes_exist"]),
                "length": float(np.median(tr["length"])),
            }
            track_rows.append(row)
            # 2 Hz speed / accel / lane profile for replay in the simulator
            tp = np.arange(k["t"][0], k["t"][-1], 0.5)
            ip = np.clip(np.searchsorted(k["t"], tp), 0, len(k["t"]) - 1)
            il = np.clip(np.searchsorted(tl, tp), 0, len(tl) - 1)
            for tt, i_, j_ in zip(tp, ip, il):
                profile_rows.append((d["label"], uid, round(float(tt - T0), 2), round(float(k["v"][i_]), 3),
                                     round(float(k["a"][i_]), 3) if k["valid"][i_] else "", int(lane[j_])))
            for e in eps:
                e.update({"bag": d["label"], "uid": uid})
                all_eps.append(e)
            for p in pls:
                p.update({"bag": d["label"], "uid": uid})
                plateau_rows.append(p)
            a = k["a"][(k["t"] > k["t"][0] + 0.5) & k["valid"]]
            frac_all["n"] += a.size
            for key, m in (("a>0.3", a > 0.3), ("a>0.5", a > 0.5), ("a>1.0", a > 1.0),
                           ("a<-0.3", a < -0.3), ("a<-1.0", a < -1.0)):
                frac_all[key] += int(m.sum())

        hwy_uids = set(r["uid"] for r in track_rows if r["bag"] == d["label"])
        eps_by_uid = {}
        for e in all_eps:
            if e["bag"] == d["label"]:
                eps_by_uid.setdefault(e["uid"], []).append(e)

        # Relevant population: target-lane NPCs near the ego in decision windows
        for wi, w in enumerate(wins):
            tg_lane = w["tgt"]
            if tg_lane not in lanes:
                continue
            tw = np.arange(max(w["t0"], eg["t"][0]), min(w["t1"], eg["t"][-1]) + 1e-9, nm.LANE_DT)
            if len(tw) < 2:
                continue
            ex, ey = np.interp(tw, eg["t"], eg["x"]), np.interp(tw, eg["t"], eg["y"])
            ev_ = np.interp(tw, eg["t"], eg["v"])
            es, _, _, _ = lanes[tg_lane].project(ex, ey, nm.EXT_BEFORE_M)
            rel_by_uid = {}
            closest_rear = np.full(len(tw), -1)
            closest_front = np.full(len(tw), -1)
            best_r = np.full(len(tw), np.inf)
            best_f = np.full(len(tw), np.inf)
            for uid in hwy_uids:
                k = kin[uid]
                cov = (tw >= k["t"][0]) & (tw <= k["t"][-1])
                if not cov.any():
                    continue
                lane_w = np.full(len(tw), -1)
                lane_w[cov] = k["lane"][np.clip(np.searchsorted(k["tl"], tw[cov]), 0, len(k["tl"]) - 1)]
                nx = np.interp(tw, k["t"], k["x"])
                ny = np.interp(tw, k["t"], k["y"])
                ns, _, _, _ = lanes[tg_lane].project(nx, ny, nm.EXT_BEFORE_M)
                dsv = ns - es
                rel = cov & (lane_w == tg_lane) & (dsv >= -REAR_RANGE) & (dsv <= FRONT_RANGE)
                if not rel.any():
                    continue
                rel_by_uid[uid] = (rel, dsv)
                r_mask = rel & (dsv < 0)
                f_mask = rel & (dsv >= 0)
                upd = r_mask & (-dsv < best_r)
                best_r = np.where(upd, -dsv, best_r)
                closest_rear = np.where(upd, uid, closest_rear)
                upd = f_mask & (dsv < best_f)
                best_f = np.where(upd, dsv, best_f)
                closest_front = np.where(upd, uid, closest_front)
            for uid, (rel, dsv) in rel_by_uid.items():
                k = kin[uid]
                ti = tw[rel]
                a_rel = np.interp(ti, k["t"], k["a"])
                v_rel = np.interp(ti, k["t"], k["v"])
                first = np.flatnonzero(rel)[0]
                role = "rear" if dsv[first] < 0 else "front"
                for rr in ("rear", "front"):
                    mm = (dsv[rel] < 0) if rr == "rear" else (dsv[rel] >= 0)
                    fr = frac_rel[rr]
                    fr["n"] += int(mm.sum())
                    for key, m in (("a>0.3", a_rel > 0.3), ("a>0.5", a_rel > 0.5),
                                   ("a>1.0", a_rel > 1.0), ("a<-0.3", a_rel < -0.3),
                                   ("a<-1.0", a_rel < -1.0)):
                        fr[key] += int((m & mm).sum())
                t_a, t_b = ti[0], ti[-1]
                acc_eps = [e for e in eps_by_uid.get(uid, []) if e["sign"] > 0 and
                           e["t_onset"] <= t_b and e["t_end"] >= t_a]
                dec_eps = [e for e in eps_by_uid.get(uid, []) if e["sign"] < 0 and
                           e["t_onset"] <= t_b and e["t_end"] >= t_a]
                acc_first = min(acc_eps, key=lambda e: e["t_onset"]) if acc_eps else None
                pair_rows.append({
                    "bag": d["label"], "window": wi, "window_kind": w["kind"],
                    "lc_pair": "%d->%d" % (w["src"], w["tgt"]),
                    "window_outcome": w.get("outcome"), "tgt": tg_lane, "uid": uid,
                    "role_first": role,
                    "was_closest_rear": bool(np.any(closest_rear[rel] == uid)),
                    "was_closest_front": bool(np.any(closest_front[rel] == uid)),
                    "relevant_s": float(rel.sum() * nm.LANE_DT),
                    "ds_first": float(dsv[first]), "ds_min": float(np.min(dsv[rel])),
                    "ds_max": float(np.max(dsv[rel])),
                    "v_first": float(v_rel[0]), "v_last": float(v_rel[-1]),
                    "v_ego_first": float(ev_[first]),
                    "a_max": float(np.max(a_rel)), "a_min": float(np.min(a_rel)),
                    "a_mean": float(np.mean(a_rel)),
                    "dv": float(v_rel[-1] - v_rel[0]),
                    "accel_episode": bool(acc_eps), "decel_episode": bool(dec_eps),
                    "accel_onset_rel_window_start": (acc_first["t_onset"] - t_a) if acc_first else None,
                    "accel_onset_inside": bool(acc_first and acc_first["t_onset"] >= t_a),
                    "accel_mean": acc_first["mean"] if acc_first else None,
                    "accel_peak": acc_first["peak"] if acc_first else None,
                    "accel_dv": acc_first["dv"] if acc_first else None,
                    "accel_is_spawn_episode": acc_first["spawn_episode"] if acc_first else None,
                })

    # NPC lane changes that happened while that NPC was relevant to a decision window
    win_index = {}
    for w in windows_all:
        win_index.setdefault(w["bag"], []).append(w)
    rel_pairs_by = {}
    for p_ in pair_rows:
        rel_pairs_by.setdefault((p_["bag"], p_["uid"]), []).append(p_)
    for c in lc_rows:
        c["in_relevant_window"] = False
        c["into_target_lane_in_window"] = False
        for p_ in rel_pairs_by.get((c["bag"], c["uid"]), []):
            w = win_index[c["bag"]][p_["window"]]
            if w["t0"] - 1.0 <= c["t"] <= w["t1"] + 1.0:
                c["in_relevant_window"] = True
                if c["to"] == w["tgt"]:
                    c["into_target_lane_in_window"] = True

    # ------------------------------------------------------------------ summaries
    def ep_summary(eps):
        return {
            "n": len(eps),
            "mean_accel": bt.stats([abs(e["mean"]) for e in eps]),
            "peak_accel": bt.stats([abs(e["peak"]) for e in eps]),
            "duration": bt.stats([e["duration"] for e in eps]),
            "dv": bt.stats([e["dv"] for e in eps]),
            "v_start": bt.stats([e["v_start"] for e in eps]),
            "v_end": bt.stats([e["v_end"] for e in eps]),
            "t_since_track_start": bt.stats([e["t_since_track_start"] for e in eps]),
            "mean_accel_hist": hist([abs(e["mean"]) for e in eps], ACCEL_BINS),
            "frac_from_steady": (sum(1 for e in eps if e["from_steady"]) / len(eps)) if eps else None,
        }

    acc = [e for e in all_eps if e["sign"] > 0]
    dec = [e for e in all_eps if e["sign"] < 0]
    acc_spawn = [e for e in acc if e["spawn_episode"]]
    acc_other = [e for e in acc if not e["spawn_episode"]]
    dec_other = [e for e in dec if not e["spawn_episode"]]
    hwy_minutes = sum(r["highway_s"] for r in track_rows) / 60.0
    spawned = [r for r in track_rows if r["spawned_in_bag"]]
    rel_uids = {(p["bag"], p["uid"]) for p in pair_rows}
    rel_tracks = [r for r in track_rows if (r["bag"], r["uid"]) in rel_uids]
    rel_eps = [e for e in all_eps if (e["bag"], e["uid"]) in rel_uids]

    # inter-onset intervals of non-spawn accel episodes within a track
    gaps = []
    by_track = {}
    for e in acc_other:
        by_track.setdefault((e["bag"], e["uid"]), []).append(e["t_onset"])
    for v in by_track.values():
        v = sorted(v)
        gaps += list(np.diff(v))

    def pair_summary(rows):
        n = len(rows)
        if n == 0:
            return {"n": 0}
        accs = [r for r in rows if r["accel_episode"]]
        return {
            "n_pairs": n,
            "n_unique_npcs": len({(r["bag"], r["uid"]) for r in rows}),
            "P_accel_episode_overlaps": len(accs) / n,
            "P_accel_onset_inside_window": sum(1 for r in rows if r["accel_onset_inside"]) / n,
            "P_accel_episode_nonspawn": sum(1 for r in accs if not r["accel_is_spawn_episode"]) / n,
            "P_decel_episode_overlaps": sum(1 for r in rows if r["decel_episode"]) / n,
            "frac_accel_eps_that_are_spawn_episode": (sum(1 for r in accs if r["accel_is_spawn_episode"]) / len(accs)) if accs else None,
            "P_a_max_gt_0.5": sum(1 for r in rows if r["a_max"] > 0.5) / n,
            "P_a_max_gt_1.0": sum(1 for r in rows if r["a_max"] > 1.0) / n,
            "accel_mean": bt.stats([r["accel_mean"] for r in accs]),
            "accel_peak": bt.stats([r["accel_peak"] for r in accs]),
            "accel_dv": bt.stats([r["accel_dv"] for r in accs]),
            "accel_onset_rel_window_start": bt.stats([r["accel_onset_rel_window_start"] for r in accs]),
            "a_max_in_window": bt.stats([r["a_max"] for r in rows]),
            "a_min_in_window": bt.stats([r["a_min"] for r in rows]),
            "dv_in_window": bt.stats([r["dv"] for r in rows]),
            "v_first": bt.stats([r["v_first"] for r in rows]),
            "ds_first": bt.stats([r["ds_first"] for r in rows]),
            "closing_speed_first": bt.stats([r["v_first"] - r["v_ego_first"] for r in rows]),
            "relevant_s": bt.stats([r["relevant_s"] for r in rows]),
        }

    rear_pairs = [p for p in pair_rows if p["role_first"] == "rear"]
    front_pairs = [p for p in pair_rows if p["role_first"] == "front"]
    rel_summary = {
        "rear_all": pair_summary(rear_pairs),
        "rear_closest": pair_summary([p for p in rear_pairs if p["was_closest_rear"]]),
        "rear_lc_attempt": pair_summary([p for p in rear_pairs if p["window_kind"] == "lc_attempt"]),
        "rear_lane_end_wait": pair_summary([p for p in rear_pairs if p["window_kind"] == "lane_end_wait"]),
        "rear_0->1": pair_summary([p for p in rear_pairs if p["lc_pair"] == "0->1"]),
        "rear_1->2": pair_summary([p for p in rear_pairs if p["lc_pair"] == "1->2"]),
        "rear_2->3": pair_summary([p for p in rear_pairs if p["lc_pair"] == "2->3"]),
        "front_all": pair_summary(front_pairs),
        "rear_by_date": {dt: pair_summary([p for p in rear_pairs if p["bag"].startswith(dt)])
                         for dt in sorted({p["bag"][:10] for p in rear_pairs})},
        "front_closest": pair_summary([p for p in front_pairs if p["was_closest_front"]]),
    }

    def frac_dict(f):
        n = max(f["n"], 1)
        return {"n_samples": f["n"], **{k: f[k] / n for k in f if k != "n"}}

    lc_real = [c for c in lc_rows if c["both_lanes_exist"]]
    res = {
        "definition": {
            "true_accel": "Savitzky-Golay derivative (0.55 s, quadratic) of GT speed on a 20 Hz grid, repeated MORAI frames removed",
            "episode": "a >= %.1f m/s^2 (or <= -%.1f), merged gaps <= %.1f s, duration >= %.1f s, |dv| >= %.1f m/s; onset = foot where |a| <= %.1f" % (
                nm.EP_ON, nm.EP_ON, nm.EP_MERGE_GAP, nm.EP_MIN_DUR, nm.EP_MIN_DV, nm.EP_FOOT),
            "spawn_episode": "accel episode starting <= 1.5 s after a track first appears (track appeared > 0.5 s after bag start)",
            "highway_npc": "assigned to a lane CSV (|d| <= %.1f m) for >= %.0f s" % (nm.LANE_ASSIGN_MAX_D, MIN_HWY_S),
            "relevant": "GT lane == ego target lane, ds in [-%d, +%d] m, during LC attempt [t0-2 s, end] or lane-end wait (v<0.5, dist_to_end<=60 m)" % (REAR_RANGE, FRONT_RANGE),
        },
        "n_bags": len(ds),
        "n_highway_tracks": len(track_rows),
        "n_spawned_in_bag": len(spawned),
        "highway_npc_minutes": hwy_minutes,
        "n_windows": {"lc_attempt": sum(1 for w in windows_all if w["kind"] == "lc_attempt"),
                      "lane_end_wait": sum(1 for w in windows_all if w["kind"] == "lane_end_wait")},
        "morai_ax_vs_derivative_corr": bt.stats(ax_corr),
        "all_highway": {
            "spawn_speed": bt.stats([r["v_first"] for r in spawned]),
            "spawn_speed_frac_at_13.89": (sum(1 for r in spawned if abs(r["v_first"] - 13.89) < 0.15) / len(spawned)) if spawned else None,
            "spawn_lane_counts": {str(l): sum(1 for r in spawned if r["first_lane"] == l) for l in range(-1, 4)},
            "spawn_s_on_lane": bt.stats([r["first_s"] for r in spawned]),
            "spawn_frac_at_0mps": (sum(1 for r in spawned if r["v_first"] < 0.5) / len(spawned)) if spawned else None,
            "n_despawned_near_ego": sum(1 for r in track_rows if r["despawned_near"]),
            "object_topic_range_note": "NPCs enter/leave /Object_topic at ~300 m from the ego",
            "v_max": bt.stats([r["v_max"] for r in track_rows]),
            "v_mean": bt.stats([r["v_mean"] for r in track_rows]),
            "final_cruise_v": bt.stats([r["final_cruise_v"] for r in track_rows]),
            "plateau_v": bt.stats([p["v_mean"] for p in plateau_rows]),
            "plateau_v_hist": hist([p["v_mean"] for p in plateau_rows], [0, 2, 6, 10, 14, 18, 22, 26, 30, 34]),
            "accel_spawn_episode": ep_summary(acc_spawn),
            "accel_nonspawn": ep_summary(acc_other),
            "decel_nonspawn": ep_summary(dec_other),
            "accel_nonspawn_from_steady": ep_summary([e for e in acc_other if e["from_steady"]]),
            "accel_episode_rate_per_npc_min": len(acc_other) / hwy_minutes if hwy_minutes else None,
            "decel_episode_rate_per_npc_min": len(dec_other) / hwy_minutes if hwy_minutes else None,
            "accel_inter_onset_s": bt.stats(gaps),
            "time_fraction": frac_dict(frac_all),
        },
        "relevant": {
            "n_tracks": len(rel_tracks),
            "track_v_max": bt.stats([r["v_max"] for r in rel_tracks]),
            "track_final_cruise_v": bt.stats([r["final_cruise_v"] for r in rel_tracks]),
            "accel_nonspawn": ep_summary([e for e in rel_eps if e["sign"] > 0 and not e["spawn_episode"]]),
            "decel_nonspawn": ep_summary([e for e in rel_eps if e["sign"] < 0 and not e["spawn_episode"]]),
            "time_fraction_rear": frac_dict(frac_rel["rear"]),
            "time_fraction_front": frac_dict(frac_rel["front"]),
            "pairs": rel_summary,
        },
        "lane_changes": {
            "n_assignment_changes_total": len(lc_rows),
            "n_between_existing_lanes": len(lc_real),
            "n_npcs_with_lane_change": len({(c["bag"], c["uid"]) for c in lc_real}),
            "kind_counts": {k: sum(1 for c in lc_real if c["kind"] == k)
                            for k in ("mid_lane", "lane_end_merge", "upstream_extension_zone")},
            "mid_lane_directions": {"%d->%d" % (a, b): sum(1 for c in lc_real if c["kind"] == "mid_lane" and c["from"] == a and c["to"] == b)
                                    for a in range(4) for b in range(4) if abs(a - b) == 1},
            "n_npcs_with_mid_lane_change": len({(c["bag"], c["uid"]) for c in lc_real if c["kind"] == "mid_lane"}),
            "mid_lane_lateral_rate_mps": bt.stats([c["lateral_rate_mps"] for c in lc_real if c["kind"] == "mid_lane"]),
            "mid_lane_in_relevant_window": sum(1 for c in lc_real if c["kind"] == "mid_lane" and c.get("in_relevant_window")),
            "mid_lane_into_target_lane_in_window": sum(1 for c in lc_real if c["kind"] == "mid_lane" and c.get("into_target_lane_in_window")),
            "mid_lane_speed": bt.stats([c["v"] for c in lc_real if c["kind"] == "mid_lane"]),
            "n_highway_npcs": len(track_rows),
            "directions": {"%d->%d" % (a, b): sum(1 for c in lc_real if c["from"] == a and c["to"] == b)
                           for a in range(4) for b in range(4) if abs(a - b) == 1},
            "bags_with_lane_change": sorted({c["bag"] for c in lc_real}),
            "events": lc_real[:400],
        },
    }
    # Compact parametric model for the simulator
    sp = res["all_highway"]
    ra = rel_summary["rear_all"]
    res["parametric_for_sim"] = {
        "spawn_speed_mps": sp["spawn_speed"],
        "spawn_accel_mps2": sp["accel_spawn_episode"]["mean_accel"],
        "spawn_accel_duration_s": sp["accel_spawn_episode"]["duration"],
        "desired_speed_after_spawn_mps": sp["accel_spawn_episode"]["v_end"],
        "cruise_speed_plateaus_mps": sp["plateau_v"],
        "accel_episode_rate_per_npc_min": sp["accel_episode_rate_per_npc_min"],
        "accel_mean_mps2": sp["accel_nonspawn"]["mean_accel"],
        "accel_mean_hist": sp["accel_nonspawn"]["mean_accel_hist"],
        "accel_duration_s": sp["accel_nonspawn"]["duration"],
        "accel_dv_mps": sp["accel_nonspawn"]["dv"],
        "accel_inter_onset_s": sp["accel_inter_onset_s"],
        "decel_episode_rate_per_npc_min": sp["decel_episode_rate_per_npc_min"],
        "decel_mean_mps2": sp["decel_nonspawn"]["mean_accel"],
        "decel_duration_s": sp["decel_nonspawn"]["duration"],
        "decel_dv_mps": sp["decel_nonspawn"]["dv"],
        "rear_P_accelerating_in_window": ra.get("P_accel_episode_overlaps"),
        "rear_P_accel_onset_inside_window": ra.get("P_accel_onset_inside_window"),
        "rear_accel_mean_mps2": ra.get("accel_mean"),
        "rear_accel_onset_rel_window_start_s": ra.get("accel_onset_rel_window_start"),
        "rear_time_fraction_a_gt_0.5": res["relevant"]["time_fraction_rear"].get("a>0.5"),
        "all_time_fraction_a_gt_0.5": sp["time_fraction"].get("a>0.5"),
        "npc_lane_changes_present": len(lc_real) > 0,
    }

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "day1_m3.json")
    res["episodes_sample_note"] = "full episode list in day1_m3_episodes.csv"
    bt.write_json(out, res)
    with open(os.path.join(args.out, "day1_m3_tracks.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(track_rows[0].keys()))
        w.writeheader()
        for r in track_rows:
            w.writerow(r)
    with open(os.path.join(args.out, "day1_m3_profiles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bag", "uid", "t_rel_bag_s", "v_mps", "a_mps2", "lane"])
        w.writerows(profile_rows)
    with open(os.path.join(args.out, "day1_m3_episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_eps[0].keys()))
        w.writeheader()
        for r in all_eps:
            w.writerow(r)
    with open(os.path.join(args.out, "day1_m3_relevant_pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        for r in pair_rows:
            w.writerow(r)

    print("bags=%d highway tracks=%d spawned=%d npc-min=%.1f windows=%s" % (
        len(ds), len(track_rows), len(spawned), hwy_minutes, res["n_windows"]))
    print("MORAI ax vs derivative corr:", bt.fmt_stats(res["morai_ax_vs_derivative_corr"]))
    print("spawn speed:", bt.fmt_stats(sp["spawn_speed"]), "frac@13.89=%.2f" % (sp["spawn_speed_frac_at_13.89"] or 0))
    print("spawn accel:", bt.fmt_stats(sp["accel_spawn_episode"]["mean_accel"]),
          "dur", bt.fmt_stats(sp["accel_spawn_episode"]["duration"]),
          "v_end", bt.fmt_stats(sp["accel_spawn_episode"]["v_end"]))
    print("plateau v:", bt.fmt_stats(sp["plateau_v"]))
    print("non-spawn accel ep: n=%d rate=%.2f/npc-min mean %s dur %s dv %s" % (
        len(acc_other), sp["accel_episode_rate_per_npc_min"],
        bt.fmt_stats(sp["accel_nonspawn"]["mean_accel"]), bt.fmt_stats(sp["accel_nonspawn"]["duration"]),
        bt.fmt_stats(sp["accel_nonspawn"]["dv"])))
    print("non-spawn decel ep: n=%d rate=%.2f/npc-min mean %s dur %s dv %s" % (
        len(dec_other), sp["decel_episode_rate_per_npc_min"],
        bt.fmt_stats(sp["decel_nonspawn"]["mean_accel"]), bt.fmt_stats(sp["decel_nonspawn"]["duration"]),
        bt.fmt_stats(sp["decel_nonspawn"]["dv"])))
    print("time fraction all:", {k: round(v, 3) for k, v in sp["time_fraction"].items()})
    print("time fraction rear:", {k: round(v, 3) for k, v in res["relevant"]["time_fraction_rear"].items()})
    print("time fraction front:", {k: round(v, 3) for k, v in res["relevant"]["time_fraction_front"].items()})
    for k, s in rel_summary.items():
        if k == "rear_by_date":
            continue
        if s.get("n_pairs"):
            print("%-18s pairs=%d npcs=%d P(acc ep)=%.2f [spawn-ep frac %s] P(onset inside)=%.2f P(dec ep)=%.2f P(amax>0.5)=%.2f acc_mean %s" % (
                k, s["n_pairs"], s["n_unique_npcs"], s["P_accel_episode_overlaps"],
                "%.2f" % s["frac_accel_eps_that_are_spawn_episode"] if s["frac_accel_eps_that_are_spawn_episode"] is not None else "-",
                s["P_accel_onset_inside_window"], s["P_decel_episode_overlaps"], s["P_a_max_gt_0.5"],
                bt.fmt_stats(s["accel_mean"])))
    lc = res["lane_changes"]
    print("NPC lane changes between existing lanes: %d events, %d/%d NPCs, kinds %s, mid-lane dirs %s, mid-lane NPCs %d, lateral rate %s" % (
        lc["n_between_existing_lanes"], lc["n_npcs_with_lane_change"], lc["n_highway_npcs"],
        lc["kind_counts"], lc["mid_lane_directions"], lc["n_npcs_with_mid_lane_change"],
        bt.fmt_stats(lc["mid_lane_lateral_rate_mps"])))
    print("  mid-lane NPC LCs during a decision window (NPC relevant): %d, into the ego target lane: %d" % (
        lc["mid_lane_in_relevant_window"], lc["mid_lane_into_target_lane_in_window"]))
    print("wrote", out)


if __name__ == "__main__":
    main()
