#!/usr/bin/env python3
"""GT check A (object-mode replay vs GT) and C (recorded LiDAR mode vs GT).

Inputs: original lc_gt bags (GT /Object_topic, /odom, status, recorded LiDAR
/smpc/targets) and the replayed object-mode /smpc/targets in
research/results/gtcheck_replay/<label>.bag (gtcheck_replay.py).

GT recomputation = Python port of the selector rules (baseline_spec.md S5) fed
with /Object_topic + (302595.0, 4124145.0), heading deg->rad, size max/min.
Two ego poses:
  E1 "sync": /odom interpolated at the targets header stamp (= the
     /Object_topic stamp), i.e. the true synchronous ego.
  E2 "node": the ego the node actually used, rebuilt from the message's
     ego_s_current / ego_d_current on the current lane CSV.
E2 isolates the implementation (NPC side), E1 adds the ego-pose staleness.

Output: research/results/gtcheck_a.json (+ worst-case CSVs).
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bagtools as bt  # noqa: E402
import gtcheck_common as gc  # noqa: E402

ROLES = gc.ROLES
GAP_BINS = [0, 20, 40, 60, 90, 120]


def lc_windows(d, pre=5.0):
    return [(iv["t_start"] - pre, iv["t_end"]) for iv in bt.status_intervals(d)]


def in_windows(t, wins):
    m = np.zeros(len(t), bool)
    for a, b in wins:
        m |= (t >= a) & (t <= b)
    return m


class BagCheck:
    def __init__(self, orig, replay, lanes):
        self.label = bt.bag_label(orig)
        self.lanes = lanes
        self.d = bt.load_bag(orig)
        self.G = gc.GTFrames(self.d)
        self.R = gc.extract_targets(replay)
        self.L = gc.extract_targets(orig)
        G = self.G
        # per GT row projections on every lane
        self.P0, self.P1 = {}, {}
        for lid, l in lanes.items():
            self.P0[lid] = l.project(G.x, G.y, 0.0)[:3]
            self.P1[lid] = l.project(G.x, G.y, gc.EXT_BEFORE)[:3]
        self.phys0 = gc.gt_lane(lanes, G.x, G.y, ext=False)
        self.phys1 = gc.gt_lane(lanes, G.x, G.y, ext=True)
        self.wins = lc_windows(self.d)

    # ------------------------------------------------------------------
    def ego_poses(self, T):
        o = self.d["odom"]
        e1x, e1y, e1yaw = gc.interp_pose(o, T["t"])
        e2x, e2y = np.zeros(len(T["t"])), np.zeros(len(T["t"]))
        for k in range(len(T["t"])):
            lane = self.lanes[int(T["cur_lane"][k])]
            e2x[k], e2y[k] = gc.lane_xy(lane, T["ego_s_cur"][k], T["ego_d_cur"][k])
        v = np.interp(T["t"], o["t"], o["v"])
        return {"E1": (e1x, e1y), "E2": (e2x, e2y), "v": v, "yaw": e1yaw}

    def ego_proj(self, ex, ey):
        out = {}
        for lid, l in self.lanes.items():
            out[(lid, 0)] = l.project(ex, ey, 0.0)[:2]
            out[(lid, 1)] = l.project(ex, ey, gc.EXT_BEFORE)[:2]
        return out

    # ------------------------------------------------------------------
    def check_object_mode(self):
        G, R, lanes = self.G, self.R, self.lanes
        n = len(R["t"])
        idx = np.array([G.index_of(t) if G.index_of(t) is not None else -1 for t in R["t"]])
        res = {"label": self.label, "n_msgs": n, "unmatched_stamps": int(np.sum(idx < 0))}
        eg = self.ego_poses(R)
        self.eg = eg
        EP = {e: self.ego_proj(*eg[e]) for e in ("E1", "E2")}
        win = in_windows(R["t"], self.wins)

        # ---- (6) rate / coverage ---------------------------------------
        t0, t1 = R["t"][0], R["t"][-1]
        gin = (G.t >= t0) & (G.t <= t1)
        have = set(np.round(R["t"], 6))
        miss = np.array([round(t, 6) not in have for t in G.t[gin]])
        gt_dt = np.diff(G.t)
        dup_in = np.zeros(len(G.t), bool)
        for i in range(1, len(G.t)):
            a, b = slice(G.start[i - 1], G.end[i - 1]), slice(G.start[i], G.end[i])
            if G.end[i] - G.start[i] == G.end[i - 1] - G.start[i - 1] and G.end[i] > G.start[i]:
                dup_in[i] = (np.array_equal(G.uid[a], G.uid[b]) and np.array_equal(G.x[a], G.x[b])
                             and np.array_equal(G.y[a], G.y[b]))
        sig = []
        nb = R["nearby"]
        for k in range(n):
            s = [R["cur_lane"][k], R["tgt_lane"][k]]
            for role in ROLES:
                s += [R[role][f][k] for f in ("uid", "x", "y", "v_long", "delta_s")]
            m = nb["k"] == k
            s += list(np.round(nb["uid"][m], 3)) + list(nb["x"][m]) + list(nb["y"][m])
            sig.append(tuple(s))
        same_prev = np.array([False] + [sig[k] == sig[k - 1] for k in range(1, n)])
        # identical role state of a moving role vehicle
        rep_same = {}
        for role in ROLES:
            V = R[role]
            ok = (V["valid"][1:] > 0) & (V["valid"][:-1] > 0) & (V["uid"][1:] == V["uid"][:-1]) \
                & (np.abs(V["v_long"][1:]) > 1.0)
            same = ok & (V["x"][1:] == V["x"][:-1]) & (V["y"][1:] == V["y"][:-1])
            rep_same[role] = {"pairs": int(ok.sum()), "identical": int(same.sum()),
                              "frac": float(same.sum() / max(ok.sum(), 1))}
        dt = np.diff(R["t"])
        # where are the missing frames: inside an Object_topic gap or a dropped callback?
        gtin = G.t[gin]
        miss_t = gtin[miss]
        res["rate"] = {
            "duration_s": float(t1 - t0), "out_msgs": n, "out_hz": float((n - 1) / (t1 - t0)),
            "in_msgs_window": int(gin.sum()), "in_hz": float((gin.sum() - 1) / (t1 - t0)),
            "coverage": float(1 - miss.mean()), "missing": int(miss.sum()),
            "first_out_after_first_in_s": float(t0 - G.t[0]),
            "dt_out": bt.stats(dt), "dt_out_gt_0p1s": int(np.sum(dt > 0.1)),
            "dt_out_max": float(dt.max()) if len(dt) else None,
            "dt_in": bt.stats(gt_dt[(G.t[1:] >= t0) & (G.t[1:] <= t1)]),
            "dt_in_le_1ms_frac": float(np.mean(gt_dt[(G.t[1:] >= t0) & (G.t[1:] <= t1)] <= 1e-3)),
            "input_dup_frames_frac": float(dup_in[gin].mean()),
            "output_identical_to_prev_frac": float(same_prev.mean()),
            "output_identical_given_input_dup": float(
                np.mean([same_prev[k] for k in range(1, n)
                         if idx[k] >= 0 and dup_in[idx[k]]]) if np.any(dup_in[idx[idx >= 0]]) else np.nan),
            "role_state_identical_moving": rep_same,
            "missing_in_runs": _runs_summary(miss),
        }

        # ---- per message comparisons --------------------------------------
        role_cmp = {e: {r: [] for r in ROLES} for e in ("E1", "E2")}
        lane_rows, vel_rows, ds_rows, near_impl = [], [], [], []
        mism_cases = []
        for k in range(n):
            i = idx[k]
            if i < 0:
                continue
            sl = slice(G.start[i], G.end[i])
            f = G.frame(i)
            cur, tgt = int(R["cur_lane"][k]), int(R["tgt_lane"][k])
            pc = tuple(a[sl] for a in self.P0[cur])
            pt = tuple(a[sl] for a in self.P1[tgt])
            gt_roles = {}
            for e in ("E1", "E2"):
                esc = EP[e][(cur, 0)][0][k]
                est = EP[e][(tgt, 1)][0][k]
                gt_roles[e] = gc.roles_from_proj(f, pc, pt, esc, est)
                for role in ROLES:
                    nu = int(R[role]["uid"][k]) if R[role]["valid"][k] > 0 else -1
                    gu = gt_roles[e][role]["uid"]
                    role_cmp[e][role].append((k, nu, gu, win[k]))
                    if e == "E1" and nu != gu:
                        mism_cases.append(self._mismatch_case(k, role, nu, gu, f, gt_roles[e],
                                                              cur, tgt, sl, eg, EP))
            uid2row = {int(u): j for j, u in enumerate(f["uid"])}
            m = np.flatnonzero(nb["k"] == k)
            node_lane = {int(nb["uid"][j]): j for j in m}
            # (1) physical lane of every NPC within +-120 m
            pl, pd, ps, pyaw, _ = (a[sl] if isinstance(a, np.ndarray) else a for a in self.phys1)
            pl0 = self.phys0[0][sl]
            for jr, u in enumerate(f["uid"]):
                u = int(u)
                lid = int(pl[jr])
                if lid >= 0:
                    ego_s = EP["E1"][(lid, 1)][0][k]
                    dsp = ps[jr] - ego_s
                else:
                    dsp = np.nan
                dist = np.hypot(f["x"][jr] - eg["E1"][0][k], f["y"][jr] - eg["E1"][1][k])
                if lid >= 0 and abs(dsp) > 120.0:
                    continue
                if lid < 0 and dist > 120.0:
                    continue
                j = node_lane.get(u)
                lane_rows.append((k, u, lid, int(pl0[jr]), float(pd[jr]) if lid >= 0 else np.nan,
                                  float(dsp), float(dist),
                                  int(nb["lane_id"][j]) if j is not None else -9,
                                  float(ps[jr]) if lid >= 0 else np.nan, win[k]))
            # nearby implementation equivalence (E2)
            gl, gds = self._nearby_port(f, sl, cur, EP["E2"], k)
            for jr, u in enumerate(f["uid"]):
                j = node_lane.get(int(u))
                near_impl.append((int(gl[jr]), int(nb["lane_id"][j]) if j is not None else -1,
                                  float(gds[jr]) if gl[jr] >= 0 else np.nan,
                                  float(nb["delta_s"][j]) if j is not None else np.nan))
            # (3)(4)(5) per vehicle entry (roles + nearby)
            entries = [(role, R[role], k) for role in ROLES if R[role]["valid"][k] > 0]
            entries += [("nearby", nb, j) for j in m]
            for role, V, j in entries:
                u = int(V["uid"][j])
                jr = uid2row.get(u)
                if jr is None:
                    vel_rows.append((role, u, np.nan) + (np.nan,) * 12)
                    continue
                lid = int(V["lane_id"][j])
                ext = 0 if role == "current_front" else 1
                P = (self.P0 if ext == 0 else self.P1)[lid]
                s_gt, d_gt, ly = P[0][sl][jr], P[1][sl][jr], P[2][sl][jr]
                hs_gt, _ = gc.footprint(f["yaw"][jr], ly, f["length"][jr], f["width"][jr])
                ego_s1 = EP["E1"][(lid, ext)][0][k]
                ego_s2 = EP["E2"][(lid, ext)][0][k]
                dpsi = gc.wrap(f["yaw"][jr] - gc.HEADING_BIAS_RAD - ly)
                v_ll = f["v"][jr] * np.cos(dpsi) - f["v_lat"][jr] * np.sin(dpsi)
                v_lt = f["v"][jr] * np.sin(dpsi) + f["v_lat"][jr] * np.cos(dpsi)
                vel_rows.append((role, u, V["v_long"][j] - f["v"][jr], V["v_long"][j] - v_ll,
                                 V["v_lat"][j] - f["v_lat"][jr], V["v_lat"][j], v_lt,
                                 V["delta_s"][j] - (s_gt - ego_s1), V["delta_s"][j] - (s_gt - ego_s2),
                                 V["s"][j] - s_gt, V["front_delta_s"][j] - (s_gt - ego_s1 + hs_gt),
                                 V["rear_delta_s"][j] - (s_gt - ego_s1 - hs_gt),
                                 V["half_s"][j] - hs_gt, np.degrees(gc.wrap(V["yaw"][j] - f["yaw"][jr])),
                                 f["v"][jr]))
                ds_rows.append((k, role, u, V["delta_s"][j], s_gt - ego_s1, win[k]))
        res["role_cmp"] = {e: {r: _role_summary(role_cmp[e][r]) for r in ROLES} for e in role_cmp}
        res["lane"] = _lane_summary(lane_rows)
        res["nearby_impl"] = _near_impl_summary(near_impl)
        res["vel"] = _vel_summary(vel_rows)
        res["heading"] = self._heading_summary()
        res["ego_lag"] = self._ego_lag(R, eg, EP)
        res["lane_pair_inputs"] = self._lane_pair_inputs(R)
        res["mismatch_classes_E1"] = _classify(mism_cases)
        res["worst_E1"] = sorted(mism_cases, key=lambda c: -c["severity"])[:12]
        res["physical"] = self._physical_roles(R, idx, EP, win)
        res["rear_for_gap"] = self._rear_for_gap(R, idx, win)
        self.obj_roles = role_cmp
        self.idx = idx
        self.eg = eg
        return res

    def _nearby_port(self, f, sl, cur, EPe, k):
        n = len(f["uid"])
        best = np.full(n, np.inf)
        lane = np.full(n, -1)
        dso = np.full(n, np.nan)
        thr = gc.LANE_HALF_WIDTH + gc.BBOX_MARGIN + gc.NEARBY_MARGIN
        for lid in sorted(self.lanes):
            s, d, yw = (a[sl] for a in self.P1[lid])
            es = EPe[(lid, 1)][0][k]
            hs, hd = gc.footprint(f["yaw"], yw, f["length"], f["width"])
            clear = np.abs(d) - hd
            ds = s - es
            ok = (clear <= thr) & ~((ds + hs) < -gc.NEARBY_REAR) & ~((ds - hs) > gc.NEARBY_FRONT)
            score = np.maximum(0.0, clear) + 0.01 * np.abs(ds) + 0.001 * abs(lid - cur)
            better = ok & (score < best)
            best = np.where(better, score, best)
            lane = np.where(better, lid, lane)
            dso = np.where(better, ds, dso)
        # max_nearby_vehicles truncation by |delta_s|
        sel = np.flatnonzero(lane >= 0)
        if len(sel) > gc.MAX_NEARBY:
            keep = sel[np.argsort(np.abs(dso[sel]), kind="stable")[:gc.MAX_NEARBY]]
            drop = np.setdiff1d(sel, keep)
            lane[drop] = -1
        return lane, dso

    def _mismatch_case(self, k, role, nu, gu, f, roles, cur, tgt, sl, eg, EP):
        R = self.R
        uid2row = {int(u): j for j, u in enumerate(f["uid"])}
        aux = roles["_aux"]
        c = {"bag": self.label, "t_rel": float(R["t"][k] - self.G.t[0]), "k": int(k), "role": role,
             "node_uid": nu, "gt_uid": gu, "cur": cur, "tgt": tgt,
             "ego_v": float(eg["v"][k]), "ego_s_cur": float(R["ego_s_cur"][k]),
             "cur_len": float(R["cur_len"][k])}
        for tag, u in (("node", nu), ("gt", gu)):
            if u < 0 or u not in uid2row:
                c[tag] = None
                continue
            j = uid2row[u]
            side = "c" if role == "current_front" else "t"
            ds = aux["ds_" + side][j]
            hs = aux["f%ss" % side][j]
            dd = aux["p%sd" % side][j]
            c[tag] = {"ds": float(ds), "front": float(ds + hs), "rear": float(ds - hs),
                      "d": float(dd), "rep": bool(aux["rep_" + side][j]),
                      "phys_lane": int(self.phys1[0][sl][j]), "phys_d": float(self.phys1[1][sl][j]),
                      "s_on_lane": float(self.P1[cur if side == "c" else tgt][0][sl][j]),
                      "v": float(f["v"][j])}
        gaps = []
        for tag in ("node", "gt"):
            v = c[tag]
            if v is None:
                gaps.append(120.0)
            elif role == "target_rear":
                gaps.append(max(0.0, -v["front"]))
            else:
                gaps.append(max(0.0, v["rear"]))
        c["gap_node"], c["gap_gt"] = gaps
        c["severity"] = abs(gaps[0] - gaps[1])
        return c

    def _heading_summary(self):
        G = self.G
        pl, pd, ps, pyaw, _ = self.phys0
        ok = (pl >= 0) & (np.abs(pd) < 0.5) & (G.v > 1.0)
        err = np.degrees(gc.wrap(G.yaw - pyaw))[ok]
        # alternative conventions, to show the deg->rad ENU reading is the right one
        alt = {"enu_deg": err,
               "north_cw_deg": np.degrees(gc.wrap(np.radians(90.0 - G.heading_deg) - pyaw))[ok],
               "rad_as_is": np.degrees(gc.wrap(G.heading_deg - pyaw))[ok]}
        # velocity-direction check from GT positions (moving NPCs)
        tr = bt.gt_tracks(self.d, {"dx": gc.OFF_X, "dy": gc.OFF_Y})
        dv = []
        for u, t in tr.items():
            if len(t["t"]) < 20:
                continue
            m = np.concatenate([[True], (np.diff(t["x"]) != 0) | (np.diff(t["y"]) != 0)])
            x, y, tt, hd, v = t["x"][m], t["y"][m], t["t"][m], t["heading"][m], t["v"][m]
            for a in range(0, len(x) - 10, 10):
                b = a + 10
                dist = np.hypot(x[b] - x[a], y[b] - y[a])
                if dist < 2.0:
                    continue
                vyaw = np.arctan2(y[b] - y[a], x[b] - x[a])
                hm = np.radians(hd[(a + b) // 2])
                vs = dist / max(tt[b] - tt[a], 1e-3)
                dv.append((np.degrees(gc.wrap(hm - vyaw)), np.median(v[a:b + 1]) / max(vs, 1e-3)))
        dv = np.asarray(dv) if dv else np.zeros((0, 2))
        return {"lane_centred_moving_rows": int(ok.sum()),
                "err_deg": {kk: bt.stats(np.abs(vv)) for kk, vv in alt.items()},
                "err_deg_signed_enu": bt.stats(err),
                "frac_abs_err_gt_5deg": float(np.mean(np.abs(err) > 5.0)),
                "frac_abs_err_gt_90deg": float(np.mean(np.abs(err) > 90.0)),
                "heading_vs_motion_dir_deg": bt.stats(np.abs(dv[:, 0])) if len(dv) else None,
                "speed_ratio_vel_x_over_pos_deriv": bt.stats(dv[:, 1]) if len(dv) else None}

    def _ego_lag(self, R, eg, EP):
        e1 = np.array([EP["E1"][(int(c), 0)][0][k] for k, c in enumerate(R["cur_lane"])])
        err = R["ego_s_cur"] - e1
        v = eg["v"]
        mv = v > 3.0
        lag = -err[mv] / v[mv]
        # which odom sample did the node use? lag of the ZOH odom vs the stamp
        o = self.d["odom"]
        j = np.searchsorted(o["t"], R["t"], side="right") - 1
        zoh_age = R["t"] - o["t"][np.clip(j, 0, len(o["t"]) - 1)]
        return {"ego_s_err_m": bt.stats(err), "ego_s_err_moving_m": bt.stats(err[mv]),
                "implied_lag_s_moving": bt.stats(lag),
                "zoh_odom_age_s": bt.stats(zoh_age),
                "odom_dt_s": bt.stats(np.diff(o["t"]))}

    def _lane_pair_inputs(self, R):
        d = self.d
        gl = bt.step_series(R["bag_t"], d["gps"]["t"], d["gps"]["lane"], -1)
        st = d["status"]
        act = bt.step_series(R["t"], st["t"], st["active"], 0) > 0.5
        src = bt.step_series(R["t"], st["t"], st["src"], -1)
        tgt = bt.step_series(R["t"], st["t"], st["tgt"], -1)
        exp_cur = np.where(act, src, gl)
        exp_tgt = np.where(act, tgt, np.where(exp_cur < 3, exp_cur + 1, exp_cur))
        return {"cur_agree": float(np.mean(exp_cur == R["cur_lane"])),
                "tgt_agree": float(np.mean(exp_tgt == R["tgt_lane"]))}

    def _physical_roles(self, R, idx, EP, win):
        """Spec roles (on GT) vs physical occupancy roles.

        Physical: any part of the GT footprint inside the 3.5 m lane strip
        (|d| - half_d < 1.75).  Current lane: plain projection (CSV interior);
        target lane: 300 m before-start extension so that vehicles the 100 m
        perception extension cannot see are counted too.  For every
        disagreement, record whether the physical vehicle is still visible to
        the SMPC on that lane through nearby_vehicles (lane_id == role lane,
        the targetRearForGap / ACC nearby paths)."""
        G = self.G
        nb = R["nearby"]
        out = {r: {"n": 0, "agree": 0, "classes": {}, "visible_in_nearby_same_lane": 0,
                   "gap_shortfall_m": []} for r in ROLES}
        for k in range(len(R["t"])):
            i = idx[k]
            if i < 0:
                continue
            sl = slice(G.start[i], G.end[i])
            f = G.frame(i)
            cur, tgt = int(R["cur_lane"][k]), int(R["tgt_lane"][k])
            m = nb["k"] == k
            near_lane = dict(zip(nb["uid"][m].astype(int), nb["lane_id"][m].astype(int)))
            for role in ROLES:
                if role != "current_front" and tgt == cur:
                    continue   # final lane: target == current, target roles are empty by design
                lid = cur if role == "current_front" else tgt
                ext = 0.0 if role == "current_front" else 300.0
                s, d, yw = self.lanes[lid].project(f["x"], f["y"], ext)[:3]
                hs, hd = gc.footprint(f["yaw"], yw, f["length"], f["width"])
                es = self.lanes[lid].project(np.array([self.eg["E1"][0][k]]),
                                             np.array([self.eg["E1"][1][k]]), ext)[0][0]
                ds = s - es
                inside = (np.abs(d) - hd) < 1.75
                if role == "target_rear":
                    cand = inside & (ds - hs <= 0)
                    key = np.maximum(0, -(ds + hs))
                else:
                    cand = inside & ((ds + hs > 0) if role == "current_front" else (ds + hs >= 0))
                    key = np.maximum(0, ds - hs)
                cand &= key < 120.0
                pj = np.flatnonzero(cand)[np.argmin(key[cand])] if cand.any() else -1
                pu = int(f["uid"][pj]) if pj >= 0 else -1
                nu = int(R[role]["uid"][k]) if R[role]["valid"][k] > 0 else -1
                o = out[role]
                o["n"] += 1
                if pu == nu:
                    o["agree"] += 1
                    continue
                cls = "other"
                if pu >= 0:
                    absd = abs(d[pj])
                    if ext > 0 and s[pj] < -gc.EXT_BEFORE:
                        cls = "beyond_100m_target_extension"
                    elif absd > gc.REP_MAX_ABS_D:
                        cls = "partly_in_lane_|d|>1.35"
                    else:
                        cls = "assigned_to_other_lane_by_clearance"
                    if near_lane.get(pu) == lid:
                        o["visible_in_nearby_same_lane"] += 1
                    if nu >= 0:
                        nj = np.flatnonzero(f["uid"] == nu)
                        if len(nj):
                            o["gap_shortfall_m"].append(float(key[nj[0]] - key[pj]))
                    else:
                        o["gap_shortfall_m"].append(float(120.0 - key[pj]))
                elif nu >= 0:
                    cls = "node_vehicle_not_physically_in_lane"
                o["classes"][cls] = o["classes"].get(cls, 0) + 1
        for r in ROLES:
            out[r]["agree_frac"] = out[r]["agree"] / max(out[r]["n"], 1)
            out[r]["gap_shortfall_m"] = bt.stats(out[r]["gap_shortfall_m"])
        return out

    def _rear_for_gap(self, R, idx, win):
        """SMPC targetRearForGap (spec S0) on the object-mode message: min rear bumper
        gap over target_rear + nearby(lane_id == target lane, rear edge <= 0).  Is
        the chosen vehicle physically in the target lane?"""
        G = self.G
        nb = R["nearby"]
        rows = []
        for k in range(len(R["t"])):
            i = idx[k]
            cur, tgt = int(R["cur_lane"][k]), int(R["tgt_lane"][k])
            if i < 0 or cur == tgt:
                continue
            cands = []
            if R["target_rear"]["valid"][k] > 0:
                V = R["target_rear"]
                cands.append((int(V["uid"][k]), V["front_delta_s"][k], V["rear_delta_s"][k]))
            m = np.flatnonzero((nb["k"] == k) & (nb["lane_id"] == tgt))
            for j in m:
                cands.append((int(nb["uid"][j]), nb["front_delta_s"][j], nb["rear_delta_s"][j]))
            cands = [c for c in cands if c[2] <= 0.0]
            if not cands:
                continue
            u, fr, _ = min(cands, key=lambda c: -c[1] - 0.79)
            sl = slice(G.start[i], G.end[i])
            jr = np.flatnonzero(G.uid[sl] == u)
            if not len(jr):
                continue
            jr = jr[0]
            s_, d_, yw = (a[sl][jr] for a in self.P1[tgt])
            _, hd = gc.footprint(G.yaw[sl][jr], yw, G.length[sl][jr], G.width[sl][jr])
            phys_in = abs(d_) - hd < 1.75
            rows.append((win[k], phys_in, int(self.phys1[0][sl][jr]), abs(d_), -fr - 0.79,
                         u != (int(R["target_rear"]["uid"][k]) if R["target_rear"]["valid"][k] > 0 else -1)))
        a = np.asarray(rows, float)
        if not len(a):
            return {}
        out = {}
        for name, m in (("all", np.ones(len(a), bool)), ("lc_window", a[:, 0] > 0)):
            b = a[m]
            out[name] = {"n": int(len(b)),
                         "chosen_is_nearby_not_target_rear": int(np.sum(b[:, 5] > 0)),
                         "chosen_not_physically_in_target_lane": int(np.sum(b[:, 1] < 1)),
                         "chosen_outside_all_csv_lanes": int(np.sum(b[:, 2] < 0)),
                         "gap_when_not_physical": bt.stats(b[b[:, 1] < 1, 4])}
        return out

    # ------------------------------------------------------------------
    def check_lidar_mode(self):
        """C: recorded LiDAR-mode /smpc/targets vs GT roles at the LiDAR stamps."""
        G, L, lanes = self.G, self.L, self.lanes
        n = len(L["t"])
        eg = self.ego_poses(L)
        EP1 = self.ego_proj(*eg["E1"])
        win = in_windows(L["t"], self.wins)
        res = {r: [] for r in ROLES}
        perr = []
        for k in range(n):
            t = L["t"][k]
            i = int(np.clip(np.searchsorted(G.t, t), 1, len(G.t) - 1))
            i = i if abs(G.t[i] - t) < abs(G.t[i - 1] - t) else i - 1
            if abs(G.t[i] - t) > 0.05:
                continue
            f = G.frame(i)
            dt = t - G.t[i]
            f = dict(f)
            f["x"] = f["x"] + f["v"] * np.cos(f["yaw"]) * dt
            f["y"] = f["y"] + f["v"] * np.sin(f["yaw"]) * dt
            cur, tgt = int(L["cur_lane"][k]), int(L["tgt_lane"][k])
            if cur not in lanes or tgt not in lanes:
                continue
            pc = lanes[cur].project(f["x"], f["y"], 0.0)[:3]
            pt = lanes[tgt].project(f["x"], f["y"], gc.EXT_BEFORE)[:3]
            roles = gc.roles_from_proj(f, pc, pt, EP1[(cur, 0)][0][k], EP1[(tgt, 1)][0][k])
            for role in ROLES:
                V = L[role]
                gu = roles[role]["uid"]
                ggap = roles[role]["gap"] if gu >= 0 else np.nan
                if V["valid"][k] > 0:
                    mu, md = match_gt(V["x"][k], V["y"][k], f)
                    lu = -2 if mu < 0 else mu      # -2 = ghost (no GT vehicle there)
                else:
                    lu, md = -1, np.nan
                res[role].append((k, lu, gu, ggap, win[k], md))
            # position error of every matched LiDAR vehicle (roles + nearby) on its lane
            nbl = L["nearby"]
            ents = [(L[r], k) for r in ROLES if L[r]["valid"][k] > 0]
            ents += [(nbl, j) for j in np.flatnonzero(nbl["k"] == k)]
            seen = set()
            for V, j in ents:
                lid = int(V["lane_id"][j])
                if lid not in lanes:
                    continue
                mu, md = match_gt(V["x"][j], V["y"][j], f)
                if mu < 0 or (mu, int(V["uid"][j])) in seen:
                    continue
                seen.add((mu, int(V["uid"][j])))
                jr = int(np.flatnonzero(f["uid"] == mu)[0])
                sg, dg, _, _ = lanes[lid].project(f["x"][jr:jr + 1], f["y"][jr:jr + 1], gc.EXT_BEFORE)
                sl_, dl_, _, _ = lanes[lid].project(np.array([V["x"][j]]), np.array([V["y"][j]]), gc.EXT_BEFORE)
                es = lanes[lid].project(eg["E1"][0][k:k + 1], eg["E1"][1][k:k + 1], gc.EXT_BEFORE)[0][0]
                perr.append((sg[0] - es, dl_[0] - dg[0], sl_[0] - sg[0], f["length"][jr], eg["v"][k]))
        out = {r: np.asarray(res[r], float) for r in ROLES}
        out["pos_err"] = np.asarray(perr, float)
        return out


def match_gt(x, y, f, max_out=1.5):
    """GT NPC whose footprint is closest to a LiDAR box centre (<= max_out m outside)."""
    c, s = np.cos(f["yaw"]), np.sin(f["yaw"])
    dx, dy = x - f["x"], y - f["y"]
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    out = np.hypot(np.maximum(np.abs(lx) - 0.5 * f["length"], 0),
                   np.maximum(np.abs(ly) - 0.5 * f["width"], 0))
    if len(out) == 0:
        return -1, np.nan
    j = int(np.argmin(out + 1e-3 * np.hypot(dx, dy)))
    return (int(f["uid"][j]) if out[j] <= max_out else -1), float(out[j])


def _runs_summary(mask):
    runs = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            runs.append(j - i + 1)
            i = j + 1
        else:
            i += 1
    runs = np.asarray(runs)
    return {"n_runs": int(len(runs)), "len_hist": {str(L): int(np.sum(runs == L)) for L in (1, 2, 3)},
            "len_gt3": int(np.sum(runs > 3)), "max_run": int(runs.max()) if len(runs) else 0}


def _role_summary(rows):
    a = np.asarray([(nu, gu, w) for _, nu, gu, w in rows], float)
    out = {}
    for name, m in (("all", np.ones(len(a), bool)), ("lc_window", a[:, 2] > 0)):
        b = a[m]
        nu, gu = b[:, 0], b[:, 1]
        out[name] = {"n": int(len(b)), "agree": float(np.mean(nu == gu)) if len(b) else None,
                     "both_valid_same": int(np.sum((nu == gu) & (gu >= 0))),
                     "both_invalid": int(np.sum((nu < 0) & (gu < 0))),
                     "wrong_vehicle": int(np.sum((nu >= 0) & (gu >= 0) & (nu != gu))),
                     "missed": int(np.sum((nu < 0) & (gu >= 0))),
                     "extra": int(np.sum((nu >= 0) & (gu < 0))),
                     "gt_valid": int(np.sum(gu >= 0))}
    return out


def _lane_summary(rows):
    a = np.asarray(rows, float)
    if not len(a):
        return {}
    k, u, gl, gl0, gd, dsp, dist, nl, ps, w = a.T
    out = {"n": int(len(a))}
    have = gl >= 0
    out["gt_lane_known"] = int(have.sum())
    out["agree_frac_known"] = float(np.mean(nl[have] == gl[have]))
    out["missing_from_nearby_known"] = int(np.sum(have & (nl == -9)))
    out["wrong_lane_known"] = int(np.sum(have & (nl != -9) & (nl != gl)))
    centred = have & (np.abs(gd) <= 1.25)
    out["agree_frac_centred"] = float(np.mean(nl[centred] == gl[centred]))
    out["n_centred"] = int(centred.sum())
    bnd = have & (np.abs(gd) > 1.25)
    out["n_boundary"] = int(bnd.sum())
    out["agree_frac_boundary"] = float(np.mean(nl[bnd] == gl[bnd])) if bnd.any() else None
    ext_only = have & (gl0 < 0)
    out["n_extension_zone_only"] = int(ext_only.sum())
    out["agree_frac_extension_zone"] = float(np.mean(nl[ext_only] == gl[ext_only])) if ext_only.any() else None
    nog = ~have
    out["n_no_gt_lane_within_120m"] = int(nog.sum())
    out["no_gt_lane_but_in_nearby"] = int(np.sum(nog & (nl != -9)))
    out["no_gt_lane_nearby_lanes"] = {str(int(v)): int(np.sum(nog & (nl == v))) for v in np.unique(nl[nog])}
    # confusion
    conf = {}
    for g in np.unique(gl[have]):
        for nn in np.unique(nl[have & (gl == g)]):
            conf["%d->%d" % (g, nn)] = int(np.sum(have & (gl == g) & (nl == nn)))
    out["confusion_gt->node(-9=absent)"] = conf
    wrong = have & (nl != gl)
    out["wrong_examples"] = [
        {"k": int(k[j]), "uid": int(u[j]), "gt_lane": int(gl[j]), "node_lane": int(nl[j]),
         "gt_d": float(gd[j]), "ds": float(dsp[j]), "s_on_gt_lane": float(ps[j])}
        for j in np.flatnonzero(wrong)[:15]]
    return out


def _near_impl_summary(rows):
    a = np.asarray(rows, float)
    gl, nl, gds, nds = a.T
    both = (gl >= 0) & (nl >= 0)
    return {"n": int(len(a)), "lane_equal_frac": float(np.mean(gl == nl)),
            "port_only": int(np.sum((gl >= 0) & (nl < 0))), "node_only": int(np.sum((gl < 0) & (nl >= 0))),
            "lane_diff_both": int(np.sum(both & (gl != nl))),
            "delta_s_abs_err_E2": bt.stats(np.abs(gds[both & (gl == nl)] - nds[both & (gl == nl)]))}


def _vel_summary(rows):
    out = {}
    for role in ROLES + ["nearby"]:
        a = np.asarray([r[2:] for r in rows if r[0] == role], float)
        if not len(a):
            out[role] = {"n": 0}
            continue
        (e_vx, e_vll, e_vlat, vlat_node, vlat_lane, e_ds1, e_ds2, e_s, e_front, e_rear, e_hs,
         e_yaw, v) = a.T
        ok = np.isfinite(e_vx)
        mv = ok & (v > 1.0)
        out[role] = {
            "n": int(len(a)), "uid_not_in_gt_frame": int(np.sum(~ok)),
            "v_long_minus_velx": bt.stats(np.abs(e_vx[ok])),
            "v_long_minus_lane_long": bt.stats(np.abs(e_vll[ok])),
            "v_lat_minus_vely": bt.stats(np.abs(e_vlat[ok])),
            "lane_lat_speed_abs": bt.stats(np.abs(vlat_lane[ok])),
            "node_v_lat_abs": bt.stats(np.abs(vlat_node[ok])),
            "frac_lane_lat_gt_0p25_node_lt_0p25": float(np.mean(
                (np.abs(vlat_lane[ok]) > 0.25) & (np.abs(vlat_node[ok]) < 0.25))),
            "delta_s_err_E1": bt.stats(e_ds1[ok]), "delta_s_abs_err_E1": bt.stats(np.abs(e_ds1[ok])),
            "delta_s_abs_err_E2": bt.stats(np.abs(e_ds2[ok])),
            "s_abs_err": bt.stats(np.abs(e_s[ok])),
            "front_delta_s_abs_err_E1": bt.stats(np.abs(e_front[ok])),
            "rear_delta_s_abs_err_E1": bt.stats(np.abs(e_rear[ok])),
            "half_s_abs_err": bt.stats(np.abs(e_hs[ok])),
            "yaw_abs_err_deg_moving": bt.stats(np.abs(e_yaw[mv])),
        }
    return out


def _classify(cases):
    cls = {}
    for c in cases:
        tags = []
        for tag in ("node", "gt"):
            v = c.get(tag)
            if v is None:
                continue
            if abs(v["d"]) > 1.0:
                tags.append("vehicle_near_boundary(|d|>1.0)")
            if v["s_on_lane"] < 0:
                tags.append("target_extension_zone(s<0)")
        if c["cur"] == 0 and c["cur_len"] - c["ego_s_cur"] < 40:
            tags.append("ego_near_lane0_end")
        if c["node"] is not None and c["gt"] is not None and abs(c["gap_node"] - c["gap_gt"]) < 1.0:
            tags.append("near_tie(<1 m)")
        if c["role"] != "current_front" and any(
                c.get(t) is not None and c[t]["rear"] <= 0 <= c[t]["front"] for t in ("node", "gt")):
            tags.append("overlap_with_ego")
        if c["node"] is None or c["gt"] is None:
            tags.append("role_empty_on_one_side")
        key = "+".join(sorted(set(tags))) or "unexplained"
        cls[key] = cls.get(key, 0) + 1
    return cls


def lidar_summary(arrs, obj):
    """C summaries. arrs[role]: rows (k, lidar_uid(-1 none,-2 ghost), gt_uid, gt_gap, win, md)."""
    out = {}
    for role in ROLES:
        a = arrs[role]
        o = {}
        for name, m in (("all", np.ones(len(a), bool)), ("lc_window", a[:, 4] > 0)):
            b = a[m]
            lu, gu, gap = b[:, 1], b[:, 2], b[:, 3]
            gv = gu >= 0
            bins = {}
            for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
                mm = gv & (gap >= lo) & (gap < hi)
                bins["%d-%d" % (lo, hi)] = {"n": int(mm.sum()),
                                            "det_correct": float(np.mean(lu[mm] == gu[mm])) if mm.any() else None}
            o[name] = {"n": int(len(b)), "role_agree": float(np.mean(lu == gu)) if len(b) else None,
                       "gt_valid": int(gv.sum()),
                       "det_correct_given_gt": float(np.mean(lu[gv] == gu[gv])) if gv.any() else None,
                       "det_any_valid_given_gt": float(np.mean(lu[gv] != -1)) if gv.any() else None,
                       "wrong_vehicle": int(np.sum(gv & (lu >= 0) & (lu != gu))),
                       "ghost": int(np.sum(lu == -2)), "missed": int(np.sum(gv & (lu == -1))),
                       "extra_when_gt_empty": int(np.sum(~gv & (lu != -1))),
                       "by_gt_gap": bins}
        out[role] = o
    return out


def pos_err_summary(pe):
    """LiDAR-vs-GT position error on the lane frame, binned by GT delta_s."""
    if not len(pe):
        return {}
    ds, derr, serr, length, v = pe.T
    edges = [-120, -90, -60, -40, -20, -5, 5, 20, 40, 60, 90, 120]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (ds >= lo) & (ds < hi)
        if m.sum() < 5:
            continue
        bins.append({"ds": "%d..%d" % (lo, hi), "n": int(m.sum()),
                     "d_err_med": float(np.median(derr[m])),
                     "d_err_p10": float(np.percentile(derr[m], 10)),
                     "d_err_p90": float(np.percentile(derr[m], 90)),
                     "s_err_med": float(np.median(serr[m]))})
    m = np.abs(ds) <= 100
    A = np.stack([np.ones(m.sum()), ds[m]], 1)
    # robust (iteratively trimmed) LSQ of lateral error vs distance
    w = np.ones(m.sum(), bool)
    coef = None
    for _ in range(3):
        coef = np.linalg.lstsq(A[w], derr[m][w], rcond=None)[0]
        r = derr[m] - A @ coef
        w = np.abs(r) < 3.0 * np.median(np.abs(r)) + 1e-6
    return {"n": int(len(pe)), "bins": bins, "fit_d_err": {"intercept_m": float(coef[0]),
            "slope_m_per_m": float(coef[1]), "slope_deg": float(np.degrees(np.arctan(coef[1])))}}


def obj_c_summary(role_cmp, gaps):
    """Same metrics for object mode (E1 GT)."""
    out = {}
    for role in ROLES:
        rows = role_cmp["E1"][role]
        a = np.asarray([(nu, gu, w) for _, nu, gu, w in rows], float)
        g = np.asarray(gaps[role], float)
        o = {}
        for name, m in (("all", np.ones(len(a), bool)), ("lc_window", a[:, 2] > 0)):
            b, gg = a[m], g[m]
            nu, gu = b[:, 0], b[:, 1]
            gv = gu >= 0
            bins = {}
            for lo, hi in zip(GAP_BINS[:-1], GAP_BINS[1:]):
                mm = gv & (gg >= lo) & (gg < hi)
                bins["%d-%d" % (lo, hi)] = {"n": int(mm.sum()),
                                            "det_correct": float(np.mean(nu[mm] == gu[mm])) if mm.any() else None}
            o[name] = {"n": int(len(b)), "role_agree": float(np.mean(nu == gu)),
                       "gt_valid": int(gv.sum()),
                       "det_correct_given_gt": float(np.mean(nu[gv] == gu[gv])) if gv.any() else None,
                       "by_gt_gap": bins}
        out[role] = o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", nargs="+")
    ap.add_argument("--out", default=os.path.join(gc.RESULTS, "gtcheck_a.json"))
    a = ap.parse_args()
    lanes = bt.load_lanes()
    allres = {"bags": {}}
    pooled_lidar = {r: [] for r in ROLES}
    pooled_obj = {"E1": {r: [] for r in ROLES}}
    pooled_gap = {r: [] for r in ROLES}
    worst = []
    pooled_pe = []
    for lab in a.labels:
        orig = os.path.expanduser("~/bags/lc_gt_%s.bag" % lab)
        rep = os.path.join(gc.REPLAY_DIR, lab + ".bag")
        print("[gtcheck_a] %s" % lab, file=sys.stderr)
        bc = BagCheck(orig, rep, lanes)
        r = bc.check_object_mode()
        worst += r.pop("worst_E1")
        la = bc.check_lidar_mode()
        r["lidar_C"] = lidar_summary(la, None)
        r["lidar_pos_err"] = pos_err_summary(la["pos_err"])
        pooled_pe.append(la["pos_err"])
        # object-mode gaps of the GT role for the C table
        gaps = {role: [] for role in ROLES}
        for role in ROLES:
            for (k, nu, gu, w) in bc.obj_roles["E1"][role]:
                gaps[role].append(np.nan)
        # recompute GT gaps for object mode rows (E1)
        G, R = bc.G, bc.R
        EP1 = bc.ego_proj(*bc.eg["E1"])
        for role in ROLES:
            gaps[role] = []
        for k in range(len(R["t"])):
            i = bc.idx[k]
            if i < 0:
                continue
            sl = slice(G.start[i], G.end[i])
            f = G.frame(i)
            cur, tgt = int(R["cur_lane"][k]), int(R["tgt_lane"][k])
            rr = gc.roles_from_proj(f, tuple(x[sl] for x in bc.P0[cur]), tuple(x[sl] for x in bc.P1[tgt]),
                                    EP1[(cur, 0)][0][k], EP1[(tgt, 1)][0][k])
            for role in ROLES:
                gaps[role].append(rr[role]["gap"] if rr[role]["uid"] >= 0 else np.nan)
        r["object_C"] = obj_c_summary(bc.obj_roles, gaps)
        for role in ROLES:
            pooled_lidar[role].append(la[role])
            pooled_obj["E1"][role] += bc.obj_roles["E1"][role]
            pooled_gap[role] += gaps[role]
        allres["bags"][lab] = r
    allres["pooled_lidar_C"] = lidar_summary({r: np.concatenate(pooled_lidar[r]) for r in ROLES}, None)
    allres["pooled_object_C"] = obj_c_summary(pooled_obj, pooled_gap)
    allres["pooled_lidar_pos_err"] = pos_err_summary(np.concatenate(pooled_pe))
    allres["worst_E1"] = sorted(worst, key=lambda c: -c["severity"])[:20]
    bt.write_json(a.out, allres)
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
