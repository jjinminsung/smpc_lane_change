#!/usr/bin/env python3
"""M5. 목표 차선 뒤차(MORAI NPC)가 자차 진입에 양보(감속)하는지, 언제·얼마나.

    python3 m5_rear_yield.py --bags "~/bags/research/v[234]_*.bag"      → 표 + results/m5_rear_yield.json

사건: /smpc/lane_change_status 활성 구간마다 자차 차체가 목표 차선에 0.3 m 들어간 첫 시각 t_in
(|ego_d_target| ≤ 차선 폭/2 + 자차 폭/2 − 0.3). 그 시각의 target_rear(/smpc/targets) 를 GT 트랙에서 따라가
  v0      : t_in 전 2 s 속도 중앙값
  drop    : t_in 뒤 4 s 안 최대 속도 감소 v0 − min v (0.5 s 이동 중앙값 속도)
  t_react : v0 − 0.5 m/s 아래로 처음 내려간 지연
대조군: 같은 bag 에서 자차 진입이 없던 NPC 의 아무 4 s 창에서 같은 drop(평소 감속 빈도).
"""
import json
import math
import sys

import numpy as np

from bagtools import RESULTS_DIR, base_argparser, expand_bags, gt_tracks, load_bags, status_intervals, write_json

EGO_HALF_W = 0.946
LANE_W = 3.5
ENTRY_D = 0.5 * LANE_W + EGO_HALF_W - 0.3


def med_speed(t, v, win=0.5):
    """0.5 s 이동 중앙값 속도. MORAI GT 속도는 프레임 단위로 튀어서 미분하면 쓸 수 없다."""
    if len(t) < 5:
        return v
    dt = max(np.median(np.diff(t)), 1e-3)
    h = max(1, int(round(0.5 * win / dt)))
    return np.array([np.median(v[max(0, i - h):i + h + 1]) for i in range(len(v))])


def drop_after(t, vm, t0, v0, horizon=4.0):
    """t0 뒤 horizon 안 최대 속도 감소(v0 − 최소)와 v0 − 0.5 m/s 아래로 처음 내려간 지연."""
    m = (t >= t0) & (t <= t0 + horizon)
    if m.sum() < 10:
        return None, None
    below = np.flatnonzero(vm[m] < v0 - 0.5)
    return float(v0 - vm[m].min()), (float(t[m][below[0]] - t0) if len(below) else None)


def events(d):
    tr = gt_tracks(d)
    tg = d["targets"]
    out = []
    for iv in status_intervals(d):
        m = (tg["t"] >= iv["t_start"]) & (tg["t"] <= iv["t_end"])
        idx = np.flatnonzero(m & (np.abs(tg["ego_d_tgt"]) <= ENTRY_D))
        if len(idx) == 0:
            continue
        k = idx[0]
        t_in = float(tg["t"][k])
        rr = tg["target_rear"]
        if rr["valid"][k] < 0.5:
            continue
        uid = int(rr["uid"][k])
        if uid not in tr:
            continue
        track = tr[uid]
        t = track["t"]
        vs = med_speed(t, track["v"])
        pre = (t >= t_in - 2.0) & (t <= t_in)
        if pre.sum() < 5:
            continue
        v0 = float(np.median(vs[pre]))
        drop, t_react = drop_after(t, vs, t_in, v0)
        if drop is None:
            continue
        ego_v = float(np.interp(t_in, d["odom"]["t"], d["odom"]["v"]))
        k0 = np.flatnonzero(m)[0]
        ego_v_accept = float(np.interp(tg["t"][k0], d["odom"]["t"], d["odom"]["v"]))
        gap = float(-rr["delta_s"][k] - rr["half_s"][k] - 0.790)    # 자차 뒤 범퍼(후륜축 뒤 0.79 m)까지
        out.append({"bag": d["label"], "t_in": round(t_in - float(tg["t"][0]), 2), "src": iv["src"], "tgt": iv["tgt"],
                    "uid": uid, "ego_v": round(ego_v, 2), "rear_v0": round(v0, 2), "closing": round(v0 - ego_v, 2),
                    "gap": round(gap, 1), "ego_v_accept": round(ego_v_accept, 2), "drop": round(drop, 2),
                    "t_react": t_react})
    # 대조군: 진입 사건의 NPC 를 뺀 트랙의 5 s 창
    used = {e["uid"] for e in out}
    ctrl = []
    for uid, track in tr.items():
        if uid in used or len(track["t"]) < 100:
            continue
        t = track["t"]
        vs = med_speed(t, track["v"])
        for t0 in np.arange(t[0] + 2.0, t[-1] - 4.0, 4.0):
            pre = (t >= t0 - 2.0) & (t <= t0)
            if pre.sum() < 5:
                continue
            v0 = float(np.median(vs[pre]))
            if v0 < 3.0:
                continue
            dr, _ = drop_after(t, vs, t0, v0)
            if dr is not None:
                ctrl.append(dr)
    return out, ctrl


def main():
    ap = base_argparser(__doc__)
    a = ap.parse_args()
    paths = expand_bags(a.bags)
    ds = load_bags(paths, cache_dir=a.cache, use_cache=not a.no_cache, jobs=a.jobs)
    ev, ctrl = [], []
    for d in ds:
        e, c = events(d)
        ev += e
        ctrl += c
    ctrl = np.asarray(ctrl)

    def share(x, thr):
        return float(np.mean(np.asarray(x) <= thr)) if len(x) else math.nan

    def share(x, thr):
        return float(np.mean(np.asarray(x) >= thr)) if len(x) else math.nan

    print(f"사건 {len(ev)}, 대조 창 {len(ctrl)} (진입 없는 NPC 의 아무 4 s 창, 평소 감속 빈도)")
    print("| 구간 | n | 4 s 안 1 m/s 이상 감속 | 3 m/s 이상 | 0.5 m/s 감속까지 지연 중앙값 (s) | 감속량 중앙값 (m/s) |")
    print("|---|---|---|---|---|---|")
    bins = [("대조군(진입 없음)", None), ("간격 < 15 m", lambda e: e["gap"] < 15), ("15–30 m", lambda e: 15 <= e["gap"] < 30),
            ("30–50 m", lambda e: 30 <= e["gap"] < 50), ("≥ 50 m", lambda e: e["gap"] >= 50),
            ("뒤차가 2 m/s 이상 빠름", lambda e: e["closing"] > 2.0),
            ("정지·저속 출발(승인 때 < 1 m/s)", lambda e: e["ego_v_accept"] < 1.0)]
    rows = []
    for name, f in bins:
        if f is None:
            dr, rt = list(ctrl), []
        else:
            sub = [e for e in ev if f(e)]
            dr, rt = [e["drop"] for e in sub], [e["t_react"] for e in sub if e["t_react"] is not None]
        if not dr:
            continue
        row = {"bin": name, "n": len(dr), "share_drop_ge_1": share(dr, 1.0), "share_drop_ge_3": share(dr, 3.0),
               "react_median": float(np.median(rt)) if rt else None, "drop_median": float(np.median(dr))}
        rows.append(row)
        rm = "–" if row["react_median"] is None else f"{row['react_median']:.2f}"
        print(f"| {name} | {len(dr)} | {100*row['share_drop_ge_1']:.0f} % | {100*row['share_drop_ge_3']:.0f} % | {rm} | "
              f"{row['drop_median']:.2f} |")
    write_json(f"{RESULTS_DIR}/m5_rear_yield.json", {"definition": __doc__, "entry_d_m": ENTRY_D, "bins": rows,
                                                    "control_n": int(len(ctrl)), "events": ev})


if __name__ == "__main__":
    main()
