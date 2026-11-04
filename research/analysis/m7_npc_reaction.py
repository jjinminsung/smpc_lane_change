#!/usr/bin/env python3
"""M7. MORAI NPC 뒤차가 끼어드는 자차에 어떻게 반응하는가 (plan_v2 단계 5).

    python3 m7_npc_reaction.py --bags "~/bags/research/v5_N_*.bag"     → 표 + results/m7_npc_reaction.json

사건: 차선 변경 활성 구간마다, 자차 차체가 목표 차선 경계를 넘은 첫 시각 t0 (침범 깊이 > 0)의 목표 차선 뒤차.
GT /Object_topic 의 acceleration.x(m/s², 속도 미분과 기울기 1.05)로
  onset   : t0 − 3 s 이후 처음으로 가속도 < −0.5 m/s² 가 0.3 s 이어진 시각, 그때의 침범 깊이와 t0·t_0.3m 대비 지연
  a_min   : t0 ~ t0 + 5 s 최소 가속도(0.25 s 이동 중앙값)
  need    : 뒤차가 감속하지 않으면 TTC < 4 s (자차 뒤 범퍼까지 간격 / 접근 속도)
  no_react: need 인데 t0 + 3 s 까지 onset 없음
  contact : /CollisionData 또는 GT 겹침(analysis/morai_metrics 의 접촉)
"""
import json
import math

import numpy as np

from bagtools import RESULTS_DIR, base_argparser, expand_bags, gt_tracks, load_bags, status_intervals, write_json

LANE_W, EGO_HALF_W, EGO_REAR = 3.5, 0.946, 0.790


def med(t, x, win=0.25):
    if len(t) < 5:
        return x
    dt = max(np.median(np.diff(t)), 1e-3)
    h = max(1, int(round(0.5 * win / dt)))
    return np.array([np.median(x[max(0, i - h):i + h + 1]) for i in range(len(x))])


def onset_time(t, a, t_from, t_to, thr=-0.5, hold=0.3):
    m = (t >= t_from) & (t <= t_to)
    tt, aa = t[m], a[m]
    for i in range(len(tt)):
        j = np.searchsorted(tt, tt[i] + hold)
        if j - i >= 2 and np.all(aa[i:j] < thr):
            return float(tt[i])
    return None


def events(d):
    tr = gt_tracks(d)
    tg = d["targets"]
    rr = tg["target_rear"]
    intr_all = 0.5 * LANE_W + EGO_HALF_W - np.abs(tg["ego_d_tgt"])
    out = []
    for iv in status_intervals(d):
        m = np.flatnonzero((tg["t"] >= iv["t_start"]) & (tg["t"] <= iv["t_end"]))
        if len(m) == 0:
            continue
        cross = m[intr_all[m] > 0.0]
        if len(cross) == 0:
            continue
        k0 = cross[0]
        t0 = float(tg["t"][k0])
        k03 = next((k for k in m if intr_all[k] >= 0.3), None)
        t03 = float(tg["t"][k03]) if k03 is not None else math.nan
        if rr["valid"][k0] < 0.5 or int(rr["uid"][k0]) not in tr:
            continue
        uid = int(rr["uid"][k0])
        track = tr[uid]
        t = track["t"]
        a = med(t, track["ax"])
        v = med(t, track["v"])
        v_r = float(np.interp(t0, t, v))
        v_e = float(np.interp(t0, d["odom"]["t"], d["odom"]["v"]))
        gap = float(-rr["delta_s"][k0] - rr["half_s"][k0] - EGO_REAR)
        closing = v_r - v_e
        ttc = gap / closing if closing > 0.1 else math.inf
        on = onset_time(t, a, t0 - 3.0, t0 + 5.0)
        post = (t >= t0) & (t <= t0 + 5.0)
        intr_on = float(np.interp(on, tg["t"], intr_all)) if on is not None else math.nan
        need = ttc < 4.0
        out.append({"bag": d["label"], "t0": round(t0 - float(tg["t"][0]), 2), "src": iv["src"], "tgt": iv["tgt"],
                    "uid": uid, "gap": round(gap, 1), "v_rear": round(v_r, 2), "v_ego": round(v_e, 2),
                    "closing": round(closing, 2), "ttc": round(ttc, 2) if math.isfinite(ttc) else None,
                    "need": bool(need), "onset_rel_t0": round(on - t0, 2) if on is not None else None,
                    "onset_rel_t03": round(on - t03, 2) if on is not None and math.isfinite(t03) else None,
                    "intrusion_at_onset": round(intr_on, 2) if math.isfinite(intr_on) else None,
                    "a_min": round(float(a[post].min()), 2) if post.any() else None,
                    "dv5": round(float(v[post].min()) - v_r, 2) if post.any() else None,
                    "no_react": bool(need and (on is None or on > t0 + 3.0))})
    return out


def main():
    ap = base_argparser(__doc__)
    a = ap.parse_args()
    ds = load_bags(expand_bags(a.bags), cache_dir=a.cache, use_cache=not a.no_cache, jobs=a.jobs)
    ev = [e for d in ds for e in events(d)]
    need = [e for e in ev if e["need"]]
    print(f"사건 {len(ev)}, 양보가 필요했던 사건(감속 안 하면 TTC < 4 s) {len(need)}")

    def q(xs, p):
        xs = [x for x in xs if x is not None]
        return round(float(np.percentile(xs, p)), 2) if xs else None
    rows = []
    for name, sub in (("양보 필요", need), ("필요 없음", [e for e in ev if not e["need"]])):
        if not sub:
            continue
        r = {"group": name, "n": len(sub), "no_react": sum(e["no_react"] for e in sub),
             "onset_rel_t0_p50": q([e["onset_rel_t0"] for e in sub], 50),
             "onset_rel_t03_p50": q([e["onset_rel_t03"] for e in sub], 50),
             "intrusion_at_onset_p50": q([e["intrusion_at_onset"] for e in sub], 50),
             "a_min_p50": q([e["a_min"] for e in sub], 50), "a_min_p10": q([e["a_min"] for e in sub], 10)}
        rows.append(r)
        print(f"| {name} | n {r['n']} | 무반응 {r['no_react']} | 감속 시작(경계 넘은 뒤) 중앙 {r['onset_rel_t0_p50']} s | "
              f"(0.3 m 뒤) {r['onset_rel_t03_p50']} s | 시작 때 침범 깊이 {r['intrusion_at_onset_p50']} m | "
              f"최소 가속도 중앙 {r['a_min_p50']}, p10 {r['a_min_p10']} m/s² |")
    for e in sorted(need, key=lambda e: e["ttc"] or 99)[:15]:
        print("  ", e)
    write_json(f"{RESULTS_DIR}/m7_npc_reaction.json", {"definition": __doc__, "summary": rows, "events": ev})


if __name__ == "__main__":
    main()
