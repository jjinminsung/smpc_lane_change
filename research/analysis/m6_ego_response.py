#!/usr/bin/env python3
"""M6. 자차 종방향 응답 (plan_v2 단계 1a, 1b).

MPC 가 따라가는 기준 속도 v_ref = min(미션 속도, /smpc/target_speed_mps_gated) 와 실제 속도(/Competition_topic).

    python3 m6_ego_response.py --bags "~/bags/research/v3_*.bag" "~/bags/research/v4_H_*.bag"

1) 시뮬 EgoPlant 종방향(1 차 지연 τ + 기준 변화율 앞먹임, 가속·감속·저크 한계)을 기록된 v_ref 로 열린 루프로 굴려
   1/2/4/8 s 뒤 속도·거리 오차를 잰다. 기준 속도는 기록값(정답)을 넣으므로 제어기 응답만 본다.
2) τ, 가속 한계, 감속 한계를 격자로 맞춘다(2·4·8 s 거리 오차 합 최소).
3) 정지(< 0.5 m/s)에서 차선 변경이 시작될 때 미션 속도가 어떻게 회복되는지 잰다.
결과: results/m6_ego_response.json
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")
DT = 0.05
HOR = (1.0, 2.0, 4.0, 8.0)
TOPICS = ["/Competition_topic", "/smpc/target_speed_mps_gated",
          "/control_feedback/mission_speed_target_mps", "/smpc/lane_change_status"]


def read(path):
    import rosbag
    out = {"ego": [], "gated": [], "mission": [], "status": []}
    with rosbag.Bag(path) as b:
        for top, m, t in b.read_messages(topics=TOPICS):
            ts = t.to_sec()
            if top == TOPICS[0]:
                out["ego"].append((ts, math.hypot(m.velocity.x, m.velocity.y)))
            elif top == TOPICS[1]:
                out["gated"].append((ts, m.data))
            elif top == TOPICS[2]:
                out["mission"].append((ts, m.data))
            else:
                out["status"].append((ts, float(m.active)))
    return {k: np.asarray(v, float).reshape(-1, 2) for k, v in out.items()}


def zoh(tq, src, default=np.nan):
    if len(src) == 0:
        return np.full(len(tq), default)
    i = np.searchsorted(src[:, 0], tq, side="right") - 1
    v = src[np.clip(i, 0, len(src) - 1), 1]
    return np.where(i >= 0, v, default)


def series(d):
    if len(d["ego"]) < 200 or len(d["mission"]) < 10:
        return None
    t = np.arange(d["ego"][0, 0] + 1.0, d["ego"][-1, 0], DT)
    v = np.interp(t, d["ego"][:, 0], d["ego"][:, 1])
    mis = zoh(t, d["mission"])
    gat = zoh(t, d["gated"], np.inf) if len(d["gated"]) else np.full(len(t), np.inf)
    ref = np.fmin(mis, gat)
    lc = zoh(t, d["status"], 0.0)
    ok = np.isfinite(ref)
    return {"t": t[ok], "v": v[ok], "ref": ref[ok], "mission": mis[ok], "lc": lc[ok]}


def plant(v0, a0, ref, tau, a_max, a_min, a_stop=-9.0, jerk=12.0):
    """sim/smpc_sim/plant.py EgoPlant.step 의 종방향과 같은 식."""
    v, a, prev = v0, a0, ref[0]
    vs = np.empty(len(ref))
    ss = np.empty(len(ref))
    s = 0.0
    for k, r in enumerate(ref):
        ff = (r - prev) / DT
        prev = r
        a_des = (r - v) / tau + ff
        a_des = min(a_max, max(a_stop if r < 0.01 else a_min, a_des))
        a += max(-jerk * DT, min(jerk * DT, a_des - a))
        if v <= 0.0 and a < 0.0:
            a = 0.0
        vn = max(0.0, v + a * DT)
        s += 0.5 * (v + vn) * DT
        v = vn
        vs[k], ss[k] = v, s
    return vs, ss


def windows(S, step=1.0):
    n = int(HOR[-1] / DT)
    for s in S:
        for i in range(20, len(s["t"]) - n, int(step / DT)):
            yield s, i, n


def evaluate(W, tau, a_max, a_min):
    err = {h: {"dv": [], "ds": []} for h in HOR}
    for s, i, n in W:
        a0 = (s["v"][i] - s["v"][i - 4]) / (4 * DT)
        vp, sp = plant(s["v"][i], a0, s["ref"][i:i + n], tau, a_max, a_min)
        va = s["v"][i:i + n]
        sa = np.cumsum(0.5 * (va + np.concatenate([[s["v"][i]], va[:-1]])) * DT)
        for h in HOR:
            k = int(round(h / DT)) - 1
            err[h]["dv"].append(vp[k] - va[k])
            err[h]["ds"].append(sp[k] - sa[k])
    return err


def summarize(err):
    return {f"{h:.0f}s": {"dv_mae": round(float(np.mean(np.abs(e["dv"]))), 3),
                          "ds_mae": round(float(np.mean(np.abs(e["ds"]))), 3),
                          "ds_p90": round(float(np.percentile(np.abs(e["ds"]), 90)), 3), "n": len(e["ds"])}
            for h, e in err.items()}


def standstill_switch(S):
    rows = []
    n8 = int(8.0 / DT)
    for s in S:
        on = np.flatnonzero((s["lc"][1:] > 0.5) & (s["lc"][:-1] < 0.5)) + 1
        for i in on:
            if s["v"][i] > 0.5 or i + n8 >= len(s["t"]):
                continue
            m = s["mission"][i:i + n8]
            rows.append({k: round(float(x), 2) for k, x in (
                ("m0", m[0]), ("m2", m[int(2 / DT)]), ("m4", m[int(4 / DT)]), ("m8", m[-1]),
                ("v2", s["v"][i + int(2 / DT)]), ("v4", s["v"][i + int(4 / DT)]), ("v8", s["v"][i + n8 - 1]))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bags", nargs="+", required=True)
    ap.add_argument("--step", type=float, default=2.0, help="창 시작 간격 (s)")
    a = ap.parse_args()
    paths = sorted({p for g in a.bags for p in glob.glob(os.path.expanduser(g))})
    S = []
    for p in paths:
        try:
            s = series(read(p))
        except Exception as e:                                # noqa: BLE001
            print("skip", os.path.basename(p), e, file=sys.stderr)
            continue
        if s is not None:
            S.append(s)
    W = list(windows(S, a.step))
    print(f"bags {len(S)}, 창 {len(W)}")
    base = summarize(evaluate(W, 0.7, 3.0, -6.0))
    print("지금 시뮬 모델 τ 0.7, 가속 ≤ 3.0, 감속 ≤ 6.0:", json.dumps(base))
    best = None
    for tau in (0.4, 0.7, 1.0, 1.5, 2.0):
        for a_max in (1.5, 2.0, 2.5, 3.0):
            for a_min in (-3.0, -4.5, -6.0):
                sm = summarize(evaluate(W, tau, a_max, a_min))
                score = sm["2s"]["ds_mae"] + sm["4s"]["ds_mae"] + sm["8s"]["ds_mae"]
                if best is None or score < best[0]:
                    best = (score, tau, a_max, a_min, sm)
    print(f"맞춘 값 τ {best[1]}, 가속 ≤ {best[2]}, 감속 ≤ {-best[3]}:", json.dumps(best[4]))
    ss = standstill_switch(S)
    if ss:
        med = {k: float(np.median([r[k] for r in ss])) for k in ss[0]}
        print(f"정지 출발 차선 변경 {len(ss)} 건 중앙값:", json.dumps(med))
    json.dump({"definition": __doc__, "n_bags": len(S), "n_windows": len(W), "current_sim_model": base,
               "fit": {"tau": best[1], "a_max": best[2], "a_min": best[3], "err": best[4]},
               "standstill_switch": ss}, open(os.path.join(RES, "m6_ego_response.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
