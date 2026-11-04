#!/usr/bin/env python3
"""MORAI 실험 행렬 실행기. (시나리오 × 방법 × 반복) 을 차례로 돌리고 지표를 jsonl 로 쌓는다.

    python3 run_matrix.py --phase A        # S2 안전 확인: 트리(게이트 유지/정리)
    python3 run_matrix.py --phase B        # 문서 계획: S1–S4 × {기존, 트리+게이트 정리} × 3
    python3 run_matrix.py --phase C        # 속도 격자 12 개 × {기존, 트리+게이트 정리} × 2
    python3 run_matrix.py --phase E        # 2×2: S1–S4 × {기존, A, B, C, D} × 3 (같은 세션에서 번갈아)
    python3 run_matrix.py --phase H        # 2×2: highway_spawn50 속도 격자 12 개 × {기존, A, B, C, D} × 1

2×2 방법(모두 게이트 정리): baseline_nogate = A 기존 판단 + rear_abort, bdec = B 기존 판단 + 트리 모니터,
tree_nomon = C 트리 판단 + rear_abort, tree_nogate = D 트리 판단 + 트리 모니터(제안).

이미 끝난 (tag) 는 건너뛴다(중단 후 다시 실행해도 된다). 방법 순서는 반복마다 번갈아
(기존→트리, 트리→기존) 시간대 차이(교통 상황)를 줄인다.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = Path.home() / "bags" / "research"
LOG = REPO / "research" / "results" / "morai_runs.jsonl"
SHOT = Path("/tmp") / "morai_shots"

# 방법 → (research_morai.launch method, tree_keep_gates, 추가 roslaunch 인자)
METHODS = {
    "baseline": ("baseline", "true", []),
    "tree": ("tree", "true", []),
    "tree_nogate": ("tree", "false", []),
    "baseline_nogate": ("baseline", "true", ["baseline_keep_gates:=false"]),
    "bdec": ("baseline", "true", ["baseline_keep_gates:=false", "monitor:=true"]),
    "tree_nomon": ("tree", "false", ["monitor:=false"]),
    "rx": ("tree", "false", ["reactive:=true"]),                               # plan_v2 새 구조
    "rx_single": ("tree", "false", ["reactive:=true", "reactive_single:=true"]),
}
S = ["research_S1_normal", "research_S2_rear_accel", "research_S3_rear_decel", "research_S4_front_decel"]
G = [f"research_G_s{s}_d{d}" for s in (30, 50, 70) for d in (40, 60, 80, 100)]
H = [f"research_H_s{s}_d{d}" for s in (30, 50, 70) for d in (40, 60, 80, 100)]   # make_scenarios.py
M2X2 = ["baseline", "baseline_nogate", "bdec", "tree_nomon", "tree_nogate"]
PHASES = {
    "A": ([S[1]], ["tree", "tree_nogate"], 3),
    "B": (S, ["baseline", "tree_nogate"], 3),
    "C": (G, ["baseline", "tree_nogate"], 2),
    "D": (S, ["tree"], 2),
    "E": (S, M2X2, 3),
    "H": (H, M2X2, 1),
}


def done_tags():
    if not LOG.exists():
        return set()
    return {json.loads(l)["tag"] for l in LOG.read_text().splitlines() if l.strip()}


def run(cmd, timeout):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout[-2000:], p.stderr[-2000:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=sorted(PHASES))
    ap.add_argument("--max_runs", type=int, default=999)
    ap.add_argument("--prefix", default="v2_", help="태그 접두사. v2_ = GT 인지 수정(중복 프레임·기준점) 뒤 주행")
    ap.add_argument("--methods", default="", help="쉼표 목록으로 단계의 방법을 바꾼다(예: tree_nogate)")
    ap.add_argument("--reps", type=int, default=0, help="반복 수를 바꾼다")
    ap.add_argument("--start_rep", type=int, default=0)
    a = ap.parse_args()
    scenarios, methods, reps = PHASES[a.phase]
    if a.methods:
        methods = a.methods.split(",")
    if a.reps:
        reps = a.reps
    SHOT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    done = done_tags()
    n = 0
    for rep in range(a.start_rep, reps):
        for sc in scenarios:
            order = methods if rep % 2 == 0 else list(reversed(methods))
            for m in order:
                tag = f"{a.prefix}{a.phase}_{sc.replace('research_', '')}_{m}_r{rep}"
                if tag in done:
                    continue
                if n >= a.max_runs:
                    return
                n += 1
                t0 = time.time()
                rc, so, se = run(["python3", str(HERE / "load_scenario.py"), sc], 180)
                if rc != 0:
                    rc, so, se = run(["python3", str(HERE / "load_scenario.py"), sc], 180)
                if rc != 0:
                    rec = {"tag": tag, "phase": a.phase, "scenario": sc, "method": m, "rep": rep,
                           "error": "load_scenario failed", "out": so + se}
                    LOG.open("a").write(json.dumps(rec) + "\n")
                    print(json.dumps(rec), flush=True)
                    continue
                method, keep, extra = METHODS[m]
                rc, so, se = run(["bash", str(HERE / "run_episode.sh"), tag, method, keep] + extra, 420)
                mon = None
                try:
                    mon = json.loads((OUT / f"{tag}.monitor.json").read_text().strip().splitlines()[-1])
                except Exception:
                    pass
                met = None
                bag = OUT / f"{tag}.bag"
                if bag.exists():
                    r2, so2, se2 = run(["python3", str(REPO / "research" / "analysis" / "morai_metrics.py"), str(bag)], 600)
                    try:
                        met = json.loads(so2.strip().splitlines()[-1])
                    except Exception:
                        met = {"error": se2[-500:]}
                rec = {"tag": tag, "phase": a.phase, "scenario": sc, "method": m, "rep": rep,
                       "monitor": mon, "metrics": met, "wall_s": round(time.time() - t0, 1)}
                LOG.open("a").write(json.dumps(rec) + "\n")
                print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
