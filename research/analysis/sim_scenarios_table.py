#!/usr/bin/env python3
"""MORAI 시나리오를 시뮬레이터로(스폰·목표 속도 고정, 시드 6000–6049): 기존 vs 제안 표.

    python3 sim_scenarios_table.py          → markdown 표, results/final/scenarios.json
"""
import json
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parents[1] / "results" / "final"
SC = {"S1_normal": (70, 70), "S2_rear_accel": (54, 90), "S3_rear_decel": (72, 43), "S4_front_decel": (72, 29)}
for s in (30, 50, 70):
    for d in (40, 60, 80, 100):
        SC[f"G_s{s}_d{d}"] = (s, d)


def load(name, pl):
    p = RES / f"sim_scen_{name}_{pl}_n50.json"
    if not p.exists():
        return None
    return sorted(json.loads(p.read_text())["episodes"], key=lambda r: r["seed"])


def main():
    out = []
    print("| 시나리오 | 스폰→목표 (km/h) | 기존 성공 | 제안 성공 | 기존 충돌 | 제안 충돌 | 합류 시간 차이 (s, 실패 = 120 s) [95 % CI] |")
    print("|---|---|---|---|---|---|---|")
    for name, (s, d) in SC.items():
        b, t = load(name, "baseline"), load(name, "tree_gt_nogate")
        if b is None or t is None:
            continue
        cb = np.array([r["t_merge"] if r["success"] else 120.0 for r in b])
        ct = np.array([r["t_merge"] if r["success"] else 120.0 for r in t])
        diff = ct - cb
        rng = np.random.default_rng(0)
        m = rng.choice(diff, (10000, len(diff))).mean(axis=1)
        row = {"scenario": name, "spawn_kph": s, "desired_kph": d,
               "baseline_success": int(sum(r["success"] for r in b)), "tree_success": int(sum(r["success"] for r in t)),
               "baseline_collision": int(sum(r["collision"] for r in b)), "tree_collision": int(sum(r["collision"] for r in t)),
               "n": len(b), "diff_mean": float(diff.mean()), "lo": float(np.percentile(m, 2.5)),
               "hi": float(np.percentile(m, 97.5))}
        out.append(row)
        print(f"| {name} | {s}→{d} | {row['baseline_success']}/{row['n']} | {row['tree_success']}/{row['n']} | "
              f"{row['baseline_collision']} | {row['tree_collision']} | "
              f"{row['diff_mean']:+.1f} [{row['lo']:+.1f}, {row['hi']:+.1f}] |")
    (RES / "scenarios.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
