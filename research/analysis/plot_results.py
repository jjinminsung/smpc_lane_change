#!/usr/bin/env python3
"""README 그림: research/docs/fig/
  fig_sim_cdf.png   시뮬 1,000 시드: 시간 t 까지 합류를 끝낸 에피소드 비율(누적). 끝 높이 = 성공률
  fig_morai_runs.png MORAI 단계 7: 방법별 주행 24 회의 합류 시간(H 격자 · S1–S4), 실패는 오른쪽 끝 ×

    python3 analysis/plot_results.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs" / "fig"
FIN = ROOT / "results" / "final"

# 색은 방법(실체)에 고정. 범주 팔레트 순서 1–4 (dataviz reference palette, light).
METH = [  # (이름, 시뮬 파일, 색)
    ("새 구조 (반응형 트리)", "rx_mt", "#2a78d6"),
    ("새 구조 - 가지 없음", "rx_single", "#eb6834"),
    ("D 트리 + 옛 모니터", "tree_gt_nogate", "#1baf7a"),
    ("기존 SMPC", "baseline", "#eda100"),
]
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "Noto Sans CJK JP", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": SURF,
})


def sim_cdf():
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=160)
    tmax = 120.0
    ends = []
    for name, f, c in METH:
        eps = json.load(open(FIN / f"sim_{f}_n1000.json"))["episodes"]
        t = np.sort([e["t_merge"] for e in eps if e["success"]])
        y = np.arange(1, len(t) + 1) / len(eps) * 100
        x = np.concatenate([[0], t, [tmax]])
        y = np.concatenate([[0], y, [y[-1]]])
        ax.step(x, y, where="post", color=c, lw=2, label=name)
        ends.append([y[-1], y[-1], name])
    ends.sort(key=lambda e: -e[0])                  # 이름표가 겹치지 않게 위에서부터 최소 5 %p 간격
    for i in range(1, len(ends)):
        ends[i][1] = min(ends[i][1], ends[i - 1][1] - 5.0)
    for yv, ylab, name in ends:
        ax.annotate(f"{name}  {yv:.1f} %", (tmax, yv), xytext=(tmax + 2, ylab), va="center", fontsize=9, color=INK,
                    annotation_clip=False, arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8) if abs(ylab - yv) > 0.5 else None)
    ax.set_xlim(0, tmax)
    ax.set_ylim(0, 100)
    ax.set_xlabel("출발 뒤 시간 (s)")
    ax.set_ylabel("합류를 끝낸 에피소드 (%)")
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.set_title("시뮬레이터 1,000 시드: 시간별 누적 합류 비율 (끝 높이 = 성공률)", loc="left", fontsize=11, color=INK)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.subplots_adjust(right=0.70)
    fig.savefig(FIG / "fig_sim_cdf.png", bbox_inches="tight")
    plt.close(fig)


def morai_runs():
    R = {}
    for line in (ROOT / "results" / "morai_runs.jsonl").open():
        r = json.loads(line)
        R[(r["tag"].split("_")[0], r["phase"], r["scenario"], r["method"], r["rep"])] = r
    arms = {  # 단계 7 비교 묶음 (analysis/morai_step7.py 와 같음)
        "새 구조 (반응형 트리)": [("v11", "P", "rx_fix"), ("v11", "Q", "rx_fix")],
        "새 구조 - 가지 없음": [("v11", "P", "rx_single_fix"), ("v11", "Q", "rx_single_fix")],
        "D 트리 + 옛 모니터": [("v4", "H", "tree_nogate"), ("v3", "B", "tree_nogate")],
        "기존 SMPC": [("v4", "H", "baseline"), ("v2", "B", "baseline")],
    }
    FAIL_X = 190.0
    fig, ax = plt.subplots(figsize=(7.2, 3.6), dpi=160)
    rng = np.random.default_rng(1)
    for i, (name, _, c) in enumerate(METH):
        row = len(METH) - 1 - i
        for j, (pre, ph, m) in enumerate(arms[name]):
            runs = [r for k, r in R.items() if k[0] == pre and k[1] == ph and k[3] == m and "success" in (r.get("metrics") or {})]
            off = 0.14 if j == 0 else -0.14          # 위 = H 격자, 아래 = S1–S4
            for r in runs:
                mt = r["metrics"]
                yy = row + off + rng.uniform(-0.05, 0.05)
                if mt["success"]:
                    ax.scatter(mt["t_merge"], yy, s=26, color=c, edgecolor=SURF, linewidth=1, marker="o" if j == 0 else "s",
                               zorder=3)
                else:
                    ax.scatter(FAIL_X, yy, s=34, color=INK2, marker="x", linewidth=1.5, zorder=3)
        ok = sum(1 for pre, ph, m in arms[name] for k, r in R.items()
                 if k[0] == pre and k[1] == ph and k[3] == m and (r.get("metrics") or {}).get("success"))
        ax.text(203, row, f"{ok}/24", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(METH)))
    ax.set_yticklabels([n for n, _, _ in METH][::-1], color=INK)
    ax.axvline(180, color=GRID, lw=1)
    ax.set_xlim(0, 215)
    ax.set_xticks([0, 30, 60, 90, 120, 150, FAIL_X])
    ax.set_xticklabels(["0", "30", "60", "90", "120", "150", "실패"])
    ax.set_xlabel("합류 시간 (s)   ● H 격자 12 회   ■ S1–S4 × 3 회   × 실패(차선 끝 정체)")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("MORAI 주행별 합류 시간 (GT 인지, 오른쪽 숫자 = 성공 수)", loc="left", fontsize=11, color=INK)
    fig.savefig(FIG / "fig_morai_runs.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    sim_cdf()
    morai_runs()
    print("saved", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
