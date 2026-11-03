"""트리 잎의 가치함수 V 를 근사 정책 반복(approximate policy iteration)으로 학습한다.

    python3 -m smpc_sim.rl.train_value --iters 3 --episodes 300      (research/sim 에서 실행)

한 바퀴:
  1) 평가(policy evaluation): 지금 정책 π_k (= 트리 + V_k) 로 에피소드를 돌려
     상태 s_t 와 그 뒤 실제로 받은 리턴 G_t 를 모은다.  V_0 = 0 (V 없는 트리).
  2) 회귀: V_{k+1}(s) ≈ E[G | s] 를 MSE 로 맞춘다.  지난 바퀴 데이터도 함께 쓴다(데이터 재사용).
  3) 개선(policy improvement): 트리가 잎에서 V_{k+1} 을 쓰면 '8 초 뒤 상태의 좋고 나쁨'을 보고
     루트 행동을 고르게 된다. 이것이 새 정책 π_{k+1}.
몬테카를로 리턴을 쓰므로 편향은 없고 분산은 크다. 에피소드가 짧아(≤120 s) 충분히 다룰 수 있다.

데이터 분리: 학습 시드 10000+, λ 선택(검증) 시드 5000+, 최종 비교 시드 2000+ (다른 방법과 같은 시드).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve()
SIM_DIR = HERE.parents[2]
sys.path.insert(0, str(SIM_DIR))

from run_batch import run_one, summarize  # noqa: E402
from smpc_sim.rl.features import DIM  # noqa: E402
from smpc_sim.rl.reward import RewardConfig, returns_for_samples  # noqa: E402
from smpc_sim.rl.value import ValueNet, save_value  # noqa: E402

OUT = SIM_DIR.parent / "results" / "value"
BASE = "tree_gt_nogate"


def planner_name(path, lam):
    return BASE if path is None else f"{BASE}_V:{path}:{lam}"


def collect(pool, name, seeds):
    over = {"collect_samples": True}
    return pool.map(run_one, [(name, s, over) for s in seeds])


def to_dataset(episodes, rc):
    X, G, ep_id = [], [], []
    for i, ep in enumerate(episodes):
        if not ep["samples"]:
            continue
        g = returns_for_samples(ep, rc)
        X += [f for _, f in ep["samples"]]
        G += g
        ep_id += [i] * len(g)
    return np.asarray(X, np.float32), np.asarray(G, np.float32), np.asarray(ep_id)


def fit(X, G, ep_id, epochs=80, seed=0):
    """에피소드 단위로 검증 집합을 떼어 과적합을 본다(같은 에피소드 샘플은 서로 닮았다)."""
    rng = np.random.default_rng(seed)
    eps = np.unique(ep_id)
    val_eps = set(rng.choice(eps, size=max(1, len(eps) // 10), replace=False).tolist())
    val = np.array([e in val_eps for e in ep_id])
    torch.manual_seed(seed)
    net = ValueNet()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-5)
    xt, gt = torch.from_numpy(X[~val]), torch.from_numpy(G[~val])
    xv, gv = torch.from_numpy(X[val]), torch.from_numpy(G[val])
    best, best_state = np.inf, None
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        for k in range(0, len(xt), 512):
            b = perm[k:k + 512]
            loss = nn.functional.mse_loss(net(xt[b]), gt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = nn.functional.mse_loss(net(xv), gv).item()
        if vl < best:
            best, best_state = vl, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    with torch.no_grad():
        base = float(((gv - gt.mean()) ** 2).mean())     # 상수 예측(평균)의 MSE 와 비교
    return net, {"val_mse": best, "const_mse": base, "n_train": int((~val).sum()), "n_val": int(val.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--val_episodes", type=int, default=100)
    ap.add_argument("--test_episodes", type=int, default=200)
    ap.add_argument("--lambdas", default="10,30,60")
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rc = RewardConfig()
    lambdas = [float(x) for x in a.lambdas.split(",")]
    import multiprocessing as mp
    log = {"reward": rc.__dict__, "iters": []}
    data = []
    path, lam = None, None
    with mp.Pool(a.workers) as pool:
        for k in range(a.iters):
            t0 = time.time()
            seeds = list(range(10000 + k * a.episodes, 10000 + (k + 1) * a.episodes))
            eps = collect(pool, planner_name(path, lam), seeds)
            data.append(to_dataset(eps, rc))
            X = np.concatenate([d[0] for d in data])
            G = np.concatenate([d[1] for d in data])
            E = np.concatenate([d[2] + 100000 * i for i, d in enumerate(data)])
            net, stats = fit(X, G, E, seed=k)
            # λ 선택: 검증 시드에서 성공률 → 합류 시간 순으로 가장 좋은 λ
            best = None
            for lam_try in lambdas:
                p_try = OUT / f"V_iter{k + 1}.pt"
                save_value(p_try, net, lam_try, {"iter": k + 1, **stats})
                val = pool.map(run_one, [(planner_name(p_try, lam_try), s, {}) for s in
                                         range(5000, 5000 + a.val_episodes)])
                sm = summarize(val)
                key = (sm["success_%"] - 10 * sm["collision_%"], -sm["merge_time_median_s"])
                if best is None or key > best[0]:
                    best = (key, lam_try, sm)
            lam = best[1]
            path = OUT / f"V_iter{k + 1}.pt"
            save_value(path, net, lam, {"iter": k + 1, **stats})
            entry = {"iter": k + 1, "train_policy": summarize(eps), "fit": stats, "lambda": lam,
                     "val": best[2], "wall_s": round(time.time() - t0, 1)}
            log["iters"].append(entry)
            print(json.dumps(entry, indent=1), flush=True)
        test = pool.map(run_one, [(planner_name(path, lam), s, {}) for s in range(2000, 2000 + a.test_episodes)])
    log["test"] = summarize(test)
    log["final"] = {"path": str(path), "lambda": lam}
    print(json.dumps(log["test"], indent=1))
    (OUT / "train_log.json").write_text(json.dumps(log, indent=1))
    (SIM_DIR.parent / "results" / f"sim_{BASE}_V_n{a.test_episodes}.json").write_text(
        json.dumps({"summary": log["test"], "planner": planner_name(path, lam), "episodes": test}))


if __name__ == "__main__":
    main()
