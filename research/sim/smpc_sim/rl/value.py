"""가치함수 V(s): 작은 MLP. 입력 특징(features.py) → 리턴 G 의 기댓값.

트리에서는 비용 단위로 바꿔 쓴다: 잎 비용 = -λ·V(잎). λ(value_weight) 는 '보상 1 이 비용 몇
점인가' 다. 기존 비용에서 복귀 한 번이 30 점이므로 λ 도 그 근처(10~60)에서 고른다.
"""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .features import DIM


class ValueNet(nn.Module):
    def __init__(self, dim=DIM, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def save_value(path, net, value_weight, meta=None):
    torch.save({"state_dict": net.state_dict(), "value_weight": value_weight, "meta": meta or {}}, path)


def load_value_fn(path, value_weight=None):
    """트리 플래너용: 특징 배열 [M, DIM] → λ·V [M] (numpy). CPU 에서 돈다(작은 망이라 충분)."""
    ck = torch.load(Path(path), map_location="cpu")
    net = ValueNet()
    net.load_state_dict(ck["state_dict"])
    net.eval()
    lam = ck["value_weight"] if value_weight is None else value_weight
    torch.set_num_threads(1)

    def fn(features):
        with torch.no_grad():
            return lam * net(torch.from_numpy(np.asarray(features, dtype=np.float32))).numpy()
    fn.value_weight = lam
    return fn
