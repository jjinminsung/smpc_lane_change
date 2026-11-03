"""보상 함수와 리턴(누적 보상) 계산.

강화학습의 '보상'은 우리가 원하는 결과를 숫자로 적은 것이다. 계획 문서의 초안을 따른다.
  합류 성공(차선 3 도착)      +1
  실패(시간 초과·차선 끝 이탈)  -1
  충돌                         -10
  차선 끝 정지(한 번)          -0.2   (실패의 전조를 일찍 알려 주는 보조 보상)
  시간                         -0.01 / s   (같은 성공이면 빠를수록 좋다)

리턴 G_t = Σ_{τ ≥ t} γ^(τ - t) r_τ  (γ 는 초당 할인율). V(s_t) 는 이 G_t 의 기댓값을 배운다.
γ < 1 이면 먼 미래 보상이 덜 중요해지고, 분산이 줄어 학습이 안정적이다.
"""
from dataclasses import dataclass
import math


@dataclass
class RewardConfig:
    success: float = 1.0
    failure: float = -1.0
    collision: float = -10.0
    lane_end_stop: float = -0.2
    time_per_sec: float = -0.01
    gamma_per_sec: float = 0.98


def episode_events(ep):
    """EpisodeResult(dict) → [(t, reward)] 사건 목록(시간 보상 제외)."""
    cfg_events = []
    for t in ep.get("stop_events", []):
        cfg_events.append((t, "lane_end_stop"))
    if ep["collision"]:
        cfg_events.append((ep["t_end"], "collision"))
    elif ep["success"]:
        cfg_events.append((ep["t_merge"], "success"))
    else:
        cfg_events.append((ep["t_end"], "failure"))
    return cfg_events


def returns_for_samples(ep, rc=RewardConfig()):
    """에피소드의 각 샘플 시각 t_i 에서의 리턴 G(t_i).

    사건 보상은 그 시각에 한 번, 시간 보상은 매 순간 흐른다고 보고 닫힌 꼴로 적분한다:
      ∫_{t_i}^{T} γ^(τ-t_i) · c dτ = c · (1 - γ^(T-t_i)) / (-ln γ)
    """
    events = episode_events(ep)
    T = ep["t_end"] if not ep["success"] else ep["t_merge"]
    lg = -math.log(rc.gamma_per_sec)
    out = []
    for t_i, _ in ep["samples"]:
        g = rc.time_per_sec * (1.0 - rc.gamma_per_sec ** max(0.0, T - t_i)) / lg
        for t_e, kind in events:
            if t_e + 1e-9 >= t_i:
                g += getattr(rc, kind) * rc.gamma_per_sec ** (t_e - t_i)
        out.append(g)
    return out
