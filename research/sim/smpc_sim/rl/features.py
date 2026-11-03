"""가치함수 V 의 입력 특징.

V 는 두 곳에서 쓰인다.
  1) 학습: 시뮬레이터의 '실제' 상태 s_t 에서 V(s_t) ≈ G_t (그 뒤에 받은 누적 보상)
  2) 계획: 트리의 '잎', 즉 8 초 뒤 예측 상태 x_N 에서 V(x_N) 로 8 초 너머를 내다본다
두 곳에서 반드시 같은 함수로 특징을 만들어야 V 가 의미를 갖는다. 그래서 둘 다 아래
`features_from_vehicles` 하나로 계산한다. 입력은 자차 (s, y, v) 와 차량 목록 [(s, y, v, length)].

차량 목록은 판단 시점의 대표 차량(current_front, target_front, target_rear, R)만 쓴다.
트리 잎에서는 이 차량들의 예측 위치만 알기 때문이다(다른 차량은 예측하지 않는다).
"""
import numpy as np

from ..track import CSV_END_S, FINAL_LANE

GAP_CAP, DV_CAP, DIST_CAP, TTLE_CAP = 100.0, 10.0, 400.0, 30.0
FEATURE_NAMES = (["v"] + [f"lane{i}" for i in range(4)] + ["dist_end", "ttle", "lanes_left"] +
                 ["gap_f", "dv_f", "gap_r", "dv_r", "gap_nf", "dv_nf", "gap_nr", "dv_nr"])
DIM = len(FEATURE_NAMES)


def features_from_vehicles(ego_s, ego_y, ego_v, vehicles, lane_width=3.5):
    """vehicles: [(s, y, v, length)] (s 는 차량 중심). 반환 [DIM]."""
    lane = int(np.clip(round(ego_y / lane_width), 0, FINAL_LANE))
    dist = max(0.0, CSV_END_S[lane] - ego_s) if lane < FINAL_LANE else DIST_CAP
    ttle = dist / ego_v if ego_v > 0.1 else TTLE_CAP
    f = [ego_v / 25.0] + [1.0 if lane == i else 0.0 for i in range(4)]
    f += [min(dist, DIST_CAP) / DIST_CAP, min(ttle, TTLE_CAP) / TTLE_CAP, (FINAL_LANE - lane) / 3.0]

    def nearest(target_lane):
        best_f, best_r = (GAP_CAP, 0.0), (GAP_CAP, 0.0)
        for s, y, v, length in vehicles:
            if int(round(y / lane_width)) != target_lane:
                continue
            if s >= ego_s:
                g = (s - 0.5 * length) - (ego_s + 3.845)
                if g < best_f[0]:
                    best_f = (max(0.0, g), v - ego_v)
            else:
                g = (ego_s - 0.79) - (s + 0.5 * length)
                if g < best_r[0]:
                    best_r = (max(0.0, g), v - ego_v)
        return best_f, best_r

    (gf, dvf), (gr, dvr) = nearest(lane)
    (gnf, dvnf), (gnr, dvnr) = nearest(lane + 1) if lane < FINAL_LANE else ((GAP_CAP, 0.0), (GAP_CAP, 0.0))
    for g, dv in ((gf, dvf), (gr, dvr), (gnf, dvnf), (gnr, dvnr)):
        f += [min(g, GAP_CAP) / GAP_CAP, float(np.clip(dv, -DV_CAP, DV_CAP)) / DV_CAP]
    return np.array(f, dtype=np.float32)


def state_features(t, ego):
    """시뮬레이터 실제 상태(TargetSet + 자차) → 특징. 학습 샘플용."""
    vehicles = [(v.s, v.y, v.v_long, v.length) for v in (t.current_front, t.target_front, t.target_rear) if v.valid]
    return features_from_vehicles(ego.s, ego.y, ego.v, vehicles)


def leaf_features_batch(X_leaf, Y_leaf, V_leaf, tv_leaf):
    """트리 잎 → 특징 [M, DIM]. tv_leaf: 잎마다 [(s, y, v, length)] 목록(같은 가지면 공유)."""
    out = np.zeros((len(X_leaf), DIM), dtype=np.float32)
    for m in range(len(X_leaf)):
        out[m] = features_from_vehicles(float(X_leaf[m]), float(Y_leaf[m]), float(V_leaf[m]), tv_leaf[m])
    return out
