"""반응형 트리 플래너 (research/docs/plan_v2.md 단계 3).

TreePlanner 의 판단 골격(루트 후보, 비용, 준비 행동 요청)은 그대로 두고 세 가지를 바꾼다.
  1. 루트 평가: 가지 = (뒤차 yield/nonyield) × (앞차 base/brake/accel), 자차·타겟 차량을 reactive.joint_rollout 으로
     함께 굴린다. 확률 = 기본 1 − Σ예외, 예외는 최솟값 p_exc_floor. 대응 = 계속 / 복귀(예외가 드러나는 시점부터).
     루트 값 = min over 가지별 대응 조합 Σ_b p_b J_b, 단 모든 스텝에서 Σ_b p_b r_b(k) ≤ ε.
  2. 공통 조건 G6/G7/G8 을 쓰지 않는다(drop_common). G7 이 보던 목표 차선 뒤·옆 차량은 트리가 모두 본다.
  3. 모니터: 판단과 같은 예측으로 '지금부터 계속' 과 '지금 복귀' 를 비교한다. 뒤차가 반응 지연이 지났는데도
     감속하지 않으면 nonyield 확률을 올린다(두 가설 갱신). 계속 위험 > 2ε 이고 복귀가 더 안전하면 복귀.
single=True 는 비교군: 가지 없이 기본 예측 하나(뒤차 yield, 앞차 base)로만 판단한다.
"""
from dataclasses import dataclass
import itertools
import math

import numpy as np

from .baseline import waypoint_blend_duration
from .reactive import CONTINUE, RETURN, ReactiveParams, joint_rollout, load_acc_params, target_lane_vehicles
from .rollouts import CHANGE_NOW, ActionSpec, RolloutContext
from .tree import RETURN as TREE_RETURN, TreePlanner


@dataclass
class _Roots:
    """root_trajectories 가 넘기는 것: 가지마다 굴려야 하므로 궤적 대신 입력을 들고 다닌다."""
    ctx: object
    specs: list
    t: object
    valid: np.ndarray


class ReactiveTreePlanner(TreePlanner):
    def __init__(self, P, track, cfg=None, rp=None, single=False, drop_common=True):
        super().__init__(P, track, cfg)
        self.rp = rp or ReactiveParams()
        self.A = load_acc_params()
        self.single, self.drop_common = single, drop_common
        self.name += "_rx" + ("_single" if single else "") + ("" if drop_common else "_common")
        self.intr_since = None            # 실행 중 자차가 목표 차선에 들어간 시각
        self.p_rear_post = None
        self.lc_connector = None          # 마지막 지금 바꾸기 승인의 실행 방식(connector 출발인가)

    # ------------------------------------------------------------ 훅
    def root_trajectories(self, ctx, specs, t):
        vehicles = self._vehicles(t)
        r = joint_rollout(ctx, specs, t, vehicles, self.P, self.A, self.rp, self.track,
                          [("yield", "base", None)], self._accel_of)
        return _Roots(ctx, specs, t, r["valid"])

    def common_ok(self, g6, g7, g8):
        return True if self.drop_common else (g6 and g7 and g8)

    @staticmethod
    def _accel_of(v):
        return getattr(v, "accel", 0.0)

    @staticmethod
    def _vehicles(t):
        out, seen = [], set()
        for v in [t.current_front, t.target_front, t.target_rear] + list(t.nearby_vehicles):
            if v.valid and v.unique_id not in seen:
                seen.add(v.unique_id)
                out.append(v)
        return out

    def _branches(self, t, p_rear=None):
        cf, tf, rears = target_lane_vehicles(t, self._vehicles(t))
        if self.single:
            return [("yield", "base", None, 1.0)]
        fl = self.rp.p_exc_floor
        pr = fl if p_rear is None else p_rear
        rb = [("yield", 1.0 - pr, None), ("nonyield", pr, "rear")] if rears else [("yield", 1.0, None)]
        fb = [("base", 1.0 - 2 * fl, None), ("brake", fl, "front"), ("accel", fl, None)] if (cf or tf) else [("base", 1.0, None)]
        out = []
        for (rm, p1, t1), (fm, p2, t2) in itertools.product(rb, fb):
            out.append((rm, fm, t2 or t1, p1 * p2))
        return out

    # ------------------------------------------------------------ 루트 평가
    def evaluate_tree(self, traj, specs, t, v_e, R, vehicles, root_cost, candidate_mask=None):
        """TreePlanner.evaluate_tree 와 같은 반환: (feas, cost, combo, peak, branches)."""
        P, c = self.P, self.cfg
        roots = traj
        br = self._branches(t)
        r = joint_rollout(roots.ctx, specs, t, self._vehicles(t), P, self.A, self.rp, self.track,
                          [b[:3] for b in br], self._accel_of)
        return self._solve(r, br, root_cost, roots.valid if candidate_mask is None else roots.valid & candidate_mask,
                           P.risk_epsilon)

    def _solve(self, r, br, root_cost, mask, eps):
        c = self.cfg
        B = len(br)
        N = r["risk"].shape[2]
        p = np.array([b[3] for b in br])
        opts = [[CONTINUE, RETURN] if br[b][2] else [CONTINUE] for b in range(B)]
        combos = np.array(list(itertools.product(*opts)))                    # [C, B]
        cost_r = np.array([0.0, c.cost_response[TREE_RETURN]])
        feas = np.zeros(N, dtype=bool)
        best_cost = np.full(N, np.inf)
        best_combo = np.zeros((N, B + 1), dtype=int)
        best_risk = np.ones(N)
        for i in range(N):
            if not mask[i]:
                continue
            ok = np.ones(len(combos), dtype=bool)
            rk = np.zeros((len(combos), r["risk"].shape[-1]))
            jc = np.zeros(len(combos))
            for b in range(B):
                q = combos[:, b]
                ok &= r["allowed"][b, q, i]
                rk += p[b] * r["risk"][b, q, i]
                jc += p[b] * cost_r[q]
            peak = rk.max(axis=1)
            total = root_cost[i] + jc + c.risk_weight * peak
            good = ok & (peak <= eps)
            if good.any():
                k = int(np.argmin(np.where(good, total, np.inf)))
                feas[i], best_cost[i], best_risk[i] = True, total[k], peak[k]
            elif ok.any():
                k = int(np.argmin(np.where(ok, peak, np.inf)))
                best_risk[i] = peak[k]
            # TreePlanner 형식: [0] 은 nominal 자리, 이후 가지별 대응(복귀 = TREE_RETURN)
            best_combo[i, 1:] = np.where(combos[k] == RETURN, TREE_RETURN, 0) if (good.any() or ok.any()) else 0
        self.last_tree = {"branches": [(f"{b[0]}/{b[1]}", round(b[3], 3), b[2]) for b in br]}
        return feas, best_cost, best_combo, best_risk, br

    # ------------------------------------------------------------ 모니터
    def monitor(self, now, t, v_e, mission):
        self.mission = mission
        self.hist.update(t, now)
        self._update_rear_belief(now, t)
        return self.monitor_core(t, v_e, mission, connector=self.lc_connector)

    def _update_rear_belief(self, now, t):
        """뒤차 양보 여부 관측 갱신. 자차가 목표 차선에 기준 이상 들어간 뒤 반응 지연 + 알아채는 시간이 지났는데
        뒤차가 다가오면서(자차보다 빠름) 감속하지 않으면 nonyield, 감속하면 yield 쪽으로 기울인다."""
        rp, W = self.rp, self.P.lane_width_m
        intr = 0.5 * W + 0.5 * self.P.ego_width_m - abs(t.ego_d_target)
        if intr < rp.rear_react_intrusion_m:
            self.intr_since, self.p_rear_post = None, None
            return
        if self.intr_since is None:
            self.intr_since = now
        _, _, rears = target_lane_vehicles(t, self._vehicles(t))
        if not rears:
            return
        rear = rears[0]
        if now - self.intr_since < rp.rear_react_delay_s + rp.rear_detect_s:
            return
        a = self._accel_of(rear)
        if a <= -0.5:
            self.p_rear_post = rp.p_exc_floor * 0.2
        elif a > -0.1 and rear.v_long > getattr(self, "_v_e", 0.0) + 0.5:
            self.p_rear_post = 0.9

    def monitor_core(self, t, v_e, mission, connector=None):
        """connector = 이 차선 변경의 실행 방식(C++ rxMonitorReturn 과 같다). 실행 중에는 승인 때 고정한 값,
        승인 전 일치 검사에서는 승인 평가의 값. 속도로 다시 추정하면 정지 출발 직후 blend 로 바뀌어 위험이 튄다."""
        P, c, rp = self.P, self.cfg, self.rp
        self._v_e = v_e
        d_t = abs(t.ego_d_target)
        if d_t <= c.abort_before_abs_d:
            return None
        T_rem = max(0.5, waypoint_blend_duration(v_e) * d_t / P.lane_width_m)
        near_end = connector if connector is not None else (t.distance_to_lane_end <= 30.0 and v_e <= 1.0)
        tf = t.target_front
        ctx = RolloutContext(v0=v_e, s0=t.ego_s_current, lane=t.current_lane_id, target_lane=t.target_lane_id,
                             ego_d_target=t.ego_d_target, pre_cap=mission, mission=mission,
                             tf_speed=tf.v_long if tf.valid else math.nan, D=T_rem, yield_ref_speed=v_e,
                             preview_mode="connector" if near_end else "blend", connector_join_arc=20.0)
        br = self._branches(t, self.p_rear_post)
        # 모니터의 루트 하나 = 지금부터 계속, 대응 1 = 지금 복귀(모든 가지에서). tb_rule 'now'.
        r = joint_rollout(ctx, [ActionSpec(CHANGE_NOW, 0.0, 0.0, T_rem, True)], t, self._vehicles(t), P, self.A, rp,
                          self.track, [(b[0], b[1], "now") for b in br], self._accel_of, mid_T_rem=T_rem)
        p = np.array([b[3] for b in br])
        go = (p[:, None] * r["risk"][:, CONTINUE, 0]).sum(axis=0).max()
        if not r["allowed"][:, RETURN, 0].all():
            return None
        back = (p[:, None] * r["risk"][:, RETURN, 0]).sum(axis=0).max()
        cont = go
        if c.monitor_tree:
            # 계속 쪽도 승인처럼 가지별 나중 대응(앞차 급감속은 알아챈 뒤, 뒤차 비양보는 관측했으면 지금)을 넣어 푼다
            br2 = [(b[0], b[1], "now" if (b[2] == "rear" and self.p_rear_post is not None) else b[2], b[3]) for b in br]
            r2 = joint_rollout(ctx, [ActionSpec(CHANGE_NOW, 0.0, 0.0, T_rem, True)], t, self._vehicles(t), P, self.A, rp,
                               self.track, [b[:3] for b in br2], self._accel_of, mid_T_rem=T_rem)
            _, _, _, risk2, _ = self._solve(r2, br2, np.zeros(1), np.ones(1, dtype=bool), P.risk_epsilon)
            cont = float(risk2[0])
        if cont > c.abort_risk_factor * P.risk_epsilon and back < cont:
            return "return"
        return None


def make_reactive_planner(name):
    """run_batch.py 용: 'rx' (새 구조), 'rx_single' (가지 없음 비교군), 'rx_common' (G6–G8 유지).
    민감도: 'fNN' = 예외 확률 최솟값 0.NN, 'mK' = 모니터 복귀 문턱 Kε (예: rx_f10, rx_m3),
    'mt' = 모니터 계속 위험에 가지별 나중 대응(11-05 부터 기본), 'mo' = 옛 모니터 규칙(예: rx_mo, rx_mo_m5)."""
    from .tree import TreeConfig
    flags = set(name.split("_"))
    cfg = TreeConfig(probability_table="gt", keep_gates=False, decision="tree", monitor=True)
    floor = next((int(f[1:]) / 100.0 for f in flags if f.startswith("f") and f[1:].isdigit()), None)
    factor = next((float(f[1:]) for f in flags if f.startswith("m") and f[1:].isdigit()), None)
    if factor is not None:
        cfg.abort_risk_factor = factor
    if "mt" in flags:                     # 'rx_mt' = 모니터가 계속 쪽에도 가지별 나중 대응을 넣음(지금 기본)
        cfg.monitor_tree = True
    if "mo" in flags:                     # 'rx_mo' = 옛 모니터 규칙(끝까지 그냥 계속 vs 지금 복귀)
        cfg.monitor_tree = False

    def factory(P, track):
        rp = ReactiveParams()
        if floor is not None:
            rp.p_exc_floor = floor
        tp = ReactiveTreePlanner(P, track, cfg=TreeConfig(**cfg.__dict__), rp=rp, single="single" in flags,
                                 drop_common="common" not in flags)
        tp.name = name
        return tp
    return factory
