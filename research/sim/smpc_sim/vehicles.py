"""TargetVehicle 메시지와 관측 가속도 필터(VehicleHistory)의 파이썬 판.

좌표계: 직선·평행 차선. 전역 x = 공통 s, 전역 y = lane_id * lane_width + d.
차선 번호가 커질수록 왼쪽(+y)이다. 차선 yaw 는 0 이다.
"""
from dataclasses import dataclass, field


@dataclass
class TargetVehicle:
    """msg/TargetVehicle.msg 중 판단에 쓰이는 필드."""

    valid: bool = False
    unique_id: int = -1
    role: str = ""
    lane_id: int = -1
    x: float = 0.0
    y: float = 0.0
    s: float = 0.0           # 자기 차선 CSV 위 호장
    d: float = 0.0           # 자기 차선 중심에서의 횡오프셋
    delta_s: float = 0.0     # 자차 기준 종방향 상대 위치(중심)
    yaw: float = 0.0
    yaw_rate: float = 0.0
    v_long: float = 0.0
    v_lat: float = 0.0
    length: float = 4.6
    width: float = 1.9
    half_s: float = 0.0
    front_delta_s: float = 0.0
    rear_delta_s: float = 0.0
    track_hits: int = 0
    accel: float = 0.0        # GT 종방향 가속도(MORAI /Object_topic acceleration.x). 메시지에는 아직 없다(plan_v2 2d)

    def has_role(self, word):
        return word in self.role


@dataclass
class VehicleHistory:
    initialized: bool = False
    v_long: float = 0.0
    filtered_accel: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    stamp: float = 0.0


@dataclass
class HistoryBank:
    """smpc_decision_node.cpp:878-915 updateOneVehicleHistory 이식.

    /smpc/targets 한 번에 한 번 갱신된다. raw = clamp(dv/dt, ±max),
    filtered = α·raw + (1-α)·filtered. 관측이 끊기면(gap timeout) 0 으로 리셋.
    """

    alpha: float
    max_accel: float
    gap_timeout: float
    histories: dict = field(default_factory=dict)

    @classmethod
    def from_params(cls, P):
        return cls(alpha=P.observed_accel_alpha,
                   max_accel=P.max_observed_accel_mps2,
                   gap_timeout=P.change_commit_observation_gap_timeout_sec)

    def update_one(self, v, stamp):
        if not v.valid or v.unique_id < 0:
            return
        h = self.histories.setdefault(v.unique_id, VehicleHistory())
        discontinuous = (h.initialized and h.last_seen > 0.0 and
                         stamp - h.last_seen > max(0.0, self.gap_timeout))
        if not h.initialized or discontinuous:
            h.initialized = True
            h.v_long = v.v_long
            h.filtered_accel = 0.0
            h.first_seen = h.last_seen = h.stamp = stamp
            return
        dt = stamp - h.stamp
        if 1e-3 < dt < 1.0:
            raw = min(max((v.v_long - h.v_long) / dt, -self.max_accel), self.max_accel)
            a = min(max(self.alpha, 0.0), 1.0)
            h.filtered_accel = a * raw + (1.0 - a) * h.filtered_accel
        h.v_long = v.v_long
        h.last_seen = h.stamp = stamp

    def update(self, targets, stamp):
        """targets: TargetSet. current_front/target_front/target_rear/nearby 순서."""
        for v in (targets.current_front, targets.target_front, targets.target_rear):
            self.update_one(v, stamp)
        for v in targets.nearby_vehicles:
            self.update_one(v, stamp)

    def observed_accel(self, v):
        if v.unique_id < 0:
            return 0.0
        h = self.histories.get(v.unique_id)
        return h.filtered_accel if h is not None and h.initialized else 0.0


@dataclass
class TargetSet:
    """msg/TargetVehicleSet.msg."""

    stamp: float = 0.0
    current_lane_id: int = 0
    target_lane_id: int = 1
    ego_s_current: float = 0.0
    ego_d_current: float = 0.0
    ego_d_target: float = 0.0
    current_lane_length: float = 0.0
    distance_to_lane_end: float = float("inf")
    time_to_lane_end: float = float("inf")
    lane_change_required: bool = False
    lane_change_prepare: bool = False
    lane_change_urgent: bool = False
    emergency_stop_required: bool = False
    current_front: TargetVehicle = field(default_factory=TargetVehicle)
    target_front: TargetVehicle = field(default_factory=TargetVehicle)
    target_rear: TargetVehicle = field(default_factory=TargetVehicle)
    nearby_vehicles: list = field(default_factory=list)
