"""고속도로 합류 구간 형상. waypoint_system/data/default/{0..3}.csv 에서 잰 값.

공통 좌표 s: 차선 1·2·3 CSV 의 시작점이 s = 0. 차선은 직선·평행(폭 3.5 m)으로 편다.
  lane 0 (합류 차선) : CSV 끝 s = 106.2  (차선 1 과 나란한 구간 0..106)
  lane 1             : CSV 끝 s = 403.9
  lane 2             : CSV 끝 s = 417.9
  lane 3 (최종)       : CSV 끝 s = 1110.4
SMPC 가 보는 "차선 끝"은 CSV 끝이다(target_selector distance_to_lane_end).
NPC 가 달리는 실제 도로는 CSV 보다 길다(차선 1·2 는 CSV 끝 뒤에도 이어진다).
"""
from dataclasses import dataclass, field

from .prediction import StraightLanes

CSV_END_S = {0: 106.2, 1: 403.9, 2: 417.9, 3: 1110.4}
CSV_START_S = {0: -1112.0, 1: 0.0, 2: 0.0, 3: 0.0}
FINAL_LANE = 3


@dataclass
class Track:
    lane_width: float = 3.5
    csv_end: dict = field(default_factory=lambda: dict(CSV_END_S))
    csv_start: dict = field(default_factory=lambda: dict(CSV_START_S))
    # NPC 도로 범위(시뮬레이터 전용). 차선 1·2 는 CSV 끝 뒤로도 이어진다.
    road_start: dict = field(default_factory=lambda: {0: -400.0, 1: -300.0, 2: -300.0, 3: -300.0})
    road_end: dict = field(default_factory=lambda: {0: 106.2, 1: 1110.4, 2: 1110.4, 3: 1110.4})
    final_lane: int = FINAL_LANE

    def lanes_for_prediction(self, extension_before_m=100.0):
        """SMPC 예측용 차선(CSV 범위). TV 예측이 CSV 끝을 넘으면 그 mode 의 위험 계산이 멈춘다."""
        return StraightLanes(self.lane_width, dict(self.csv_end), extension_before_m)

    def center_y(self, lane):
        return lane * self.lane_width

    def lane_of_y(self, y):
        return int(round(y / self.lane_width))

    def distance_to_lane_end(self, lane, s):
        return self.csv_end[lane] - s
