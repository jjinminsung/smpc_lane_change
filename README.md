# smpc_lane_change

주변 차량(TV)의 주행·행동 불확실성을 고려해 자차(EV)의 차선 변경 가능성을 예측하고,
차선 변경을 가능하게 만드는 **종방향 행동(가속 / 감속 / 양보 / 대기)** 을 결정하는
확률적 행동 계획기(stochastic behavior planner) ROS 패키지입니다.

Multimodal SMPC를 저수준 차량 제어기로 쓰지 않고, **높은 수준의 행동 결정기**로 사용하는 것이 목표입니다.
실제 조향·속도 추종은 Kinematic 이륜차 model을 사용한 MPC를 사용합니다.

- 환경: ROS1 (catkin), MORAI 시뮬레이터, IONIQ 5 차량 모델, VLP-16 LiDAR
- 시나리오: 고속도로 합류 구간에서 차선 0 → 1 → 2 → 3 순차 좌측 차선 변경

---

## 시스템 구성

```
 /velodyne_points, /odom, /imu
        │
        ▼
 lidar_ttc_tracker ──/tracked_objects──▶ target_selector ──/smpc/targets──┬──▶ smpc_decision
 (별도 패키지)                          (차선별 대표 차량 선정)            │     (SMPC 행동 결정)
                                                                           │        │ /smpc/decision
                                                                           │        │ /smpc/behavior_longitudinal_request
                                                                           │        ▼
                                                                           ├──▶ acc_speed_planner ──/smpc/target_speed_mps──▶ 속도 제어
                                                                           │     (최종 목표 속도 결정)
                                                                           │        │ /smpc/behavior_longitudinal_status (→ smpc_decision)
                                                                           │
                                                                           └──▶ lane_change_supervisor ──/path_number──▶ waypoint_system
                                                                                 (차선 변경 승인·중단)   /smpc/lane_change_status
```

### 노드

| 노드 | 역할 | 주요 구독 | 주요 발행 |
|---|---|---|---|
| `target_selector_node` | 트래커 결과를 차선 CSV의 Frenet 좌표로 투영하고, 현재 차선 앞차 / 목표 차선 앞·뒤차 / 주변 차량을 선정. 차선 끝 거리와 차선 변경 필요 여부 계산 | `/tracked_objects` (또는 `/Object_topic`), `/odom`, `/gps_state`, `/smpc/lane_change_status` | `/smpc/targets`, `/smpc/target_markers`, `/smpc/csv_path_markers` |
| `smpc_decision_node` | EV의 KEEP / CHANGE 후보 궤적을 굴려보고, TV를 다중 모드로 예측해 horizon 전체의 bbox 충돌 확률을 평가. 가장 비용이 낮은 안전 행동을 결정 (20 Hz) | `/smpc/targets`, `/odom`, `/path_switch_preview`, `/smpc/target_speed_mps`, `/smpc/behavior_longitudinal_status` | `/smpc/decision`, `/smpc/behavior_plan`, `/smpc/behavior_longitudinal_request`, `/smpc/smpc_decision_debug`, 예측 경로·마커 |
| `acc_speed_planner_node` | 앞차 추종(ACC)과 SMPC의 감속·양보 요청을 합쳐 **최종 목표 속도를 단독으로 발행** | `/smpc/targets`, `/odom`, `/smpc/decision`, `/smpc/behavior_longitudinal_request`, `/path_number`, `/path_switch_connector_active` | `/smpc/target_speed_mps`, `/smpc/acc_active_lead`, `/smpc/behavior_longitudinal_status` |
| `lane_change_supervisor_node` | SMPC의 차선 변경 요청을 받아 경로 전환을 승인하고, 진행 중 뒤차 위협 시 중단(`rear_abort`) | `/smpc/decision`, `/smpc/targets`, `/gps_state`, `/odom`, `/final_waypoint` | `/path_number`, `/smpc/lane_change_status`, `/smpc/lane_change_supervisor_debug` |
| `gap_test_decision_node` | 단순 gap 기반 비교용 결정기 (`use_gap_test_decision:=true` 일 때 SMPC 대신 사용) | `/smpc/targets` | `/smpc/decision` |

### 메시지 (`msg/`)

| 메시지 | 설명 |
|---|---|
| `TargetVehicle` | 차량 1대의 Frenet 상태(s, d, 속도), bbox 크기·투영 범위, 트랙 성숙도(`track_hits`) |
| `TargetVehicleSet` | 자차 Frenet 상태, 차선 끝 정보, 대표 차량(`current_front`, `target_front`, `target_rear`)과 `nearby_vehicles` |
| `LaneChangeDecision` | KEEP / CHANGE_LEFT 결정과 비용·충돌 위험 |
| `BehaviorPlan` | 행동 계획(KEEP, CHANGE_NOW, YIELD_DECEL_THEN_CHANGE, HOLD_THEN_CHANGE, ACCEL_THEN_CHANGE, WAIT_FOR_GAP), 예상 차선 변경 시작 시각, step별 위험도 |
| `BehaviorLongitudinalRequest` | SMPC → ACC 종방향 요청(YIELD, HOLD, ACCELERATE, WAIT_FOR_GAP)과 속도 상한 |
| `BehaviorLongitudinalStatus` | ACC → SMPC 응답. 요청한 상한이 실제로 적용됐는지, 외부 제한(앞차·차선 끝)이 있는지 |
| `LaneChangeStatus` | 차선 변경 진행 여부, 출발·목표 차선, 경과 시간 |

### 설정 파일 (`config/`)

| 파일 | 대상 노드 | 주요 항목 |
|---|---|---|
| `target_selector.yaml` | target_selector | 입력 소스, 탐지 거리(`max_rear_range`), 최종 차선(`final_lane_id: 3`), 차선 변경 준비 거리(`lane_end_prepare_distance_m`) |
| `smpc_decision.yaml` | smpc_decision | horizon(`horizon_steps: 40` × `prediction_dt_sec: 0.2` = 8 s), 허용 위험도(`risk_epsilon`), 게이트·비용 파라미터 |
| `acc_planner.yaml` | acc_speed_planner | ACC 추종 파라미터, 차선 끝 대기, connector 속도 상한 |
| `supervisor.yaml` | lane_change_supervisor | 경로 전환 확인, 타임아웃, `rear_abort` 조건 |

---

## 빌드

의존 패키지: `roscpp`, `std_msgs`, `nav_msgs`, `visualization_msgs`, `morai_msgs`,
`waypoint_maker`, `waypoint_system`, `lidar_ttc_tracker`

```bash
cd ~/catkin_ws
catkin_make --pkg smpc_lane_change   # 또는 catkin_make
source devel/setup.bash
```

## 실행

```bash
# 전체 통합 실행 (LiDAR 트래커 + target_selector + ACC + SMPC + supervisor)
roslaunch smpc_lane_change smpc_integration.launch

# RViz 디버그 화면 포함
roslaunch smpc_lane_change smpc_debug.launch
```

주요 launch 인자:

| 인자 | 기본값 | 설명 |
|---|---|---|
| `waypoint_directory` | `$(find waypoint_system)/data/default` | 차선 CSV 경로 |
| `target_source` | `tracked_objects` | `tracked_objects`(LiDAR) 또는 `object_topic`(MORAI GT) |
| `use_lidar_tracker` | `true` | `lidar_ttc_tracker` 실행 여부 |
| `use_smpc_decision` | `true` | SMPC 결정기 사용 |
| `use_gap_test_decision` | `false` | SMPC 대신 단순 gap 결정기 사용 |
| `use_rviz` | `false` | RViz 실행 |

bag 재생 방법, 녹화 토픽 목록, 측정 규칙은 [`docs/PROGRESS_0910.md`](docs/PROGRESS_0910.md) 5~7절을 참고하세요.

---

## 진행 과정

> 새 기록은 이 절 아래에 날짜별로 추가합니다. 긴 실험 결과는 `docs/PROGRESS_MMDD.md`로 따로 쓰고 여기서 링크합니다.

### 최종 목표

Multimodal SMPC를 저수준 차량 제어기로 사용하는 것이 아니라, 주변 차량의 주행·행동 불확실성을 고려해
EV와 TV의 차선 변경 가능성을 예측하고, 차선 변경을 가능하게 만들기 위한
EV 종방향 제어(가속, 감속, 양보, 대기)를 결정하는 stochastic behavior planner(높은 수준의 행동 결정기) 구현.

### ~ 2026-08-20 — 기본 구현

- **Horizon 위험도**: `dt × N` 만큼의 horizon의 각 step에서 mode 확률 가중 위험을 계산
- **max risk**
  - 기존: 각 mode의 horizon에서의 max risk 값을 모든 step에서 사용
  - 개선: `risk_by_step`을 각 step에서 검사 + clearance, 전후방 gap, TTC 추가 검사
- **행동 후보**: 바로 변경, 대기(유지) 후 변경, 양보(감속) 후 변경
- **행동 우선순위**: 즉시 변경 → 속도 유지 → 감속 → 가속(아직 비활성화)
- **t_LC** (부분 구현): {0.4, 0.8, 1.2, 1.6, 2.0} s 후의 미래 후보를 20 Hz마다 재평가
- **ACC 연동**: SMPC는 감속 트리거를 전달하고, 최종 속도는 ACC에서 발행
- **TV 예측 (초급)**
  - front는 nominal / brake, rear는 nominal / accelerate heuristic
  - 초기 확률은 파라미터로 설정하고, 이후 차량의 Frenet 좌표, 상대속도, gap, TTC를 이용해 accel·brake 확률 계산
- **공분산(불확실성)**: 시간이 지날수록 커지는 scalar sigma 사용, mode별로 따로 계산하지 않음
- **lateral** (사용 안 함): TV가 차선 변경을 하지 않는다고 가정

### 2026-08-21 — 정지·저속 상황 처리

- 기존: 정지 상태에서는 무조건 차선 번호 변경, 8초 timeout
- 개선: 경로 끝점에 도달한 경우처럼 차량이 정지·저속일 때 target lane의 차선 변경 안전 판단 → 안전하면 target lane 변경
  - 저속에서 차량이 급하게 꺾이지 않도록 연속 곡선 생성
- 기존: SMPC에서 차량 중심 기준 4초 예측 blend 생성
- 개선: `waypoint_system`의 4초 차선 보간 경로를 사용하고, 저속에서는 생성한 연속 곡선을 EV trajectory로 사용
  - `waypoint_system`도 함께 수정

### 2026-08-24

- 속도 입력을 `twist.x` → `/Competition_topic`으로 변경

### 2026-09-04

- TV trajectory가 경로 끝점을 넘어가면, 끝점 이전 구간만 risk 계산에 포함하고 끝점을 넘은 구간은 버림

### 2026-09-08 ~ 09-10 — 차선 0→1→2→3 실주행 성공

상세 내용은 [`docs/PROGRESS_0910.md`](docs/PROGRESS_0910.md) 참고.

- 파라미터 변경 F1~F12 (상대차량 yaw, ROI 확장, 지면 구간별 추정, connector 연장 등)
- `ego_traj=` 디버그 계측 추가로 `empty_ego_traj` 실패 원인을 세분화
- 지면 구간별 추정(F11)으로 유령 트랙 88% 감소, 실차 검출 손실 없음
- connector 연장(F12)으로 `empty_ego_traj` 771 → 5
- 실주행에서 차선 변경 0→1→2→3 전부 성공, `rear_abort` 첫 정상 작동
- 미해결: D3~D12 (뒤차 검출률, `rear_abort` 리셋 조건, SMPC–ACC 정책 불일치 등)

### 앞으로의 계획

**Track A** — 현재 구조를 유지하면서 판단을 더 안정적·확률적으로 개선

- **공분산 전파**: 시간에 따라 sigma를 키우는 대신, 위치·속도·가속도 불확실성이 차량 모델을 따라 미래에 어떻게 커지는지 계산
- **Hard cap**: Σ pⱼRⱼ 평균 risk가 낮아도 특정 mode 하나가 너무 위험하면 후보를 탈락시켜, 드물지만 위험한 mode가 평균에 묻히지 않게 함
- **확률 기반 ε(위험도) 배분**: 모든 mode에 같은 허용 위험을 주지 않고, 발생 확률이 높은 mode는 더 엄격하게, 낮은 mode는 더 느슨하게 설정
- **Risk-cost 전환**: `TTC < 3.0 → WAIT`, `TTC ≥ 3.0 → CHANGE`처럼 한 임계값에서 행동이 뒤집히는 규칙을 연속적인 cost로 변경
  - 3.01 → 2.99가 됐다고 바로 CHANGE → WAIT로 바뀌지 않고, risk cost만 조금 늘어나며 다른 cost와 합친 결과가 역전될 때 행동이 바뀜
  - 단, 실제 충돌이나 매우 낮은 TTC 같은 진짜 안전 조건은 hard gate로 유지

**Track B** — 최적화 기반 정책

```
후보 선택 → SMPC optimizer → feedback policy + risk allocation 최적화 → 실패 시 Track A fallback
```

- 트리를 생성한 뒤 미래 반응 규칙 자체를 optimizer가 결정
- information tree를 이용해 미래 mode가 실제로 구분된 뒤 서로 다른 대응을 하도록 함

<!-- 새 진행 기록 템플릿 (복사해서 "앞으로의 계획" 위에 붙여 넣기)

### YYYY-MM-DD — 제목

- 기존:
- 개선:
- 결과:

-->
