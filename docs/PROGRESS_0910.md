# SMPC 차선변경 — 진행 상황 (2026-09-08 ~ 09-10)

목표: 고속도로 합류에서 차선변경 0→1→2→3 을 안정적으로 수행.

---

## 1. 현재 적용된 변경 (전부 파라미터로 되돌리기 가능)

| # | 항목 | 값 | 파일 | 되돌리기 |
|---|---|---|---|---|
| F1 | 상대차량 yaw 를 차선 접선 100% | blend 1.0 / scale 1.0 / max_error 3.15 | `target_selector.yaml` | `scratchpad/target_selector.yaml.orig` |
| F2 | 자차 궤적 절단 허용 | `ego_reference_preview_allow_truncation: true` | `smpc_decision.yaml` | false |
| F3 | 차선변경 준비 거리 | `lane_end_prepare_distance_m: 400` | `target_selector.yaml` | 200 |
| F4 | ROI 횡폭 거리비례 확장 | `roi_y_growth_per_m: 0.07` / `roi_y_half_max: 12.0` | 두 launch | 0 |
| F5 | scene 판정 뒤차 신뢰거리 | `change_commit_scene_rear_max_range_m: 60` | `smpc_decision.yaml` | 0 |
| F6 | 추월 중 멀어지는 앞차 램프 미적용 | `pass_gap_skip_ramp_while_receding: true` | `smpc_decision.yaml` | false |
| F7 | 위험도에 자차 제동 반응 모델링 | `ego_brake_reaction_enabled: true` / `_max_decel: 4.0` | `smpc_decision.yaml` | false |
| F8 | 길고 납작한 트랙 대표 제외 | `representative_flat_max_length_m: 8.0` / `_height_m: 0.25` | `target_selector.yaml` | 0 |
| F9 | 비차량 장애물 RViz 별도 표시 | `obstacle_marker_enabled: true` | tracker | false |
| F10 | ACC 리드는 차체 중심이 자차 앞일 때만 | `lead_requires_center_ahead: true` | `acc_planner.yaml` | false |
| F11 | **지면 추정 구간별** | `ground_pct_bin_m: 10.0` | 두 launch | 0 |

### connector 연장 (F12) — 6개 값이 한 묶음, 하나만 바꾸면 깨짐

| 파일 | 파라미터 | 값 |
|---|---|---|
| `waypoint_system/base.yaml` | `endpoint_connector_source_remaining_distance_m_` | 30.0 |
| `waypoint_system/base.yaml` | `endpoint_connector_target_forward_m_` / `_min_forward_m_` | 20.0 / 20.0 |
| `acc_planner.yaml` | `lane_end_merge_wait_stop_distance_m` | 25.0 |
| `acc_planner.yaml` | `endpoint_connector_speed_cap_mps` / `_max_accel_mps2` | 7.0 / 4.0 |
| `smpc_decision.yaml` | `stopped_launch_connector_speed_cap_mps` / `_max_accel_mps2` | 7.0 / 4.0 |

**지켜야 할 제약**

```
발동거리(30) > 정지선(25)                              ← 안 지키면 connector 미생성
SMPC cap/accel == ACC cap/accel                        ← 안 지키면 지평 초과로 빈 궤적
delay(0.55) + 완주시간 + buffer(1.5) <= horizon(8.0)   ← 현재 5.86 s, 여유 2.14 s
```

세 번째가 실질 상한. 현재 여유로 connector 를 30 m 까지는 늘릴 수 있음(완주 5.0 s).

---

## 2. 코드 변경

| 파일 | 내용 |
|---|---|
| `smpc_decision_node.cpp` | 궤적 절단 허용, `ego_traj=` 디버그 계측 추가, 자차 제동 반응 모델, 추월 램프 미적용 |
| `target_selector_node.cpp` | flat-long 트랙 대표 제외 |
| `acc_speed_planner_node.cpp` | `leadIsInFront` 에 중심-앞 조건 |
| `lidar_ttc_tracker_node.py` | ROI 거리비례 확장, `obstacle_box` 마커, `_append_track_text` 분리 |

`ego_traj=` 계측이 특히 유용했음. 이전에는 실패가 전부 `empty_ego_traj` 하나로 뭉뚱그려졌는데,
`preview_stale` / `origin_arc_invalid` / `connector_anchor_stale` / `connector_horizon_short` /
`no_valid_step` / `truncated_before_merge_complete` / `terminal_not_on_target_lane` /
`truncated_after_merge` / `full` 로 나뉘어 원인 특정이 즉시 가능해짐.

---

## 3. 검증된 효과

### F11 지면 구간별 추정 (lc_gt 2026-09-08-17-52 재생)

| | bin 0 (전역) | bin 10 m |
|---|---|---|
| tracked/frame | 8.35 | 3.95 |
| 트랙 id | 95 | 36 |
| 유령/frame | 4.74 | **0.58** (−88%) |
| ACC hard | 8 | 2 |
| 차선 0 정지 유령(길이≥8 m) | 13 프레임 | **0** |

**차선 위 GT 차량의 검출률 (모집단 = 검출되어야 할 차량)**

| 거리 | bin 0 | bin 10 |
|---|---|---|
| 0~20 m | 53.2% | 52.6% |
| 20~40 m | 68.0% | 68.5% |
| 40~60 m | 60.1% | 62.7% |
| 60~90 m | 38.6% | 36.4% |
| 90~120 m | 2.7% | 4.9% |
| **합계** | **42.5%** (451/1060) | **42.9%** (455/1060) |

**실차 손실 없음.** 유령만 88% 제거. 코드 주석이 우려한 car-like 커버리지 하락은
이 시나리오에서 나타나지 않음.

> 주의 — 처음에 "GT 실차 대응 556 → 519, 손실 6.7%" 로 보고했으나 이는 **트랙 프레임 수**
> 기준이라 틀렸음. 유령이 줄면 한 차량에 중복으로 붙던 트랙도 줄어 그 수가 감소함.
> 올바른 모집단은 "검출되어야 할 GT 차량" 이고, 그 기준으로는 손실이 없음.
> (§5 측정 규칙의 '모집단' 항목 위반 사례)

### F10 ACC 리드 중심-앞 조건 (lc_gt 2026-09-08-14-45 재생)

| | 원본 | target_selector 에서 차단 | **leadIsInFront 에서 차단** |
|---|---|---|---|
| hard 총계 | 48 | 108 (악화) | **6** |
| ├ current_front | 17 | 6 | 6 |
| └ nearby_overlap | 31 | 102 | **0** |
| 목표속도 0 | 70 | 172 | **70** |
| reason=none | 55.4% | 52.9% | **62.1%** |

`current_front` 에서만 빼면 `nearby_overlap` 이 대신 받아 악화됨.
`leadIsInFront` 는 세 리드 경로가 모두 지나는 유일한 지점이라 우회가 없음.

### F12 connector 연장 (lc_gt 2026-09-08-17-37 재생)

| | cap 4.0/2.0 | cap 7.0/4.0 |
|---|---|---|
| `connector_horizon_short` | 764 | **0** |
| `ego_traj = full` | 184 | **925** |
| `empty_ego_traj` | 771 | **5** |
| feasible | 2 | **44** |
| CHANGE_LEFT 3연속 이상 | 없음 | **6 회** |

connector 곡률: L=10 → 0.185 1/m(반경 5.4 m), L=20 → 0.049 1/m(반경 20.6 m).
7.0 m/s 에서 횡가속 2.40 m/s² 로 기존 4.0 m/s @ 2.98 m/s² 보다 오히려 낮음.

### 실주행 성과 (lc_gt 2026-09-08-17-43)

차선변경 0→1→2→3 **전부 성공**. `rear_abort` 가 처음으로 정상 작동
(9.57 s 확정 중단 → 4.4 초 뒤 재시도 성공).

---

## 4. 미해결

| # | 내용 | 근거 |
|---|---|---|
| D3 | `rear_abort` 확정 카운터가 뒤차 미검출로 리셋됨 | `resetRearAbortConfirmation()` 이 `rear_threat=false` 에서 호출. 차선변경 중 `target_rear` 유효율 중앙 18%(8회 중 1회는 0%) |
| D4 | 차선변경 중 뒤차 검출률 중앙 18% | 8회: 0/8/9/16/19/21/32/63% |
| D5 | 뒤차 60 m 밖 검출률 0% (센서 물리 한계) | VLP-16 링 간격 d·tan(2°), 60 m 에서 2.10 m > 차량 높이 1.5 m. `max_rear_range: 120.0` 은 줄 수 없는 값을 전제 |
| D6 | SMPC 가 자기 승인 직후의 ACC 정책을 모름 | `targetFrontOverlapSoftAccAllowed()` 가 `lane_change_active_` 이후에만 발동. 14:45 2→3 에서 1.9 초 만에 ACC 목표 25.0 → 10.4 m/s |
| D7 | 속도 유지 상태에서 게이트 통과 실패 | 3회 중 2회가 15 m/s → 0 m/s 후 변경. 차단 사유 ttc/cutin/scene |
| D8 | 주행의 절반이 차선끝 대기 | 16:15 에서 `lane_end_wait*` 611/1178 (52%) |
| D10 | 한 차량이 target_front 와 target_rear 동시 점유 | 겹침 시 두 조건 모두 만족. `targetRearForGap()` 우회 로직 이미 존재 |
| D11 | 속도 오검출 | 14:45 8.04 s `target_rear` v_long −48.31 m/s |
| D12 | 두 대 병합 클러스터 | 차선 위 길이≥8 m 252건, 폭 정상 1.64 m |
| ? | `rear_abort` 가 `escapable=1` 인데 확정된 사례 | 17:52 9.73 s. 다른 뒤차가 `escapable=0` 이었을 가능성 — `last_rear_guard_detail_` 은 마지막 차량만 기록 |

---

## 5. 측정 방법 — 반드시 지킬 것

이 세션에서 **잘못된 측정으로 여러 번 틀린 결론**을 냈음. 원인은 매번 같았음:
결정이 적용되는 모집단이 아니라 뽑기 쉬운 모집단에서 재고 바로 보고함.

| 항목 | 규칙 |
|---|---|
| GT 매칭 | **`unique_id` 기준.** `npc_list` 인덱스는 프레임마다 다른 차량을 가리킴 (인덱스로 재서 "yaw 역방향 23.2%" 오진 → 실제 0.0%) |
| 시간축 | **`header.stamp` 사용.** bag 수신 시각은 버스트로 들어와 dt=0 구간이 있음 |
| 모집단 | 규칙이 **실제로 적용되는 집합**에서 측정. 전체 트랙 4,135건에서 잰 정밀도 90.2% 가 대표 599건에서는 62% 였음 |
| 대조군 | 항상 함께 제시. "유령의 79%가 좌측" → 실차는 99%가 좌측이었음 |
| 속도 단위 | `/Object_topic` npc velocity 는 **km/h**, `/Ego_topic` 은 m/s (위치미분 대비 비율 3.593 로 확인) |
| 필터 자기확인 | 쿼리에 넣은 조건을 데이터의 성질로 착각하지 말 것 (`abs(y)<=1.15` 로 걸러놓고 "\|y\| 가 1.15 에서 잘린다"고 보고) |
| 충돌 판정 | **`/CollisionData` 는 접촉 이벤트가 아님.** 자차(uid 0)를 항상 포함하고 `collision_objecta` 는 늘 비어 있음. OBB 대 OBB 거리로 직접 계산할 것 |
| 재생 편차 | 3 회 반복 측정 결과 ±0.2~1.2%p. 그보다 큰 차이는 실제 효과 |

---

## 6. 재생으로 확인 가능한 것 / 아닌 것

**가능** — 인지(트랙, 대표 차량), SMPC 게이트 판정, ACC 명령, 감독자 승인/중단.

**불가능** — 자차의 실제 거동. 자차 위치가 bag 값이라 ACC 명령이 주행에 반영되지 않음.
`/gps_state` 와 `/final_waypoint` 도 bag 이 덮어써서, 감독자의 `path_switch_confirmed_` /
`target_path_valid_` 가 영영 참이 되지 않아 `target_path_switch_timeout` 이 필연적으로 발생함
(재생 아티팩트이지 결함이 아님).

### 재생 스크립트

```bash
# 격리 마스터 11411 사용 — 사용자의 시뮬레이터(11311)와 충돌하지 않음
ROS_MASTER_URI=http://localhost:11411
roscore -p 11411
rosparam set /use_sim_time true
roslaunch smpc_lane_change smpc_integration.launch use_rviz:=false
rosbag play <bag> --clock --topics <입력 토픽들>
```

`--clock` + `use_sim_time` 없으면 header 신선도 검사(0.35/0.50 s)가 항상 만료됨.
`/path_switch_preview` 를 재생 목록에 반드시 포함할 것.

---

## 7. 다음 단계

1. **F11 지면 구간별 추정을 시뮬레이터에서 검증** — 유령 급제동이 실제로 사라지는지
2. D3 — 뒤차 미검출을 위협 해소로 읽지 않도록 (`rear_abort` 리셋 조건)
3. D6 — SMPC 자차 롤아웃에 차선변경 중 ACC 정책 반영
4. D7 — 속도 유지 상태 게이트 통과. 근본이지만 게이트 5개가 얽혀 범위가 큼

### 시뮬레이터 녹화 명령

```bash
rosbag record -o /home/kuuve/bags/lc_gt \
  /velodyne_points /imu /odom /gps_state /gps_back \
  /Object_topic /Ego_topic /CollisionData /Competition_topic \
  /global_path /local_path /final_waypoint /trajectory_path /path_number \
  /path_switch_preview /path_switch_connector_active /ctrl_cmd /tf \
  /control_feedback/actual_speed_mps /control_feedback/mission_speed_target_mps \
  /control_feedback/mpc_target_speed_mps /control_feedback/curvature_speed_limit_mps \
  /control_feedback/path_curvature_for_speed_limit \
  /tracked_objects /tracked_objects_markers \
  /smpc/targets /smpc/decision /smpc/smpc_decision_debug \
  /smpc/behavior_longitudinal_status \
  /smpc/lane_change_status /smpc/lane_change_supervisor_debug
```
