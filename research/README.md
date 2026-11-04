# research/ — 시나리오 트리 SMPC 차선 변경 판단기

기존 SMPC(이 저장소의 `src/`, dd1cf54)는 차선 변경 후의 대응(복귀·제동)을 위험도 평가에 넣지 않고,
확률과 무관한 게이트(모든 mode clearance, 간격·TTC·헤드웨이)를 확률 제약 위에 겹쳐 둔다. 그래서 느리고
촘촘한 합류 구간에서 차선을 바꾸지 못한다. 이 폴더는 그것을 고치는 연구 코드다.

1. **트리 판단**: mode 가 드러나기 전에는 공통 행동을 하고, 드러난 뒤에는 가지별 대응(계속·복귀·제동,
   실행 계층이 실제로 하는 것만)을 고른다. 스텝별 Σ_j p_j r_j ≤ ε 를 지키는 대응 조합을 열거해 기대 비용이
   가장 낮은 루트를 고른다.
2. **실행 중 재계획(모니터)**: 차선 변경 중 매 틱 "계속" 과 "복귀" 두 루트를 같은 트리로 풀어(2ε) `rear_abort`
   를 대체한다. supervisor 는 연속 2 회일 때 복귀하고, 트리 메시지가 0.3 s 끊기면 rear_abort 로 돌아간다.
3. **게이트 정리**: G3(지금 간격·TTC), G5(예측 간격·TTC), 모든-mode clearance 를 걷어낸다.
   G1/G2/G6/G7/G8 과 정상 가지 clearance, 예측 실패 시 fail-closed 는 남긴다.
4. **RL 가치함수**: 트리 잎(8 s 뒤)을 학습한 V 로 평가한다 (`sim/smpc_sim/rl/README.md`). 시뮬레이터에만 있고,
   최종 로직으로 다시 학습해도 V 없는 제안보다 빠르지 않았다(+1.7 s [−2.2, +5.4]).

보고서: Claude Docs "시나리오 트리 SMPC — 고속 연속 합류 판단의 보수성 줄이기".
독립 코드 리뷰와 조치: `docs/review_20261104.md`.

## 핵심 결과 (시뮬레이터 env3, 같은 시드 2000–2199, GT 인지)

| 방법 | 판단 | 실행 | 성공 | 충돌 (자차 차선 변경 중) | 합류 중앙값 | 기존 대비 합류 시간 [95 % CI] |
|---|---|---|---|---|---|---|
| 기존 SMPC (dd1cf54) | 기존 | rear_abort | 42.0 % | 0.5 % (0) | 49.1 s | — |
| A. 게이트만 정리 | 기존 | rear_abort | 64.0 % | 9.0 % (11) | 32.6 s | −22.6 s [−29.4, −15.9] |
| B. 기존 판단 + 트리 모니터 | 기존 | 모니터 | 67.0 % | 2.0 % (0) | 32.1 s | |
| C. 트리 판단 + rear_abort | 트리 | rear_abort | 65.0 % | 10.5 % (15) | 32.3 s | |
| **D. 제안 (트리 + 모니터)** | 트리 | 모니터 | **66.0 %** | **1.5 % (0)** | **34.0 s** | **−20.8 s [−27.4, −14.0]** |

- **속도는 게이트 정리에서, 안전은 실행 중 재계획에서 나온다.** 게이트만 걷어내도 합류는 빨라지지만(A) 자차 차선
  변경 중 충돌이 생긴다. 모니터는 판단 방식과 무관하게 그 충돌을 0 으로 만든다(충돌 −7 / −9 %p). 같은 실행에서
  판단만 트리로 바꾸면 합류 시간 +0.9 s [−1.7, +3.7], 충돌 −0.5 %p 로 차이를 보이지 않는다(CI 폭 ±3–4 s).
- 남은 충돌(모든 방법)은 차선 끝 NPC 가 TTC 1 s 만 보고 끼어들거나 합류 도중 두 차선에 걸쳐 멈춘 경우다.
- 실행 안 되는 대응(계속+가속, 복귀+감속)을 허용하면 4.2 s [−6.8, −1.7] 더 빨라 보인다. 제안은 이것을 뺀 값이다.
- 게이트를 유지한 트리는 기존과 같다(41.5 %, 44.1 s): 병목은 게이트다.

표 전체: `python3 analysis/sim_final_table.py`, 충돌 분류: `python3 analysis/sim_collisions.py <planner…>`.

### 민감도 (시뮬레이터, 같은 시드 2000–2199)

| 변형 | 비교 | 합류 시간 차이 [95 % CI] | 비고 |
|---|---|---|---|
| NPC 반응 약화 (침범 1.0 m, 최대 제동 4 m/s²) | 제안 − 기존 | −21.1 s [−27.7, −14.3] | 판단 효과(D − B) +0.7 s [−2.4, +3.9], 자차 변경 중 충돌 A 11 → B·D 0 |
| RL 잎 V_w (최종 로직 재학습, `results/value_final_w/`) | V 있음 − 제안 | +1.7 s [−2.2, +5.4] | 성공 65.5 %, 충돌 1.0 % |
| ε = 0.02 / 0.08 | − 제안 | +4.1 s [+1.3, +7.1] / −6.2 s [−9.4, −3.4] | 0.08 에서 2-D 분리 거리 p05 1.12 → 0.90 m |
| LiDAR 수준 인지 | 제안 − 기존 (둘 다 LiDAR) | +8.9 s [+3.9, +14.1] | 뒤차 검출 깜빡임 → 복귀 2.44 회/에피소드, 이탈 17.5 % |
| MORAI 속도 조합 16개 × 시드 50개 | 제안 − 기존 | 목표 ≥ 60 km/h 11개 중 9개 유의하게 빠름 | 목표 ≤ 43 km/h 5개는 차이 없음 (`analysis/sim_scenarios_table.py`) |

## MORAI 결과 (GT 인지, 제안 = 트리 + 모니터 + 게이트 정리 vs 기존)

| 묶음 | 쌍 | 성공 (기존 / 제안) | 제안이 빠른 쌍 | 합류 시간 차이 중앙값 (실패 = 180 s) |
|---|---|---|---|---|
| S1–S4 × 3 회 | 12 | 6 / 12 | 12 / 12 | −69.9 s |
| 속도 격자 {30,50,70} → {40,60,80,100} km/h × 2 회 | 24 | 20 / 23 | 20 / 24 | −54.8 s |
| 전체 | 36 | 26 / 35 | 32 / 36 | −57.2 s |

- 기존은 S3·S4(목표 43·29 km/h)와 격자 50→40·70→40 에서 모두 차선 끝 정체로 끝났다. S3·S4 에서 목표 차선이 옆일 때
  바꾸기를 원한 틱의 99.8 % 가 G3–G5 중 하나 이상에 막혔다(`analysis/morai_reject_reasons.py`).
- 제안의 실패 1 건(격자 30→100 1 회차)은 모니터 복귀 뒤 차선 0 끝에서 끼어들기 인터록(G7, `cutin_safe=0`)에 막힌 정체다.
- 자차 차선 변경 중 접촉은 두 방법 모두 0 건이다. 기존 5 건은 합류 완료 뒤 NPC 끼어듦, 제안 1 건은 S3 정지 출발 중
  오른쪽 차선 끝 NPC 의 측면 합류다.
- 계산 시간(제안): 트리 판단 p50 5.5 / p90 68 / p99 134 ms, 차선 변경 중 모니터 주기 p50 50 ms / p99 0.14 s
  (`analysis/morai_timing.py`). p99 는 20 Hz 주기를 넘는다.
- MORAI 는 제안(D)과 기존(게이트 유지)만 비교했다. C++ 에 "기존 판단 + 모니터" 모드가 없어 2×2 는 시뮬레이터에서만 확인했다.
  S1–S4 의 기존 주행은 직전 세션(01:38–02:33) 것이다.

표: `python3 analysis/morai_final.py` (기존 `v2_B_*`·`v3_C_*`, 제안 `v3_*`, 기록 `results/morai_runs.jsonl`).
거짓 완료로 무효 처리한 1 회는 `results/morai_runs_invalid.jsonl` 로 옮기고 다시 돌렸다(`morai/rerun_invalid.sh`).

## 구성

| 경로 | 내용 |
|---|---|
| `analysis/` | MORAI bag 측정, GT 인지 검증, 최종 표(`sim_final_table.py`, `sim_collisions.py`, `sim_scenarios_table.py`, `morai_final.py`), 기존 게이트 거절(`morai_reject_reasons.py`), 계산 시간(`morai_timing.py`) |
| `docs/baseline_spec.md` | 기존 SMPC 판단 사양 (파이썬 이식의 근거) |
| `docs/review_20261104.md` | 독립 코드 리뷰 16 건 |
| `sim/smpc_sim/` | 1 차원 Frenet 시뮬레이터, 기존 SMPC 이식, 트리 플래너(`tree.py`), RL(`rl/`) |
| `sim/tests/` | 파이썬 이식 ↔ 원본 C++ 수치 비교 (위험도, 트리 평가, 실행 중 재계획) |
| `cpp_check/` | 원본 노드 소스를 include 하는 비교 하네스 |
| `morai/` | MORAI 자동화, 연구 시나리오 S1–S4 + 속도 격자 12 개, 주행 행렬(`run_matrix.py`) |
| `results/final/` | 최종 시뮬레이터 결과 json (`_npc1` NPC 반응 약화, `sim_scen_*` 속도 조합) |
| `results/` | MORAI 기록(`morai_runs.jsonl`, `morai_final.json`), RL V(`value_final_w/`) |

C++ 쪽 변경(모두 기본 꺼짐, 끄면 기존 동작과 같다):
- `smpc_decision_node`: `tree_planner_enabled`, `tree_keep_gates`, `tree_monitor_enabled`,
  `tree_executable_responses_only`(기본 true), `tree_plant_return`(기본 true), `/smpc/tree_abort`, `/smpc/tree_debug`
- `lane_change_supervisor_node`: `tree_abort_enabled`, `tree_abort_confirm_count`(2), `tree_abort_heartbeat_timeout_sec`(0.3)
- `target_selector_node`: GT 경로 보정(`object_topic_ref_to_center_m`, heading 오프셋, 중복 프레임 제거, 자차 외삽)

## 실행

```bash
cd research
HARNESS=risk_harness cpp_check/build.sh && HARNESS=tree_harness cpp_check/build.sh
python3 sim/tests/test_risk_vs_cpp.py && python3 sim/tests/test_tree_vs_cpp.py       # C++ 일치 확인
python3 sim/run_batch.py --planner baseline --n 200 --seed0 2000 --workers 4          # 기존 SMPC
python3 sim/run_batch.py --planner tree_gt_nogate --n 200 --seed0 2000 --workers 4    # 제안
#   접미사: _bdec(기존 판단+모니터) _nomon(트리 판단+rear_abort) _full(실행 안 되는 대응 포함) _linret(선형 복귀)
cd sim && python3 -m smpc_sim.rl.train_value --iters 2 --episodes 300 --wait_root    # V 학습

# MORAI (GT 인지, 집 PC sim_ip 192.168.100.10)
roslaunch smpc_lane_change research_morai.launch method:=baseline
roslaunch smpc_lane_change research_morai.launch method:=tree tree_keep_gates:=false
cd morai && python3 run_matrix.py --phase B --prefix v3_      # S1–S4
python3 run_matrix.py --phase C --reps 2 --prefix v3_                 # 속도 격자
python3 ../analysis/morai_final.py
```
