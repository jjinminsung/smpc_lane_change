# research/ — 시나리오 트리 SMPC 차선 변경 판단기

기존 SMPC(이 저장소의 `src/`, dd1cf54)는 차선 변경 후의 대응(복귀·가속·제동)을 위험도 평가에 넣지 않는다.
그래서 "뒤차가 가속할 수도 있다"는 가능성만으로 차선 변경을 막는다. 이 폴더는 이를 고치는 연구 코드다.

1. **트리 플래너**: mode 가 드러나기 전에는 공통 행동을 하고, 드러난 뒤에는 가지별 대응을 고른다.
   스텝별 Σ_j p_j r_j ≤ ε 를 지키는 대응 조합을 열거해 기대 비용이 가장 낮은 루트를 고른다.
2. **게이트 정리**: 확률과 무관한 게이트(모든 mode clearance, 간격·TTC·헤드웨이)를 걷어내고
   확률 제약 하나로 판단한다. 트리의 대응이 위험을 실제로 낮출 때만 안전하다.
3. **RL 가치함수**: 트리 잎(8 s 뒤)을 학습한 V 로 평가해 지평선 너머를 본다 (`sim/smpc_sim/rl/README.md`).

## 구성

| 경로 | 내용 |
|---|---|
| `analysis/` | MORAI bag 측정 (차선 변경 시간, 복귀, NPC 가감속, mode 감지 지연, GT 인지 검증) |
| `docs/baseline_spec.md` | 기존 SMPC 판단 사양 (파이썬 이식의 근거) |
| `sim/smpc_sim/` | 1 차원 Frenet 고속도로 시뮬레이터, 기존 SMPC 이식, 트리 플래너, RL |
| `sim/tests/` | 파이썬 이식 ↔ 원본 C++ 수치 비교 (위험도, 트리 평가, 실행 중 재계획) |
| `cpp_check/` | 원본 노드 소스를 include 하는 비교 하네스 |
| `morai/` | MORAI 자동화(`reset_morai.py`, `xctl.py`)와 연구 시나리오 S1–S4 |
| `results/` | 측정값, 시뮬레이션 결과 json |

C++ 쪽 변경(모두 기본 꺼짐, 끄면 기존 동작과 같다):
- `smpc_decision_node`: `tree_planner_enabled`, `tree_keep_gates`, `/smpc/tree_abort`, `/smpc/tree_debug`
- `lane_change_supervisor_node`: `tree_abort_enabled` (켜면 rear_abort 대신 트리 재계획을 따른다)
- `target_selector_node`: `object_topic_offset_x/y` (/Object_topic 로컬 좌표 → UTM)

## 실행

```bash
cd research
python3 -m unittest sim/tests/test_risk_vs_cpp.py sim/tests/test_tree_vs_cpp.py   # C++ 일치 확인
python3 sim/run_batch.py --planner baseline --n 200 --seed0 2000                # 기존 SMPC
python3 sim/run_batch.py --planner tree_gt_nogate --n 200 --seed0 2000          # 트리 + 게이트 정리
cd sim && python3 -m smpc_sim.rl.train_value --iters 3 --episodes 300           # V 학습

# MORAI (GT 인지, 집 PC sim_ip 192.168.100.10)
roslaunch smpc_lane_change research_morai.launch method:=baseline
roslaunch smpc_lane_change research_morai.launch method:=tree tree_keep_gates:=false
```

## 지금까지 결과 (시뮬레이터, 200 에피소드, 같은 시드 2000–2199)

| 방법 | 성공 | 충돌 | 합류 시간 중앙값 | 차선 끝 정지/에피소드 | 기존 대비 합류 시간 (95% CI) |
|---|---|---|---|---|---|
| 기존 SMPC | 91.0% | 0% | 49.7 s | 0.95 | — |
| 기존 SMPC, 게이트만 제거 | 89.0% | 6.5% | 36.6 s | 0.80 | −6.0 s [−10.7, −1.3] |
| 트리 (게이트 유지) | 91.0% | 0% | 46.2 s | 0.91 | −1.3 s [−5.1, +2.3] |
| 트리 + 게이트 정리 | 92.0% | 0% | 29.8 s | 0.60 | −14.9 s [−19.5, −10.4] |

실패는 120 s 로 검열한 짝비교(paired bootstrap). 시뮬레이터 NPC 는 MORAI 실측 분포 + IDM 으로 움직여
예측 모델과 다르다. MORAI 검증 전까지는 시뮬레이터 결과다.
