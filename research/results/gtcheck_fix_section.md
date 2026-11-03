## Re-verification after fixes (2026-11-04)

Fix under test (uncommitted `src/target_selector_node.cpp` + `config/target_selector.yaml`, node rebuilt 00:40): `yaw = (npc.heading - 1.365 deg)`, position moved from the rear axle to the bbox centre by 1.25 m (L < 7.5 m) or 0.5 L - 1.1 m along the corrected heading, ego = /odom extrapolated to the /Object_topic stamp (dt in [0, 0.1] s). Same 3 bags, same replay recipe (master 11411, --clock), output `gtcheck_replay/<bag>_fix.bag`. GT side updated to the same model (centre = rear axle + shift, corrected yaw); script `research/analysis/gtcheck_fix.py`, numbers in `gtcheck_fix.json`. v_lat is unchanged on purpose (still vehicle-frame) and is listed as a known limitation.

| # | check | before fix | after fix | result |
|---|---|---|---|---|
| R0 | published x/y = npc.position + offset + shift | - | max deviation from the fixed GT centre 0.0000 m | **PASS** |
| A1 | lane_id within +-120 m vs GT lane | 99.98% centred, 22 wrong | 99.98% centred (n=33394), 23/33874 wrong, 0 missing | **PASS** |
| A2 | roles vs GT, node ego (E2) | 100% | cf 100.00%, tf 100.00%, tr 100.00% | **PASS** |
| A2' | roles vs GT, synchronous ego (E1) | cf 99.98%, tf 99.77%, tr 99.97% (tf: 4 wrong / 3 missed / 7 extra) | cf 100.00%, tf 99.91%, tr 99.93% (tf: 0 / 4 / 1) | **PASS** |
| A2' LC | same, LC windows | tf 99.61%, tr 99.96% | tf 99.79%, tr 100.00% | info |
| A3 | v_long vs velocity.x/3.6 | 0.0000 | max 0.0000 m/s | **PASS** |
| A3' | v_lat (vehicle frame) | latent | unchanged by design: missed lane-lateral motion in 4.9%-17.5% of nearby entries | **known limitation** |
| A4 | delta_s vs true centre (E1), target_rear p90 / max | 0.63 / 1.67 m | 0.27 / 1.49 m | **PASS** |
| A4 | delta_s vs true centre (E1), nearby p90 / max | 0.57 / 1.67 m | 0.21 / 1.49 m | **PASS** |
| A4 | ego_s error while moving, median (p10..p90) | -0.17 (-0.47..0.02); -0.22 (-0.81..0.25); -0.16 (-0.49..0.16) | 0.02 (-0.05..0.12); 0.05 (-0.10..0.36); 0.03 (-0.04..0.19) | **PASS** |
| A4 | node ego_s - /Ego_topic (true rear axle), moving: p10 / p50 / p90 | -0.52 / -0.17 / 0.00; -0.87 / -0.24 / 0.22; -0.50 / -0.15 / 0.18 | -0.08 / 0.00 / 0.12; -0.10 / 0.01 / 0.31; -0.05 / 0.01 / 0.20 | **PASS** |
| A4 new | frames whose newest /odom is newer than the frame (dt<0, clamped to 0): share, ego_s error p90 | - | 24%, p90 0.31 m (vs 0.10 m when dt>=0); 35%, p90 0.64 m (vs 0.18 m when dt>=0); 34%, p90 0.43 m (vs 0.12 m when dt>=0) | **WARN** (new) |
| A5 | heading - lane yaw (median per bag) | +1.36 deg | -0.008, 0.003, -0.003 deg | **PASS** |
| A6 | output Hz / frames processed / identical outputs | 32.1-37.8 Hz / 66-76% / 0.8-10.7% | 27.2-36.1 Hz / 56%-73% / 0.3%-4.4% | WARN (unchanged) |
| B | LiDAR rear face (car ahead, bumper zone) residual, median (p10..p90) | 1.32 (1.09..1.52) m | 0.07 (-0.16..0.27) m, per-NPC median 0.04 m | **PASS** |
| B | LiDAR front face (car behind, bumper zone) residual, median (p10..p90) | 1.20 (-0.36..1.41) m | -0.05 (-1.61..0.16) m, per-NPC median -0.05 m | **PASS** |
| B | LiDAR rear face (all points) residual, median (p10..p90) | 1.41 (1.19..1.81) m | 0.16 (-0.08..0.48) m, per-NPC median 0.15 m | **PASS** |
| B | LiDAR front face (all points) residual, median (p10..p90) | 1.04 (-0.28..1.33) m | -0.21 (-1.53..0.08) m, per-NPC median -0.29 m | **PASS** |

Per-bag detail (after the fix):

| bag | msgs | E1 agree cf / tf / tr | E1 tf wrong / missed / extra | dS E1 target_rear p50 / p90 / max | ego_s err moving p50 / max abs | ego lag implied p50 / p90 [s] | out Hz | processed |
|---|---|---|---|---|---|---|---|---|
| 2026-09-28-11-05-29 | 2382 | 100.00% / 100.00% / 100.00% | 0 / 0 / 0 | 0.00 / 0.02 / 0.67 | 0.016 / 0.97 | -0.002 / 0.004 | 34.1 | 68.9% |
| 2026-09-28-11-30-19 | 1348 | 100.00% / 100.00% / 100.00% | 0 / 0 / 0 | 0.01 / 0.27 / 1.25 | 0.046 / 1.25 | -0.003 / 0.005 | 27.2 | 55.7% |
| 2026-09-21-15-43-18 | 1828 | 100.00% / 99.73% / 99.78% | 0 / 4 / 1 | 0.03 / 0.21 / 1.49 | 0.026 / 1.49 | -0.002 / 0.003 | 36.1 | 72.6% |

Face residual by length class (after, all scans, bumper zone): 3.5-4.7: ahead 0.04 / behind -0.03; 4.7-5.2: ahead 0.08 / behind -0.06; 5.2-7.5: ahead 0.08 / behind 0.01; 7.5-13.0: ahead 0.41 / behind n/a m.

Worst remaining E1 role mismatches:

| bag | t [s] | role | node uid | GT uid | node gap | GT gap |
|---|---|---|---|---|---|---|
| 2026-09-21-15-43-18 | 0.19 | target_rear | 38 | -1 | 0.00 | 120.00 |
| 2026-09-21-15-43-18 | 0.20 | target_rear | 38 | -1 | 0.00 | 120.00 |
| 2026-09-21-15-43-18 | 0.23 | target_rear | -1 | 38 | 120.00 | 0.00 |
| 2026-09-21-15-43-18 | 30.67 | target_rear | 18 | -1 | 0.00 | 120.00 |
| 2026-09-21-15-43-18 | 31.96 | target_front | -1 | 18 | 120.00 | 0.00 |
| 2026-09-21-15-43-18 | 32.11 | target_front | -1 | 18 | 120.00 | 0.00 |
| 2026-09-21-15-43-18 | 32.13 | target_front | -1 | 18 | 120.00 | 0.00 |
| 2026-09-21-15-43-18 | 32.15 | target_front | -1 | 18 | 120.00 | 0.00 |

Remaining physical-role gaps (spec filter by design): 09-28-11-05-29 current_front 97.6%; 09-28-11-05-29 target_front 98.9%; 09-28-11-05-29 target_rear 95.0%; 09-28-11-30-19 current_front 98.7%; 09-28-11-30-19 target_front 91.2%; 09-28-11-30-19 target_rear 100.0%; 09-21-15-43-18 current_front 97.4%; 09-21-15-43-18 target_front 99.6%; 09-21-15-43-18 target_rear 99.7%.

LC-window target_rear rear bumper gaps (now centre-based): 09-28-11-05-29: 728 frames, 106 < 9 m, 44 in [9, 15) m; 09-28-11-30-19: 322 frames, 24 < 9 m, 12 in [9, 15) m; 09-21-15-43-18: 714 frames, 165 < 9 m, 29 in [9, 15) m.

