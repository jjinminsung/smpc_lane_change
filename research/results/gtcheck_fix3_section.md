## Re-verification 3 (2026-11-04): duplicate drop limited to repeats within 5 ms

Node rebuilt ~01:19. A frame is dropped only if its (unique_id, position.x/y, velocity.x) list equals the last published frame's AND its stamp is 0-5 ms after that frame. Replays with `gtcheck_replay2.py` (master 11411, --clock, suffix `_fix3`): func2_baseline_S2 with the research_morai settings (offsets 0, heading 0), and 2026-09-28-11-30-19 / 2026-09-04-15-00-35 with the yaml settings. 'fix2' = Re-verification 2 build (drop on any repeat), 'fix1' = Re-verification 1 build (no drop). Script `research/analysis/gtcheck_fix3.py`, numbers `gtcheck_fix3.json`.

### (1) Live UDP bag func2_baseline_S2

| metric | live recording (no drop) | fix2 (drop any repeat) | fix3 (drop repeats < 5 ms) | result |
|---|---|---|---|---|
| output count / rate | 2831 / 31.7 Hz | 1432 / 16.0 Hz | 1446 / 16.2 Hz | **PASS** |
| consecutive outputs < 5 ms apart | 49.0% | 0.0% | 0.0% | **PASS** |
| outputs with the same NPC content as the previous output | 49.5% | 0.0% | 1.0% | **PASS** |
| identical consecutive /smpc/targets (all fields but header) | 1.2% | 0.0% | 0.0% | **PASS** |
| distinct object states never published | 0 | 0 | 0 | **PASS** |
| accel EMA / true accel, all vehicles, sustained > 0.5: median (p10..p90), n | 0.58 (0.36..1.01), 5450 | 0.95 (0.86..1.04), 2753 | 0.95 (0.85..1.04), 2780 | **PASS** |
| same, target_rear only | 0.94 (n=202) | 0.98 (n=101) | 0.97 (n=102) | **PASS** |
| output dt p90 / max | 0.068 / 0.112 s | 0.072 / 0.114 s | 0.072 / 0.114 s | **PASS** |

### (2) ROS-path bags with MORAI stalls

| bag | build | outputs | rate | longest stall (t, span) | outputs inside it | max gap inside any stall | max output gap overall | identical consecutive msgs: all / outside stalls / inside stalls | result |
|---|---|---|---|---|---|---|---|---|---|
| 2026-09-28-11-30-19 | fix | 1348 | 27.2 Hz | 38.5 s, 1.07 s | 25 | 0.085 s | 0.140 s | 1.5% / 0.6% / 12.0% |  |
| 2026-09-28-11-30-19 | fix2 | 1279 | 25.8 Hz | 38.5 s, 1.07 s | 1 | 1.073 s | 1.116 s | 0.0% / 0.0% / 0.0% |  |
| 2026-09-28-11-30-19 | fix3 | 1422 | 28.7 Hz | 38.5 s, 1.07 s | 27 | 0.101 s | 0.140 s | 0.4% / 0.0% / 4.4% | **PASS** |
| 2026-09-04-15-00-35 | fix3 | 3282 | 38.4 Hz | 30.2 s, 12.80 s | 612 | 0.068 s | 0.097 s | 0.8% / 0.0% / 3.0% | **PASS** |

Inside a stall the identical outputs are expected: MORAI repeats one state and the ego is frozen too, so the node correctly republishes the same content with fresh stamps. Outside stalls the identical-consecutive fraction is the ROS path's own repeated frames.

