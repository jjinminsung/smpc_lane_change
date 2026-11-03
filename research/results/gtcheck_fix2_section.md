## Re-verification 2 (2026-11-04): duplicate-frame drop + backward ego extrapolation

Node rebuilt 01:03 with `object_topic_drop_duplicate_frames: true` (objectCallback returns without publishing when the (unique_id, position.x/y, velocity.x) list equals the previous frame's) and the ego extrapolation clamp widened to [-0.1, +0.1] s. Replays with `gtcheck_replay2.py` (master 11411, --clock): the live UDP bag `func2_baseline_S2` with the research_morai settings (offset 0/0, heading offset 0; inputs /Object_topic /odom /gps_state /smpc/lane_change_status) -> `gtcheck_replay/func2_baseline_S2_fix2.bag`, and the 3 September ROS-path bags with the yaml settings -> `<bag>_fix2.bag`. 'Before' = the live node recording in the bag (previous build: no drop, clamp [0, 0.1]) and the re-verification-1 replays `<bag>_fix.bag`. Script `research/analysis/gtcheck_fix2.py`, numbers `gtcheck_fix2.json`.

### Live UDP bag (func2_baseline_S2)

| # | check | before (live recording) | after (rebuilt node) | result |
|---|---|---|---|---|
| (a) | outputs taken from a copy frame (the original was dropped by the queue-1 subscriber and the copy published instead; content still new) | 49.8% | 9.1% | info |
| (a) | consecutive outputs < 5 ms apart / same NPC content as previous output | 49.0% / 49.5% | 0.0% / 0.0% | **PASS** |
| (a) | input frames not published: copies / originals (an original counts here when its copy was published instead) | 0 / 10 | 1281 / 130 | **PASS** |
| (a) | distinct states never published (neither original nor copy) | - | 0 | **PASS** |
| (a) | output count / rate / max output gap | 2831 / 31.7 Hz / 0.112 s | 1432 / 16.0 Hz / 0.114 s | **PASS** |
| (b) | observed-accel EMA / true MORAI accel, target_rear, sustained accel > 0.5: median (p10..p90), n | 0.94 (0.58..1.05), n=202 | 0.98 (0.90..1.04), n=101 | **PASS** |
| (b) | same, every vehicle in the output (roles + nearby), whole bag | 0.58 (0.36..1.01), n=5450 | 0.95 (0.86..1.04), n=2753 | **PASS** |
| (b) | same, t >= 20 s only (copy gap > 1 ms there: 97% of copies pass the dt > 1e-3 guard vs 10% before 20 s) | 0.49 (0.37..0.60), n=12 | 0.89 (0.87..0.96), n=6 | **PASS** |
| (c) | node ego_s - /odom at frame stamp, moving: p10 / p50 / p90 / max abs | -0.08 / 0.00 / 0.02 / 0.42 m | -0.08 / 0.00 / 0.02 / 0.41 m | **PASS** |
| (c) | frames whose newest /odom is newer than the frame: share, error p10..p90 | 0%, 0.03..0.03 m | 0%, n/a..n/a m | **PASS** |
| (c) | E1 edge flips (role mismatches cf+tf+tr) | 0 | 0 | **PASS** |
| (d) | A2 roles, node's own ego (E2) | 100% | cf 100.00%, tf 100.00%, tr 100.00% | **PASS** |
| (d) | A2' roles, synchronous ego (E1) | cf 100.00%, tf 100.00%, tr 100.00% | cf 100.00%, tf 100.00%, tr 100.00% | **PASS** |
| (d) | A1 lane_id, lane-centred NPC-frames | 100.00% (15289) | 100.00% (7706), 5 wrong, 0 missing | **PASS** |
| (d) | A4 delta_s vs centre (E1), target_rear p50 / p90 / max | 0.00 / 0.03 / 0.42 m | 0.00 / 0.03 / 0.41 m | **PASS** |
| (d) | A3 v_long vs velocity.x/3.6 | 0.0000 | 0.0000 m/s | **PASS** |

### September ROS-path bags (/Ego_topic = exact ego truth)

| bag | input dup frames | outputs from dup frames | distinct frames not published | accel EMA ratio fix1 -> fix2 (median, n) | ego_s - /Ego_topic p10 / p50 / p90, fix1 -> fix2 | share odom newer than frame; its error p90, fix1 -> fix2 | E1 agree cf / tf / tr (fix2) | E1 mismatches fix1 -> fix2 | A1 centred | A4 dS E1 target_rear p90 / max | max output gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-09-28-11-05-29 | 2.9% | 1.5% | 999 | 1.01 (1265) -> 1.01 (1282) | -0.08 / 0.00 / 0.12 -> -0.08 / 0.00 / 0.09 | 24%, 0.31 -> 22%, 0.04 m | 100.00% / 100.00% / 100.00% | 0 -> 0 | 99.95% | 0.01 / 0.25 m | 0.127 s |
| 2026-09-28-11-30-19 | 14.7% | 4.1% | 838 | 1.01 (288) -> 1.03 (277) | -0.10 / 0.01 / 0.31 -> -0.16 / 0.00 / 0.16 | 35%, 0.64 -> 31%, 0.04 m | 100.00% / 99.92% / 100.00% | 0 -> 1 | 100.00% | 0.15 / 0.68 m | 1.116 s |
| 2026-09-21-15-43-18 | 1.5% | 0.7% | 603 | 1.03 (149) -> 1.03 (166) | -0.05 / 0.01 / 0.20 -> -0.07 / 0.00 / 0.10 | 34%, 0.43 -> 35%, 0.06 m | 100.00% / 99.74% / 99.84% | 9 -> 8 | 100.00% | 0.11 / 0.48 m | 0.105 s |


New caveat (dedup rule): the rule drops any frame equal to the previous one, not only the 1 ms UDP copy. When MORAI stalls, /Object_topic repeats one state with fresh stamps while the ego is frozen too (NPC velocity.x non-zero but positions constant). Across the 54 bags, 2863 of 138248 frames (2.07%) are such distinct-time repeats; 13 bags have a stall longer than 0.35 s. Longest: 2026-09-04-15-00-35 12.8 s; 2026-09-19-07-18-59 8.4 s; 2026-09-08-17-35-18 2.7 s; 2026-09-10-13-15-15 2.3 s. During a stall the node now publishes nothing, so consumers see /smpc/targets go stale (0.35/0.5 s freshness gates; SMPC vehicle histories reset after 0.35 s, re-arming the 0.8 s new-rear-track guard on resume). The max output gap in the fix2 replays is in the table above. Suggested guard: drop a repeat only when it arrives within ~5 ms of the previous frame (the UDP copy is 1.0-1.5 ms), or republish at least every 0.1 s.

Accel ratio note: the dt > 1e-3 guard in smpc_decision already skipped most copies during the first 20 s of this bag (copy gap <= 1 ms in 90% of them), where most target_rear acceleration samples fall. That is why the target_rear 'before' median is 0.94. After 20 s the copy gap is ~1.3 ms, the copies pass the guard, and the ratio halves (0.49), as the all-NPC GT simulation in Re-verification 1 showed (0.56). With the drop rule the ratio is ~0.9-1.0 everywhere.

Edge flips: the remaining E1 mismatches are NPCs alongside the ego whose front/rear edge is within 0.0-0.3 m of zero. There the reference itself (/odom interpolated) differs from /Ego_topic by -0.02..-0.09 m, so these are ties, not a defect. On the September bags the backward dt removes the ego-ahead tail: in frames where /odom is newer than the frame, ego error p90 0.31-0.64 m -> 0.04-0.06 m. The live UDP bag has no such frames (its /Object_topic stamp is the receive time), so there the clamp change is a no-op.

Remaining E1 mismatches (largest): 09-28-11-30-19 t=17.86 target_front node -1 / GT 14 (gap 120.0 vs 0.0); 09-21-15-43-18 t=0.19 target_rear node 38 / GT -1 (gap 0.0 vs 120.0); 09-21-15-43-18 t=0.20 target_rear node 38 / GT -1 (gap 0.0 vs 120.0); 09-21-15-43-18 t=0.23 target_rear node -1 / GT 38 (gap 120.0 vs 0.0); 09-21-15-43-18 t=32.11 target_front node -1 / GT 18 (gap 120.0 vs 0.0); 09-21-15-43-18 t=32.13 target_front node -1 / GT 18 (gap 120.0 vs 0.0).

