### Live MORAI bag, UDP object path (`func2_baseline_S2.bag`)

Scenario research_S2_rear_accel, method baseline, recorded 00:51-00:53 with `target_source:=object_topic`. /Object_topic = `morai_udp_bridge/morai_object_info_node.py` (UTM positions, grid heading = true-north heading - 1.347 deg, stamp = PC receive time). Live selector params (launch log): offset 0/0, heading offset 0, ref_to_center 1.25 m, long vehicle 7.5 m / 1.1 m, extrapolate_ego true. The recorded /smpc/targets is the live node output (no replay). GT = /Object_topic + the same shift; ego = /odom (no /Ego_topic in the bag, and the Object Info stream carries no ego record). 2849 /Object_topic frames, 2831 /smpc/targets, 4 LC windows.

| # | check | live result | result |
|---|---|---|---|
| R0 | published x/y = /Object_topic position + shift | max 0.0000 m from the shifted GT (median 1.25 m from the raw position = the shift) | **PASS** |
| A1 | lane_id within +-120 m vs GT lane | 100.00% of 15289 lane-centred NPC-frames, 10/15835 wrong, 0 missing; 2986 NPC-frames outside the 4 CSVs (2136 in nearby) | **PASS** |
| A1' | nearby port == node lane_id | 100.00% (port-only 0, node-only 0) | **PASS** |
| A2 | roles vs GT, node ego (E2) | cf 100.00%, tf 100.00%, tr 100.00% (LC windows: tf 100.00%, tr 100.00%) | **PASS** |
| A2' | roles vs GT, /odom at frame stamp (E1) | cf 100.00%, tf 100.00%, tr 100.00% (LC windows: tf 100.00%, tr 100.00%) | **PASS** |
| A3 | v_long vs velocity.x/3.6 | max 0.0000 m/s (target_rear n=1968, nearby n=20376) | **PASS** |
| A3' | v_lat (vehicle frame, known limitation) | missed lane-lateral motion in 22.4% of nearby entries | known limitation |
| A4 | delta_s vs centre, E2 (implementation) | p90 0.0000 m, max 0.54 m (the max is only where the ego is past the lane-2 CSV end and ego_s is clamped, so rebuilding the ego from (s, d) is not exact; 6 entries) | **PASS** |
| A4 | delta_s vs centre, E1: target_rear p50 / p90 / max | 0.00 / 0.03 / 0.42 m (nearby p90 0.02, max 0.45) | **PASS** |
| A4 | node ego_s - /odom at frame stamp, moving: p10 / p50 / p90 | -0.08 / 0.00 / 0.02 m (implied lag median -0.000 s) | **PASS** |
| A5 | heading - lane yaw, lane-centred moving NPCs (median, p10..p90) | 0.017 deg (-0.22..0.35), n=24552 | **PASS** |
| U2 | heading already grid-corrected: heading - course on straights | 0.014 deg (p10..p90 -0.01..0.05; September ROS path: +1.361) | **PASS** |
| U1 | position still the rear axle (kinematic x_ref) | 3-9 m/s: -0.17 m (n=4587, abs kappa>0.008); 9-16 m/s: -0.42 m (n=5297, abs kappa>0.008); 16-30 m/s: -0.88 m (n=224, abs kappa>0.008) | **PASS** |
| U3 new | duplicate /Object_topic frames (UDP path) | 49.7% of frames are exact copies of the previous one, 1.3 ms later; distinct states every 62.8 ms (p90 72.4) = 15.9 Hz | **FAIL** (new) |
| U3 effect | SMPC observed-accel EMA / true MORAI accel, sustained accel > 0.5 m/s^2 | all frames 0.56 (p10..p90 0.36..1.00) vs duplicates removed 0.95 | **FAIL** (new) |
| A6 | rate | in 31.8 Hz (header), out 31.7 Hz, 99.6% of frames processed, 5.1% identical consecutive outputs, output dt p90 0.068 s / max 0.112 s | **PASS** |

Role counts (E1): current_front GT valid 441, wrong 0, missed 0, extra 0; target_front GT valid 1813, wrong 0, missed 0, extra 0; target_rear GT valid 1968, wrong 0, missed 0, extra 0. /Object_topic receive - stamp median 0.3 ms; object dt median 37.9 ms (p90 67.5); /odom dt median 20.0 ms.

U3: every Object Info state arrives twice ~1 ms apart. The copy is harmless for positions (1 ms), but smpc_decision's per-vehicle acceleration EMA (`dt > 1e-3`, a = 0.35 raw + 0.65 a) takes the copy as a dv = 0 sample, so the observed TV acceleration settles at ~0.4-0.6x the truth. That under-reads rear acceleration, which this rear_accel scenario is about. Suggested fix: dedup in `morai_object_info_node.py` by the packet sim stamp (it is decoded and then discarded) or by content, and/or skip repeated states in `objectCallback`. A6 counts the copies as frames, so the real object update rate is the distinct-state rate above.

Caveats: the E1 'synchronous' ego here is /odom interpolated at the /Object_topic receive stamp; the relative latency of the GPS and Object Info UDP streams is not observable in this bag. The UDP record also carries overhang / wheelbase / rear_overhang per object (object_info.py), which the ROS bridge drops; publishing them would let the selector use the exact L/2 - rear_overhang shift instead of the 1.25 m constant.

