# Day-1 measurements (baseline dd1cf54, 53 bags)

Generated 2026-11-03T23:29:38 by `research/analysis/run_day1.py`. Bags: 53 `lc_gt_*.bag` (2026-09-04-14-59-42 .. 2026-09-28-11-46-42). Time axis = header.stamp. Ego lateral = /odom rear axle on lane CSVs (`waypoint_system/data/default`, matches /smpc/targets ego_d to p99 < 0.13 m). Full numbers: `day1_measurements.json`.

## M1 Lane-change duration (from status rising edge = /path_number publish)

Population: completed changes with the start inside the bag. Times in s after t0: median (p10-p90), mean.

| population | N | cross boundary (abs(d_tgt) <= W/2) | abs(d_tgt) <= 0.45 m | supervisor complete |
|---|---|---|---|---|
| all completed | 119 | 2.86 (2.66-3.03), mean 2.84 | 4.30 (3.61-4.48), mean 4.18 | 4.20 (3.56-4.34), mean 4.09 |
| moving start (v0>=3, BLEND) | 68 | 2.87 (2.72-3.00), mean 2.87 | 4.34 (4.29-4.52), mean 4.44 | 4.26 (4.19-4.41), mean 4.34 |
| standstill start (CONNECTOR) | 49 | 2.77 (2.59-3.04), mean 2.81 | 3.81 (3.53-4.08), mean 3.83 | 3.74 (3.49-4.02), mean 3.75 |
| moving 0->1 | 8 | 2.85 (2.73-2.90), mean 2.83 | 4.34 (4.22-4.53), mean 4.36 | 4.28 (4.22-4.34), mean 4.28 |
| moving 1->2 | 34 | 2.81 (2.71-2.94), mean 2.83 | 4.33 (4.30-4.66), mean 4.52 | 4.26 (4.20-4.39), mean 4.40 |
| moving 2->3 | 26 | 2.90 (2.80-3.10), mean 2.94 | 4.35 (4.28-4.49), mean 4.37 | 4.25 (4.19-4.41), mean 4.28 |

Doc value 4.16 s: matches only the pooled mean of two different populations (all completed: mean 4.18, supervisor mean 4.09). For a moving-start change (BLEND) the 0.45 m time is median 4.34 / p90 4.52 s, so 2.0 + 0.55 + 4.34 + 1.5 = 8.39 s vs the 8.0 s horizon (doc: 8.21 s).

Control (all 144 accepted attempts): abort_rear 24, complete 118, complete_inferred 1, complete|partial_start 1. Standstill starts: motion begins 0.49 s (median) after t0.

Lateral progress 1 - abs(d_tgt)/abs(d_tgt(t0)) (median):

| t [s] | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 | 4.5 | 5.0 | 5.5 | 6.0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| moving | 0.00 | 0.01 | 0.06 | 0.18 | 0.34 | 0.46 | 0.56 | 0.67 | 0.80 | 0.92 | 0.98 | 0.96 | 0.93 |
| standstill | 0.00 | 0.00 | 0.00 | 0.05 | 0.17 | 0.39 | 0.63 | 0.81 | 0.92 | 0.98 | 0.98 | 0.97 | 0.95 |

By date (completed, 0.45 m median / supervisor median): 09-07 n=1 3.85/3.67; 09-08 n=11 4.30/4.19; 09-10 n=26 4.28/4.18; 09-14 n=6 4.38/4.36; 09-15 n=21 3.91/3.87; 09-16 n=12 4.33/4.27; 09-17 n=18 4.30/4.20; 09-19 n=12 4.39/4.26; 09-21 n=9 4.36/4.26; 09-28 n=3 4.12/4.10

## M2 Rear-abort return

24 rear aborts (confirmed 21, emergency 3) in 144 attempts; 18 never left the source lane (abs(d_src) stayed <= 0.45 m).
Never-departed aborts: 15 standstill / 3 moving, abort 0.50 s (median) after t0, lateral progress <= 0.04 m.

Aborts where the ego left the source lane (after the abort the reference path is re-blended from the target lane, so a moving ego first swings into the target lane):

| bag | pair | v@abort | t abort after start | progress@abort m (frac) | peak abs(d_src) after abort (t) | back <= 0.45 m after | outcome |
|---|---|---|---|---|---|---|---|
| 2026-09-08-17-52-30 | 0->1 | 7.0 | 0.51 | 0.01 (0.00) | 1.75 m (2.11 s) | never | stopped_between_lanes |
| 2026-09-10-13-02-42 | 1->2 | 11.3 | 1.68 | 0.77 (0.21) | 3.83 m (2.48 s) | 4.21 s | returned |
| 2026-09-10-13-02-42 | 1->2 | 10.5 | 0.09 | 0.11 (0.03) | 3.74 m (1.83 s) | 3.12 s | returned |
| 2026-09-10-13-16-44 | 1->2 | 11.9 | 1.02 | 0.22 (0.06) | 3.85 m (1.76 s) | 3.11 s | returned |
| 2026-09-17-12-47-05 | 1->2 | 12.6 | 1.22 | 0.34 (0.09) | 1.00 m (0.87 s) | 1.45 s | returned |
| 2026-09-28-11-30-19 | 1->2 | 7.3 | 3.35 | 1.92 (0.53) | 2.44 m (0.87 s) | never | stopped_between_lanes |

## M3 NPC longitudinal behaviour (GT, unique_id)

1152 highway NPC tracks (546 NPC-min), 459 spawned in view. /Object_topic only lists NPCs within ~300 m of the ego. MORAI acceleration.x vs our derivative: median corr 0.98.

| quantity | all highway NPCs | relevant rear (target lane, -120..0 m) | relevant front (0..+100 m) |
|---|---|---|---|
| time fraction a>0.3 m/s^2 | 0.30 | 0.50 | 0.32 |
| time fraction a>0.5 m/s^2 | 0.26 | 0.47 | 0.28 |
| time fraction a>1.0 m/s^2 | 0.16 | 0.34 | 0.12 |
| time fraction a<-0.3 m/s^2 | 0.23 | 0.20 | 0.22 |
| (window, NPC) pairs / unique NPCs | - | 382 / 267 | 237 / 183 |
| P(accel episode overlaps window) | - | 0.60 | 0.35 |
| ... of which spawn-acceleration episode | - | 0.72 | 0.02 |
| P(accel onset inside window) | - | 0.07 | 0.11 |
| accel episode mean a (median, p10-p90) | 1.01 (0.54-1.66) | 1.45 (0.81-1.60) | 0.98 (0.63-1.44) |

Rear P(accelerating) by lane pair: 0->1 0.85 (n=220); 1->2 0.32 (n=101); 2->3 0.20 (n=61); lc_attempt 0.57, lane_end_wait 0.64.

Parametric summary (all highway NPCs; median [p10, p90]):

| parameter | value |
|---|---|
| spawn speed | median 13.82 m/s; 61% at 13.89 m/s, 25% at 0 m/s |
| spawn acceleration | 1.45 [0.85, 1.67] (n=443) |
| spawn acceleration duration [s] | 7.65 [3.70, 12.44] (n=443) |
| speed at end of spawn acceleration | 23.63 [9.66, 28.70] (n=443) |
| cruise plateau speed (abs(a)<0.15 for >=2 s) | 11.11 [0.09, 16.66] (n=1394) |
| non-spawn accel episode mean a | 1.01 [0.54, 1.66] (n=1642) |
| accel episode duration [s] | 3.60 [1.45, 9.05] (n=1642) |
| accel episode dv [m/s] | 3.59 [1.28, 10.68] (n=1642) |
| inter-onset interval of accel episodes [s] | 12.05 [3.35, 38.89] (n=813) |
| decel episode mean abs(a) | 1.35 [0.69, 2.51] (n=1734) |
| decel episode duration [s] | 3.57 [1.30, 10.08] (n=1734) |
| accel / decel episode rate | 3.01 / 3.18 per NPC-min |
| accel mean-a histogram [0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 4.0] | [0.074, 0.202, 0.218, 0.335, 0.154, 0.016, 0.001] |

NPC lane changes: yes. 82 mid-lane changes by 64/1152 NPCs (0.15 per NPC-min; 0->1 2, 1->2 5, 2->1 34, 2->3 17, 3->2 24), plus 85 lane-end merges and 16 in the upstream extension zone. 23 mid-lane changes happened while the NPC was relevant to a decision window, 15 of them into the ego's target lane. Lateral rate median 0.49 m/s.

## M4 Mode detection delay (filter a = 0.35 raw + 0.65 a, raw = clamp(dv/dt, +-6))

Update rate: recorded /smpc/targets (LiDAR mode) 9.38 Hz median per bag (gaps > 0.35 s: 0.05%); object_topic mode = one /smpc/targets per /Object_topic message = 49.0 Hz (dt median 0.018 s, 3.3% of dt <= 1 ms and skipped by the 1e-3 guard; 6% of moving-NPC frames repeat the previous state).

Delay [s] from true onset (foot of the GT acceleration rise) to filtered > threshold, onsets from steady (abs(a) < 0.2 for 1 s before). median (p10-p90), detected fraction:

| true mean accel | N | GT @Object_topic >0.5 | >1.0 | GT @9.4 Hz >0.5 | >1.0 | tracker >0.5 (first) | >0.5 (3 consecutive) |
|---|---|---|---|---|---|---|---|
| 0.30-0.75 | 107 / trk 45 | 0.56 (0.18-1.65) 98% | 1.29 (0.42-2.80) 43% | 0.89 (0.52-1.95) 95% | 1.82 (1.11-3.39) 19% | 0.42 (0.04-1.19) 98% | 0.91 (0.08-2.06) 87% |
| 0.75-1.25 | 103 / trk 69 | 0.29 (0.20-0.83) 100% | 0.56 (0.25-1.46) 100% | 0.55 (0.35-1.01) 100% | 0.95 (0.61-1.97) 100% | 0.35 (0.04-1.23) 96% | 0.66 (0.03-1.82) 87% |
| 1.25-2.00 | 64 / trk 70 | 0.24 (0.19-0.32) 100% | 0.35 (0.21-0.46) 100% | 0.42 (0.32-0.53) 100% | 0.58 (0.42-0.80) 100% | 0.71 (0.05-1.31) 100% | 0.86 (0.12-1.60) 99% |
| all | 274 / trk 188 | 0.30 (0.19-1.12) 99% | 0.52 (0.23-1.82) 78% | 0.57 (0.37-1.48) 98% | 0.83 (0.49-2.12) 68% | 0.48 (0.04-1.28) 97% | 0.83 (0.06-1.89) 91% |

Filter lag behind the true acceleration crossing the same threshold (median): GT@Object_topic 0.04 s, GT@9.4 Hz 0.27 s. The rest of the delay is the GT acceleration ramp itself.

False positives in GT steady cruise (abs(a_true) < 0.2 over +-1 s, v > 3 m/s):

| input | steady min | frac >0.5 | frac >1.0 | frac <-0.5 | crossings/min >0.5 | >1.0 |
|---|---|---|---|---|---|---|
| GT @Object_topic | 177.5 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 |
| GT @9.4 Hz | 175.9 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 |
| tracker, all matched | 23.6 | 0.309 | 0.174 | 0.261 | 64.3 | 42.4 |
| tracker, target_rear role | 3.7 | 0.304 | 0.180 | 0.327 | 55.6 | 39.7 |
| GT @Object_topic, same target_rear samples | 3.7 | 0.000 | 0.000 | 0.000 | 0.0 | 0.0 |

Tracker vs GT: match distance median 1.57 m; v_long error median -0.23 m/s (p10 -2.19, p90 1.02); 56% of tracker onsets already had filtered > 0.5 in the 1 s before the true onset.

Rear-accel probability table check (target_rear samples; event = GT 5 s displacement >= 12.5 m above constant velocity, the yaml's event definition):

| filtered accel bin | <-0.5 | -0.5..0.5 | 0.5..1.0 | 1.0..1.5 | >=1.5 |
|---|---|---|---|---|---|
| tracker filter: event rate (n) | 0.060 (1844) | 0.109 (1708) | 0.172 (668) | 0.225 (586) | 0.259 (1451) |
| GT@Object_topic filter: event rate (n) | 0.006 (1103) | 0.007 (2580) | 0.079 (554) | 0.275 (1276) | 0.672 (744) |
| yaml table (tracker fit, 95% upper) | 0.11 | 0.15 | 0.28 | 0.30 | 0.31 |

## Caveats

- Bags span 2026-09-04..09-28; src/ is identical between 4fc908c (09-22) and the baseline dd1cf54 (09-28 13:48, after the last bag), only config changed. Only 3 completed changes are in the 09-28 bags, so pooled numbers mix code versions (see M1 by-date row). Lane-change timing is dominated by waypoint_system's 4.0 s blend (path_switch_preview.blend_duration_sec), which is outside this repo.
- Ego lateral offset uses the /odom rear axle, like target_selector and the supervisor; the vehicle centre is ~1.5 m ahead and crosses the boundary slightly earlier.
- /Object_topic and /Ego_topic positions are MORAI-local; /odom and the lane CSVs are UTM. Baseline target_selector objectCallback uses npc.position unchanged, so target_source:=object_topic will not put any NPC on a lane CSV unless the ~(302594.99, 4124145.00) m offset is added.
- /Object_topic only contains NPCs within ~300 m of the ego; NPCs that leave and re-enter keep their unique_id, and the gap is masked out of the acceleration estimate.
- MORAI NPCs are strongly non-stationary: they spawn near the start of lanes 1-3 at 13.89 m/s or 0 m/s and accelerate at ~1.45 m/s^2 for ~8 s. Most target-rear accelerations in 0->1 windows are this spawn acceleration, already running before the window opens (onset inside the window only ~7%).
- Lane-end wait windows: status inactive, lane < 3, ego v < 0.5 m/s, distance_to_lane_end <= 60 m. LC attempt windows include the 2 s before approval.
- The tracker delay is a lower bound polluted by noise: in GT steady cruise the tracker filter exceeds 0.5 m/s^2 ~30% of the time, so 'first crossing after onset' is often noise. Use the sustained (3 consecutive) column and the detection curve in the JSON.
- String topics (supervisor debug, /path_number) carry no header; bag receive time is used. It agrees with the status header stamp to < 1 ms (median), max 35 ms.
- M2 has only 6 aborts with lateral departure. The full swing into the target lane (peak 3.7-3.9 m) is seen in the three 09-10 cases; the 09-17 case peaks at 1.0 m and the 09-28 case (abort at 53% progress) ends stopped straddling the lanes for the rest of the bag. waypoint_system (which re-blends the path on the abort /path_number) is outside this repo and may differ between these dates.
- M4 onset = foot (a <= 0.1) of a centred 0.55 s Savitzky-Golay derivative of GT speed; for a sharp step this foot can lead the true step by up to ~0.2 s, so delays from onset are slight upper bounds. The filter's own lag is the 'lag behind the true crossing' number.
- Object_topic-mode rate assumes target_selector keeps up with 49 Hz (subscriber queue 1; brute-force projection of ~35 NPCs onto all lane CSVs per message) - not measured here.
