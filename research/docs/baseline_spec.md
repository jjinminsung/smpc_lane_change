# Baseline SMPC lane-change stack: decision-relevant spec (dd1cf54)

Source: `~/github/smpc_lane_change` @ `dd1cf54` (branch research/tree-planner). Read-only extraction.
Refs are `file:line`. `smpc` = `src/smpc_decision_node.cpp`, `sup` = `src/lane_change_supervisor_node.cpp`,
`acc` = `src/acc_speed_planner_node.cpp`, `sel` = `src/target_selector_node.cpp`.
Config values come from `config/*.yaml`, which the launch file loads with `rosparam` (no launch overrides,
`launch/smpc_integration.launch`). Many code defaults are different from the yaml values. The yaml wins.

Tags: **[E]** matters for decision outcomes on a straight multi-lane highway. **[P]** exists only for perception
noise or ROS plumbing; a one-line mention is enough. **[X]** depends on an external package that is not in
the baseline repo (waypoint_system, mpc_path_tracking). The version of those packages is uncertain.

Conventions used in all code:
- The ego pose is the **rear axle**. Ego footprint: front 3.845 m, rear 0.790 m, width 1.892 m.
- Frenet `d` is left-positive (`include/smpc_lane_change/lane_path.hpp:98`). An ego centred in the source lane
  has `ego_d_target = -3.5`.
- TV `delta_s`, `front_delta_s`, `rear_delta_s` are measured relative to the **ego rear-axle s** on the TV's lane.
- Prediction grid: `tau_k = 0.2*k`, k = 1..40 (8.0 s). tau = 0 is not part of the grid (`smpc_decision.yaml:8-9`).
- `W = lane_width_m = 3.5`.

---

## 0. Helper formulas the port needs (exact)

| helper | formula | ref |
|---|---|---|
| `vehicleHalfS(v)` | `v.half_s > 1e-3 ? v.half_s : 0.5*max(0, v.length)` | smpc:1625 |
| `frontEdgeDeltaS(v)` | `v.front_delta_s != 0 ? v.front_delta_s : v.delta_s + halfS` | smpc:1629 |
| `rearEdgeDeltaS(v)` | `v.rear_delta_s != 0 ? v.rear_delta_s : v.delta_s - halfS` | smpc:1633 |
| `frontBumperGap(f)` | `rearEdgeDeltaS(f) - 3.845` (can be negative) | smpc:1637 |
| `rearBumperGap(r)` | `-frontEdgeDeltaS(r) - 0.790` (can be negative) | smpc:1641 |
| `activeTargetFrontSafeGap(t)` | constant `target_front_safe_gap_m = 14.0` (lane end never relaxes it) | smpc:1765, yaml:43 |
| `activeTargetRearSafeGap(t)` | constant `target_rear_safe_gap_m = 9.0` | smpc:1773, yaml:44 |
| `activeRiskEpsilon(t)` | constant `risk_epsilon = 0.04` | smpc:1780, yaml:10 |
| `conservativeRearClosingSpeed(r, v_e)` | `c = max(0, r.v_long - max(0,v_e))`; if `age(r) < 0.8 s` (time since the vehicle was first seen, `new_rear_track_guard_sec`) and `rearBumperGap(r) ≤ 40` then `c = max(c, 15.0)` | smpc:1851-1868, yaml:166-168 |
| `targetRearRequiredGap(t, r, T_rem, v_e)` | `base = 9.0`, raised to `max(base, 15.0)` if `v_e ≤ 6.0`; result `= base + c_cons*(T_rem + 0.50)` (`rear_gap_perception_control_delay_sec`) | smpc:1870-1886, yaml:163,204-207 |
| `targetFrontIsPassingVehicle(t)` | `pass_gap_enabled ∧ tf.valid ∧ tf.id ≥ 0 ∧ tf.lane_id == target_lane ∧ (tf.v_long - v_e ≥ 2.0) ∧ frontBumperGap(tf) ≤ 40.0` (an overlapping, negative gap also qualifies) | smpc:1993-2007, yaml:128-130 |
| `targetRearForGap(t, excl_id)` | over `{t.target_rear} ∪ t.nearby_vehicles`: valid, `id ≥ 0`, `id ≠ excl_id`, `lane_id == target_lane`, `rearEdgeDeltaS ≤ 0`. Return the one with minimum `rearBumperGap` (can be negative) | smpc:2009-2031 |
| `effectiveLaneChangeStartSec(spec)` | `change_left ? max(0, t_LC) + 0.55 : +inf` (`behavior_execution_delay_sec`) | smpc:1237, yaml:288 |
| `egoChangeBlendDuration()` | if a stored preview is valid and in BLEND mode: `preview.blend_duration_sec * 1.04` (`ego_reference_preview_duration_scale`). Freshness is **not** checked here. Otherwise `waypoint_blend_duration_sec = 4.0` (not scaled) | smpc:1214-1229, yaml:223,235 |
| `actionTargetSpeedMps(cap, spec)` | KEEP / WAIT / CHANGE_NOW: `max(0,cap)`. Prep actions: `actionLongitudinalRollout(t_sw, cap, spec).v` | smpc:1614 |
| `lowSpeedRearSafetyActive()` | `v_e ≤ 6.0` | smpc:1801 |

`pre_cap` (also called `V_ref`) `= egoPredictionTargetSpeedMps() = min(mission_speed, ACC_filtered_target)`
when the ACC target is fresh (≤ 0.5 s); otherwise `mission_speed` (smpc:865-876). Mission speed comes from MPC,
with a fallback of 15.0 (yaml:30). **[E]** This is the feedback path from the ACC into the SMPC prediction.

---

## 1. Ego trajectory generation [E]

### 1.1 Shared timing for a lane-change candidate

```
D    = spec.lane_change_duration_sec = egoChangeBlendDuration()   # BLEND: T_prev*1.04, else 4.0
t_sw = t_LC + 0.55                                                # CHANGE_NOW: t_LC = 0
tau_m = first grid tau_k >= t_sw - 1e-6  where rollout.v >= 0.20  or  (ds - ds_presw) >= 0.30
        # moving ego: tau_m = ceil_0.2(t_sw). CHANGE_NOW -> 0.6 s; t_LC = 0.4 -> 1.0; 0.8 -> 1.4 ...
alpha(tau) = clamp((tau - tau_m) / D, 0, 1)                       # LINEAR in time (preview path)
```
- Motion-start thresholds: smpc:2554-2564, yaml:238-239. The path-switch preview does not apply the
  smoothstep: `egoChangeAlpha` (smpc:1231) is only used by the fallback path, and `use_waypoint_linear_blend: true`
  makes it linear there as well.
- `T_prev = PathSwitchPreview.blend_duration_sec` comes from **[X] waypoint_system**. The baseline comments assume
  4.0 s (yaml:233-235, 289: "4.16 s measured blend"). The catkin_ws copy of waypoint_system computes it
  adaptively:
  `T(v) = clamp(sqrt(5.7735*3.5 / a(v)), 2.6, 5.0)`, where `a(v)` goes linearly from 1.3 at 8 m/s to 1.0 at
  28 m/s (`waypoint_system/config/base.yaml:81-95`, `local_path_publisher_node.cpp:372-386`).
  Examples: 13.7 m/s gives 4.08 s, 20 m/s gives 4.25 s, 25 m/s gives 4.40 s.
  **Unsure** which one is paired with dd1cf54. The two differ by ≤ 0.4 s. The recorded `/path_switch_preview`
  in the lc_gt bags would settle it.

### 1.2 Lateral profile in the SMPC prediction (straight parallel lanes)

Preview path (`buildEgoTrajectoryFromReferencePreview`, smpc:2470-2639). This path is **required** for
CHANGE_LEFT: `ego_reference_preview_required_for_change: true` (yaml:230). A missing or stale preview gives an
empty trajectory, and CHANGE becomes infeasible.

| mode | when [X] | d(tau) of the predicted ego (target-lane frame) | s(tau) |
|---|---|---|---|
| BLEND | normal (moving, or stopped but not at the endpoint) | `tau < t_sw`: source centreline (`d_tgt = -W`; the ego's actual offset is **ignored**). `tau ≥ t_sw`: `d_tgt = -(1-alpha)W` | `s0 + ds(tau)` |
| CONNECTOR | stopped (≤ 1.0 m/s) and the source end is within the waypoint trigger distance (catkin_ws: 20 m) | after `t_sw`: along the waypoint quintic connector polyline (spatial, join arc about 10 to 20 m [X], **unsure**) | `s0 + ds` |
| DIRECT | not used in practice | target centreline | |

- The blend is index-wise: `p_i = (1-a)*src_i + a*tgt_i` (smpc:1068-1107). On parallel straight lanes this is a
  straight line offset by `(1-a)W`. Hence predicted ego **yaw = lane heading throughout**, and the ego box is
  never rotated (smpc:1060-1064).
- Past the source polyline end, the blend appends unblended target points. Near a lane end the predicted ego
  therefore **jumps** to the target centre once `s > s_end_src` (smpc:1091-1093). The real reference has the
  same step (acc:1324-1331).
- Truncation (smpc:2585-2620): if the polyline runs out, the steps up to that point are kept. The candidate is
  accepted only if `traj.back().t ≥ t_sw + D + 1.5`. Terminal check (smpc:2622-2635): the last pose must have
  `|d_tgt| ≤ 0.75`, otherwise the trajectory is empty.
- CONNECTOR extra checks (smpc:2508-2531): reject if `pre_switch.ds > 0.50 m`, or if
  `t_sw + connectorTravelTime(v_sw, join_arc) + 1.5 > 8.0`.
- Fallback without preview (smpc:2641-2747, only reachable if the preview is not required): linear `alpha`
  from `t_sw` (no grid rounding), starting from the ego's actual `d0`: `d = (1-a)*d0_src + a*0` in s/d
  interpolation.

KEEP (always the non-preview path, smpc:2681-2696): `d(tau) = d0*exp(-tau/1e6) ≈ d0` (`lane_center_decay_tau_sec`
is 1e6, yaml:224), `s = s0 + ds`. **Returns empty if `s0+ds > lane_length`** (lane end). Then `keep_eval` is
invalid and the keep cost gets +500.

`candidateFitsHorizon` (smpc:3357): `t_sw + max(0.1,D) + 1.5 ≤ 8.0` (`behavior_post_merge_buffer_sec = 1.5`,
yaml:290). With D = 4.16 the allowed `t_LC` is ≤ 1.79, so the candidate set is {0.4, 0.8, 1.2, 1.6}. With
D = 4.58 (25 m/s adaptive) the allowed `t_LC` is ≤ 1.37, so the set is {0.4, 0.8, 1.2}.

### 1.3 Longitudinal rollout functions

Common: `v0 = ego speed`. `a_up = ego_prediction_max_accel_mps2 = 2.4`. `a_dn = ego_prediction_max_decel_mps2 = 9.0`
(yaml:381-382). `ego_prediction_use_mission_speed_rollout: true`.

| function | law | ref |
|---|---|---|
| `constantAccelDistance(v0,a,tau)` | stops at v=0 for a<0 | smpc:1152 |
| `egoLongitudinalRollout(tau, Vref)` | `= speedTargetRollout(v0, tau, Vref)` | smpc:1163 |
| `speedTargetRollout(v0,tau,Vref)` | `a = +2.4` if `Vref > v0+0.05`; `-9.0` if `Vref < v0-0.05`; else 0. Constant accel until `Vref`, then hold. Closed form | smpc:1180-1212 |
| `boundedPreparationRollout(tau, a, cap, floor)` | a = 0: `v = v0`, `ds = v0*tau` (cap **ignored**). a > 0: if `v0 ≥ cap`, then `v = cap` const; else accelerate to `cap` and hold. a < 0: `v = max(floor', v0 + a*tau)`, where `floor' = clamp(floor,0,v0)`; hold at floor'. **The persistent action is applied over the whole horizon**, not only until t_LC | smpc:1287-1338 |
| `behaviorYieldSpeedFloorMps` | YIELD only: `max(0, v_ref - 3.0)`. `v_ref` = speed at the start of the current YIELD episode if the last published action was YIELD, else `v0` (`behavior_yield_max_speed_drop_mps`) | smpc:1248, yaml:301 |
| `changePostCompletionSpeedMps(t, pre_cap)` | `post = mission`. If tf valid: `post = min(post, max(0, tf.v_long))` and return `post` (the pre_cap floor is off, yaml:400). If no tf: return `max(pre_cap, mission) = mission` | smpc:1345-1368 |
| `blendStageRollout` (**used for CHANGE_NOW**) | see 1.4 | smpc:1446-1482 |
| `blendCapRollout`, `blendInterpRollout` | **disabled** (yaml:405, 421) | smpc:1375, 1407 |
| `changeAwareRollout(t,tau,pre_cap,spec)` | if `spec == CHANGE_NOW` (and the target-lane-speed flag, yaml:392): `blendStageRollout` (stage flag yaml:410 takes precedence). Otherwise `actionLongitudinalRollout` | smpc:1484-1527 |
| `actionLongitudinalRollout(tau,cap,spec)` | KEEP / WAIT / CHANGE_NOW: `egoLongitudinalRollout(tau,cap)`. HOLD / YIELD / ACCEL: `boundedPreparationRollout(tau, a_prep, cap, yieldFloor)` | smpc:1529-1547 |
| `stoppedLaunchConnectorCapMps(lane)` | `[7.0, 9.0, 7.0, 7.0][target_lane]` (fallback 7.0) | smpc:1549, yaml:261-262 |
| `connectorLaunchRollout(tau, cap, spec, t_sw, lane)` | `tau ≤ t_sw`: `actionLongitudinalRollout(tau)` (note: CHANGE_NOW uses the **KEEP-type** speedTarget here). After that, start from `v_sw`: accelerate at `4.0` (`stopped_launch_connector_max_accel_mps2`) up to `cap_conn[lane]`, then hold | smpc:1558-1591, yaml:263 |
| `connectorTravelTimeSec(v0, dist, lane)` | time to travel `dist` under the same accelerate-to-cap law | smpc:1593 |

### 1.4 CHANGE_NOW longitudinal rollout (`blendStageRollout`)

```
d0     = |t.ego_d_target|            # measured, ~3.5 when centred
switch = t_sw + D*(1 - 2.7/d0)  if d0 > 2.7 else t_sw     # 2.7 = change_rollout_soft_acc_start_abs_d_m
         # d0=3.5, D=4.16: switch = 0.55 + 0.951 = 1.50 s   (alpha actually starts at 0.6, not 0.55)
v_post = changePostCompletionSpeedMps(t, pre_cap)
launch = v0 < 1.0                                         # change_rollout_launch_speed_threshold_mps
V1     = launch ? max(pre_cap, cap_conn[target]) : pre_cap
a_up   = launch ? 4.0 : 2.4 ;   a_dn = 4.0                # change_rollout_blend_cap_decel_mps2
v=v0, ds=0
for sigma in 0, 0.2, ... < tau:                           # ref chosen at the START of each 0.2 s step
    ref = V1 if sigma < switch else v_post
    vn  = min(ref, v + a_up*0.2) if ref > v+0.05 else (max(ref, v - a_dn*0.2) if ref < v-0.05 else v)
    ds += 0.1*(v + vn);  v = max(0, vn)
```
yaml:392-424. On a stopped ego this predicts a launch at `t = 0` (stage 1 applies from 0), not at `t_sw`.

### 1.5 Per-candidate summary (moving ego, BLEND preview)

Let `alpha(tau)` be as in 1.1. In the target-lane frame `d_tgt(tau) = -W*(1 - alpha(tau))` (and `-W` before tau_m).

| candidate | t_LC set | v(tau) | ds | published `speed_cap` |
|---|---|---|---|---|
| KEEP / WAIT_FOR_GAP | — | speedTarget: `v0 → pre_cap` at +2.4 / -9.0, then hold | closed form | `pre_cap` |
| CHANGE_NOW | 0 (t_sw 0.55, tau_m 0.6) | blendStage (1.4): `pre_cap` until about 1.5 s, then `v_post` at +2.4 / -4.0 | trapezoid sum | `pre_cap` (request inactive) |
| HOLD_THEN_CHANGE | {0.4,0.8,1.2,1.6,(2.0)} | `v0` constant | `v0*tau` | `v0` |
| YIELD_DECEL_THEN_CHANGE(a) | same × a ∈ {-0.25,-0.40,-0.60,-0.80} | `max(floor', v0 + a*tau)` over the whole horizon | closed form | `max(floor', v0 + a*t_sw)` |
| ACCEL_THEN_CHANGE(a_r) | same; added only if `rearAccelerationOpportunity` | `min(pre_cap, v0 + a_r*tau)` | closed form | `min(pre_cap, v0 + a_r*t_sw)` |
| any change, CONNECTOR | same | until t_sw: the row above (CHANGE_NOW uses the KEEP law); after: `v_sw → cap_conn` at 4.0 | | as above |

- `a_r = clamp(0.25*c_rear, 0.40, max(0.40, 1.0))`, where `c_rear = conservativeRearClosingSpeed(t.target_rear)`
  (smpc:3383, yaml:307-310).
- `rearAccelerationOpportunity` (smpc:3366): `behavior_execute_accel` and `t.target_rear` valid and
  `c_rear ≥ 1.0` and `rearBumperGap > 9.0` and `gap/max(0.1,c_rear) > 2.0`.
- Candidate list order (smpc:3394-3424): CHANGE_NOW, then for each t_LC ascending: HOLD, then YIELD
  (-0.25 → -0.80), then ACCEL.

---

## 2. Decision flow, `timerCallback` (smpc:4006-4441), 20 Hz

```
if !enabled or !have_ego or targets older than 0.5 s: return                       [P]
vehicles = unique{current_front, target_front, target_rear}     # risk_include_nearby_vehicles: false (yaml:19)
passing  = targetFrontIsPassingVehicle(t);  pid = passing ? tf.id : -1
R        = targetRearForGap(t, pid);  if R not in vehicles: vehicles += R            (4031-4041)
keep_traj   = buildEgoTrajectory(KEEP);   keep_eval   = evaluateTrajectory(keep_traj, vehicles, change=false)
change_traj = buildEgoTrajectory(CHANGE_NOW{t_LC=0, D});  change_eval = evaluateTrajectory(..., true)

cur_gap = cf.valid ? frontBumperGap(cf) : inf ;  tf_gap = frontBumperGap(tf)|inf ;  tr_gap = rearBumperGap(R)|inf
current_safe_gap = max(7.0, 5.0 + 1.0*v_e)                                    (4063; yaml:37-38,42)
side_prio  = egoLanePriorityEligible(cf)    # true only if |cf.d| > 1.25 and gap > 8: never for centred cars [P]
current_blocked = cf.valid and !side_prio and cur_gap < 35.0                  (4080; yaml:41)
endpoint_trigger = t.lane_change_prepare or t.lane_change_urgent or t.emergency_stop_required   (4083)
eff_rear_req = R ? targetRearRequiredGap(t, R, T_rem = egoChangeBlendDuration(), v_e) : 9.0     (4087)
```

### 2.1 CHANGE_NOW gates (`change_feasible`, smpc:4153) [E unless marked]

| # | gate | exact condition | ref |
|---|---|---|---|
| G1 | target is left | `t.target_lane_id == t.current_lane_id + 1` | 4106 |
| G2 | trajectory exists | `change_eval.valid` (preview fresh and BLEND/CONNECTOR checks of 1.2 pass) | |
| G3 | deterministic gap | front: `passing ? tf_gap ≥ 2.0` (`pass_gap_entry_front_clearance_m + 0 + 0*Δv`, yaml:131-144) `: (!tf.valid or tf_gap > 14.0)` ∧ rear: `!R or tr_gap > eff_rear_req` ∧ tf TTC: `passing or c_f=max(0,v_e - tf.v) ≤ 0.1 or max(0,tf_gap)/c_f ≥ 3.0` ∧ tf headway: `passing or max(0,tf_gap) ≥ 1.2*max(0, v_e - max(0,tf.v))` (closing-speed form, yaml:55) ∧ rear TTC: `c_r ≤ 0.1 or max(0,tr_gap)/c_r ≥ (v_e ≤ 6 ? 5.0 : 3.0)` ∧ rear headway: `max(0,tr_gap) ≥ 1.2*c_r`. Here `c_r = conservativeRearClosingSpeed(R)` | 4113-4126, 1806-1849 |
| G4 | chance | `timeIndexedRiskSafe(change_eval, 0.04) ∧ clearanceSafe(change_eval)` (legacy gate off, yaml:285) | 4133-4142 |
| G5 | projected gap/TTC | `projectedTargetGapTtcSafe(t, change_traj, CHANGE_NOW, R, passing)` (already ported) | 4149 |
| G6 | scene observed | `targetLaneSceneObservedForCommit(t) or targetLanePairObservedForPassGap(t, R)`; see 2.2 | 4127-4130 |
| G7 | cut-in interlock | `observedTargetLaneIntrusionSafe(t, pid)`; see 2.3 | 4131 |
| G8 | ACC stable | `accStableForChangeCommit()`; see 2.4 | 4152 |

The lane-end urgency gives **no relaxation** of any gate. These parameters exist but are dead code:
`urgent_merge_*`, `emergency_merge_*`, `prepare_merge_*`, `urgent_risk_epsilon`, `emergency_merge_risk_epsilon`,
and `effectiveTargetRearSafeGap()` (never called). `merge_priority_enabled: false`.

### 2.2 Scene observation (G6) [E with perfect perception too]

State `SceneHistory` (smpc:917-950), updated on **every targets message** from the real observations
(before any held-vehicle injection):
```
front_present = tf.valid and tf.id >= 0 ; front_id
rear_present  = tr.valid and tr.id >= 0 and |tr.delta_s| <= 60.0     # change_commit_scene_rear_max_range_m (yaml:103)
lane_pair_changed = !init or gap since last message > 0.35 s or (current,target) changed
changed = lane_pair_changed or front_present/front_id/rear_present/rear_id differ
if changed:           stable_since = now
if lane_pair_changed: lane_pair_stable_since = now
```
- `targetLaneSceneObservedForCommit` (2057): same lane pair ∧ `now - stable_since ≥ 1.5`
  (`change_commit_scene_observation_sec`) ∧ `vehObsStable(tf)` ∧ (tr beyond 60 m `or vehObsStable(tr)`).
- `vehObsStable(v)` (1979): invalid v counts as true. Otherwise `now - first_seen(v) ≥ 0.6`
  (`change_commit_vehicle_observation_sec`) and last seen ≤ 0.35 s ago. `first_seen` is kept per id in
  `vehicle_history` for every cf/tf/tr/nearby vehicle. It is reset if the id was unseen for > 0.35 s
  (smpc:878-907).
- Pass-gap variant `targetLanePairObservedForPassGap` (2033): `passing` ∧ same lane pair ∧
  `now - lane_pair_stable_since ≥ 1.5` (`pass_gap_lane_pair_observation_sec`) ∧ `vehObsStable(tf)` ∧
  `vehObsStable(R)`. So a vehicle that switches role from target_rear to target_front does not restart the clock.
- Consequence in the sim: **any change of the nearest target_front or target_rear identity** restarts a 1.5 s wait.
  This includes a vehicle entering the 60 m rear band, a vehicle passing the ego, and the ego passing a
  vehicle. Completing a lane change also restarts it, because the lane pair changes.

### 2.3 Cut-in interlock (G7), `observedTargetLaneIntrusionSafe` (smpc:2082-2256)

`observed_cutin_assume_no_lane_change: true` (yaml:183) disables the `|v_lat|` branch and the 4.5 s
entry-prediction branch. What remains, for every `v ∈ t.nearby_vehicles` with a role containing "nearby" and
`v.rear_delta_s ≤ 0` (behind or overlapping)? Note that nearby contains **all** confirmed vehicles, including
the representative ones. The passing vehicle is skipped (id == pid ∧ passing ∧ stable).
```
g  = max(0, -v.front_delta_s - 0.79) ;  c = max(0, v.v_long - v_e)        # NOT the conservative closing
fail if (A) v.lane_id == current_lane and v's footprint reaches the target-side boundary
            (sign_to_target*v.d + v.half_d >= 1.75) and g <= 60 and (g <= max(9, 1.2c) or g/c < 3.0)
fail if (B) |d of v w.r.t. target centre| <= 1.75 + v.half_d                # i.e. any target-lane vehicle
            and (g <= max(9.0, 1.2c) or (c > 0.1 and g/c < 3.0))           # NO distance limit
```
For lane-centred traffic, (A) never fires. (B) is an extra gate on **every** target-lane rear vehicle within
the nearby range (120 m), not only the nearest. **[E]**, cheap to implement.

### 2.4 ACC-stable gate (G8), `accStableForChangeCommit` (smpc:1932-1962)

```
stable_now = ACC status fresh (<=0.35 s)                                      [P]
             and !accHardBrakeOrLaneEndActive(reason)
             and (accPlannedLaneEndWaitDecel() or ego_accel_ema >= -99.0)     # decel term disabled (yaml:115)
accHardBrakeOrLaneEndActive(reason) = reason contains "hard" | "rapid" | "emergency"
                                      | ("lane_end" and not "lane_end_wait")  (1888-1917)
stable = stable_now held continuously >= 0.50 s                               # change_commit_acc_stable_sec
```
`reason` = the ACC `active_lead` string from the previous ACC tick (section 4.6). Blocking examples:
`current_front:<id>:hard`, `...:rapid`, `lane_end_emergency_stop|creep`, `lane_end_stop` (lane 3 only).
`lane_end_wait_*` (including `:stop`) does **not** block.

### 2.5 Desire and costs (smpc:4159-4179)

```
gapPenalty(g, g*) = isfinite(g) ? 2.5*max(0, g*-g)^2 / max(1, g*) : 0            (1751)
laneEndCost = !t.lane_change_required ? 0 : emergency ? 500 : urgent ? 180 : prepare ? 60 : 0   (1757)
keep_cost   = 80*keep_eval.risk + gapPenalty(side_prio ? inf : cur_gap, current_safe_gap) + laneEndCost
              + 500*[!keep_eval.valid] + 35*[current_blocked]
target_gap_cost = gapPenalty(tf_gap, 14.0) + gapPenalty(tr_gap, eff_rear_req)
change_cost = 0.5 + 80*change_eval.risk + target_gap_cost
              + 500*[!G1] + 500*[!G2] + 180*[!G3] + 180*[!G4] + 180*[!G5]
desire  = endpoint_trigger or current_blocked or (change_cost + 0.1 < keep_cost)   # decision_hysteresis_cost
request = desire and change_feasible
```
(`.risk` is the legacy mode-max risk from `evaluateTrajectory`.) On an open highway, the change is wanted only
when (a) there is a front car within a 35 m bumper gap, (b) the keep risk is high, or (c) lane-end pressure
applies (≤ 400 m / 8 s, section 5).

### 2.6 Future candidates and ranking (smpc:4197-4280)

Future candidates are evaluated only if `desire ∧ !change_feasible ∧ G1 ∧ G6 ∧ G7 ∧ G8 ∧ behavior_planner_enabled`
(4209). They are **not** gated by G3 (the instantaneous gap). For every candidate except CHANGE_NOW,
`evaluateCandidate` (3743-3781) runs:
```
infeasible unless candidateFitsHorizon
compat = preparationLongitudinalModelCompatible: (pre_cap + 0.10 >= v_e) or ACC_ACK    (3510-3537)
   ACC_ACK = the last published request was an active YIELD/HOLD/ACCEL for this lane pair, and the ACC status
             says behavior_cap_applied ∧ !external_speed_limit_active ∧ same lane pair ∧ applied cap
             ≈ issued cap (±0.1)   [plan_id/command_revision match = P]
feasible = timeIndexedRisk(<=0.04) ∧ clearanceSafe ∧ projectedTargetGapTtcSafe(t, traj, spec, R, passing)
cost = 0.5 + target_gap_cost(the CHANGE_NOW one) + 80*peak_step_risk + 1.0*t_LC + 0.25*a_prep^2
```
Best per tier = the minimum cost. Then lexicographic selection:
```
rear_press = R exists and conservativeRearClosingSpeed(R) >= 1.0
if   rear_press and have ACCEL: pick ACCEL                       ; clearYieldConfirm
elif have HOLD:                 pick HOLD                        ; clearYieldConfirm
elif !rear_press and have YIELD:
        if yieldCandidateConfirmed(t): pick YIELD                # same (cur,tgt,tf.id,tr.id) for >= 0.6 s (1262)
        else: yield_pending (-> WAIT)
elif have ACCEL:                pick ACCEL                       ; clearYieldConfirm
else:                           none                             ; clearYieldConfirm
```
(`tr.id` here is `t.target_rear`, not R.) `request_change` also clears the yield confirmation.

### 2.7 Outputs (smpc:4378-4441, `publishBehaviorOutputs` 3798-3915, `publishKeep/Change` 3917-3953)

| case | `/smpc/decision` | BehaviorLongitudinalRequest to ACC |
|---|---|---|
| `request` | CHANGE_LEFT, `request=true` | action NONE, `active=false` (the ACC drops any cap) |
| `desire ∧ future` | KEEP | YIELD / HOLD / ACCELERATE, `active=true`, `speed_cap` per 1.5, `requested_accel = a_prep`, `planned_lc_start = t_sw` |
| `desire ∧ !feasible ∧ no future` | KEEP (reason: the first failing gate) | WAIT_FOR_GAP, `active=true`, `speed_cap = pre_cap` (**not** used as a cap; it only arms the lane-end hold) |
| else | KEEP | NONE, `active=false` |

Request semantics in the ACC (section 4.3): it is an extra upper cap only and never raises speed. The YIELD
episode start speed is latched while YIELD is selected continuously for the same lane pair (3812-3821).

### 2.8 Hysteresis and timers [E]

- SMPC never latches CHANGE. The only hysteresis is the 0.1 cost margin. CHANGE_LEFT must be re-certified
  every 50 ms, and the supervisor needs 3 consecutive messages.
- Clocks: scene 1.5 s, vehicle age 0.6 s (gap timeout 0.35 s), ACC-stable 0.5 s, yield confirmation 0.6 s,
  yield floor at episode start minus 3 m/s, new-rear guard 0.8 s / 40 m / 15 m/s.
- The SMPC ignores `/smpc/lane_change_status`. It keeps evaluating the held (source, target) pair during an
  active change. The supervisor ignores those decisions, and the ACC ignores behavior requests while active.

### 2.9 Plumbing-only pieces [P] (one line each)

- `vanished_target_hold_*` (smpc:680-766) injects recently lost target-lane tracks: perception dropout only.
- Preview freshness/mode/lane-pair match (1109-1137): treat the preview as always fresh in the sim.
- Behavior-status freshness, `plan_id` / `command_revision` / previous-revision matching (3458-3508): reduce
  to the boolean `ACC_ACK`.
- `targets_timeout 0.5 s`, `mission_speed_timeout 0.5 s`, ACC target timeout 0.5 s.
- Lateral TV modes are off (`lateral_mode_enabled: false`, yaml:429). `ego_lane_priority` and
  `merge_priority` scaling have no effect when vehicles are lane-centred (`|d| ≤ 1.25`).
- `ego_longitudinal_accel` EMA (smpc:784-790) only feeds the disabled decel gate.

---

## 3. Lane-change supervisor (sup), timer 20 Hz (sup:1112, period 0.05)

### 3.1 Decision → `/path_number` (`decisionCallback`, sup:885-964) [E]

```
on each /smpc/decision (20 Hz):
  if active or !have_lane: return
  if inactive_sync_pending: consecutive = 0; return                                       [P]
  valid = msg.request and mode==CHANGE_LEFT and msg.current==current_lane and msg.target==current_lane+1
  if !valid: consecutive = 0; return
  if now - last_switch < 3.0: return            # cooldown_sec; neither increments nor resets
  if ++consecutive < 3: return                  # confirmation_count -> accept on the 3rd message (~+0.10 s)
  pose check |d_src| <= 4.0 and |d_tgt| <= 6.0 else consecutive = 0, return              (always true in sim)
  ACCEPT: source=current, target=current+1, active=true, change_start=now, last_switch=now, consecutive=0
          d_init = max(0.85, |ego_d_target|)    (fallback when targets are absent: 4.2 = lane_change_duration_sec, a unit bug, harmless)
          rate_f = 0, rear-abort confirmation reset; publish /path_number=target; status active
```

### 3.2 Activation and completion (sup:407-436, 966-996, 1112-1167)

- Completion clock start [P in sim]: after the `/gps_state` lane equals the target **and** `/final_waypoint`
  has ≥ 4 points on the target **and** motion progress (`v ≥ 0.20` or ≥ 0.30 m from the accept pose).
  For a moving ego this happens immediately.
- Watchdog [P]: if the path is not ready within 2.0 s (`path_switch_timeout_sec`), abort
  "target_path_switch_timeout". `motion_start_timeout_sec: 0` (disabled).
- Completion [E]: `elapsed = now - clock_start ≥ 2.0` (`completion_min_active_sec`) and the target-lane
  `|d| ≤ 0.75` (`completion_lateral_d_m`) for **5 consecutive** 20 Hz ticks
  (`completion_confirmation_count`; about 0.2 s). Then `finish`: active=false, `current_lane = target`,
  status inactive. The target_selector then switches to the lane pair (target, target+1).
- Timeout [E, rare]: at `elapsed ≥ 8.0` (`completion_timeout_sec`) the pose is classified on every tick
  (sup:324-378):
  - `|d_t| ≤ 0.75`: finish.
  - `|d_s| ≤ 0.90 ∧ |d_s| + 0.30 < |d_t|`: abort to the source.
  - `min(|d_s|,|d_t|) > 6.0`: abort to the nearest lane.
  - otherwise stay active. `completion_hard_timeout_sec: 0` (no forced completion).

### 3.3 Active rear abort (FULL), on each targets message (sup:822-883, 697-785) [E]

Inputs per message: `v_e = |odom twist|`, `D_t = |t.ego_d_target|`. The supervisor's own gap is
`g(v) = max(0, -v.front_delta_s - 0.79)` (clipped).

```
# always (also before the change starts): first_seen[id] for target-lane vehicles with rear_delta_s<=0
#   (target_rear + nearby). Reset if unseen > 0.35 s (rear_track_observation_gap_timeout_sec)  (799-820)
c(v) = max(0, v.v_long - v_e); if now-first_seen[v] < 0.8 and g(v) <= 40: c = max(c, 15.0)   (571-584)
while active, with lane pair == (source,target):
   rate_raw = max(0, (D_t_prev - D_t)/dt);  rate_f = 0.3*rate_raw + 0.7*rate_f              (830-845)

requiresAbort():
  if !enabled or !active or lane pair != (source,target): false
  if D_t <= 1.75: false                          # past the target-lane boundary: never abort  (abort_before_target_center)
  T = remaining():
        span = max(0.1, d_init - 0.75);  rl = max(0, D_t - 0.75)
        T = 4.2*clamp(rl/span, 0, 1)                                   # lane_change_duration_sec 4.2
        if v_e < 0.20:                              T = max(T, 4.5)    # stationary_remaining_sec
        elif clock_started and rate_f >= 0.05:      T = max(T, rl/max(0.05, rate_f))
        T = clamp(T, 0.30, 10.0)
  for v in {target_rear} ∪ nearby with v.lane_id == target and v.rear_delta_s <= 0 (rear OR overlap):
      intrudes = g <= 9.0 + c*(T + 0.5)  or  (c > 0.1 and g/c <= 3.0)
      if !intrudes: continue
      threat = true
      v_r = max(v.v_long, v_e + c)
      escapable   = g - (v_r*T - (v_e*T + 0.5*2.0*T^2)) > 9.0 + 0.5*c          # escape_accel 2.0, no speed cap
      launch_esc  = same, but ego distance uses accel 2.0 up to max(v_e, cap[target]=[7,9,7,7]) then constant
      emergency  |= g <= 4.0 or (c > 0.1 and g/c <= 1.0)
      need_abort |= !escapable ;  need_launch_abort |= !launch_esc
  if !threat: false
  front_ok = !tf.valid or max(0, tf.rear_delta_s - 3.845) >= max(10.0, 1.2*max(v_e, tf.v))
  if v_e < 0.20:                                                 # stationary
      if (now - change_start < 1.5) and !emergency and !need_launch_abort and front_ok: false
      return true, emergency_flag = emergency
  abort = !front_ok or need_abort ;  emergency_flag = emergency and abort ;  return abort

targetsCallback:
  if !requiresAbort(): reset(count=0, since=0); return
  if !emergency_flag:
      if since == 0: since = now; count = 1; return
      count++ ; if count < 3 or now - since < 0.30: return      # abort_confirmation_count / _sec
  abortLaneChange("rear_intrusion_before_target_center")
```
Notes:
- `remaining()` grows large early in a real (S-shaped) manoeuvre: a slow initial lateral rate means
  `rl/rate_f` is up to 10 s. That inflates the required gap `9 + c*(T+0.5)`. The plant lateral shape and the
  targets rate (LiDAR, about 8 to 10 Hz, **unsure**) both affect this.
- What the abort does (sup:1022-1063): `active=false`, `current_lane = source` (or the nearest lane for the
  off-lanes timeout), counters reset, publish `/path_number = source`, `last_switch = now` (starts the 3 s
  cooldown), inactive-sync pending [P].
  - There is no speed command. The ACC simply leaves lane-change mode (section 4.5).
  - Path back [X]: in the catkin_ws waypoint_system, a reverse pair has no preview. With spatial blend enabled,
    `/path_number` then switches **directly** to the source path (`local_path_publisher_node.cpp:1489-1495`), and
    the MPC steers back.
  - Measured (research/results/day1_m2.json, 24 rear aborts): median abort 0.52 s after start; 18/24 had not
    departed 0.45 m; the departed ones returned within 0.45 m in a median of 3.1 s (range 1.4 to 4.2 s);
    2 stopped between lanes.
- Retry: after the inactive sync and 3.0 s since the abort, 3 new consecutive CHANGE_LEFT messages are needed.
  The SMPC state is untouched by an abort. The lane pair stays (source, target), so the scene clock continues.
  **Unsure**: a one-frame `/gps_state` flicker to the target lane after an abort could reset it.

---

## 4. ACC speed planner (acc), 20 Hz → `/smpc/target_speed_mps`

The MPC [X] uses `min(ref_v, ACC)` as its speed reference. An ACC value of 0 forces a stop command
(`mpc_path_tracking.h:1638-1660`).

### 4.1 Lead law `speedForLead(L, cruise)` (acc:607-662) [E]

```
leadIsInFront(L) = L.valid and frontEdge(L) > -0.79 and L.delta_s > 0      # lead_requires_center_ahead (acc:499-517)
g  = rearEdge(L) - 3.845 ;  vL = clamp(L.v_long, 0, cruise) ;  c = max(0, v_e - vL)
if g <= 3.0: return 0                                                          # critical_gap_m
g* = 5.0 + 1.2*v_e + 1.6*c + c^2/(2*3.8)                                       # standstill, headway, closing_gap_time, braking_decel
cmd = vL + 1.05*(g - g*) ;  cmd = max(cmd, vL - 8.0)                           # gap_gain, max_follow_speed_drop
cmd = min(cmd, sqrt(vL^2 + 2*3.8*max(0, g - 5.0)))
if v_e - max(0,L.v) > 0.1:  ttc = g/(v_e - max(0,L.v))
     if ttc <= 2.2: return 0                                                   # critical_ttc
     cmd = min(cmd, vL + max(0, g - 5.0)/7.0)                                  # emergency_gap 5, soft_ttc 7
     if ttc <= 2.8: cmd = min(cmd, vL)                                         # hard_ttc
return clamp(cmd, 0, cruise)
hard(L)  = inFront and (g <= 3.0 or ttc <= 2.2)
rapid(L) = inFront and c >= 4.0 and (ttc <= 3.0 or c^2/(2(g-5)) >= 5.0)
```
Immature-track floor (`immature_lead_min_hits 5 → 2 m/s`) is [P].

### 4.2 Lead selection and target_front authority (acc:1145-1190, 725-766)

- `current_front` is always a candidate (it is the source lane during an active change).
- `target_front` is a candidate only if `(laneChangeIntent ∨ prep_active ∨ laneEndPressure) ∧ targetFrontAccAllowed`:
  - **hard**: `|ego_d_target| ≤ 0.75` (or `current == target`), with full authority.
  - **soft**: lane change active ∧ `0.75 < |d_t| ≤ 2.7`, so `allow_hard = false` and the decel is limited to
    1.5 m/s². This is skipped if `suppressTargetFrontSoftAccForRearIntrusion` holds: an ACC-side rear intrusion
    (same form as 3.3 but non-conservative `c` and ACC remaining with `d_init` span vs 0.75, stationary 8.0 s,
    acc.yaml:207) ∧ `|d_t| > 1.75` ∧ tf not hard/rapid.
  - Before the change starts, `|d_t| ≈ 3.5`, so the target_front **never** limits the ACC.
- nearby (`nearby_acc_overlap_only`): only `nearby_overlap` vehicles with `|v.d - ego_d_lane| ≤ 0.946 + half_d + 0.2`
  that also pass `leadIsInFront`.
- `raw = min(cruise, speedForLead over the candidates)`. `active_lead` is the label of the last lead that
  lowered `raw` (+":hard" / ":rapid" / ":overlap_soft").

### 4.3 SMPC behavior request application (acc:1263-1300, 366-408) [E]

```
prep_active = enabled and request fresh(<=0.35 s) and request.active and lane pair matches targets
              and !(lane_change_active or CHANGE_LEFT received <= 0.5 s ago for this pair)   # apply_while_lane_change: false
              and action in {YIELD, HOLD, ACCELERATE(apply_accel true), WAIT_FOR_GAP}
external = active_lead not in {"none","merge_keep_rolling"}     # computed BEFORE the cap -> fed back as the ACK
if prep_active and action != WAIT:
   cap = clamp(speed_cap, 0, cruise)
   if cap < raw: raw = cap; cap_applied = true
                 if !hard and !external and YIELD and a<0: yield_slew = |a|
WAIT_FOR_GAP: no cap; only arms the lane-end wait (4.4); latched for 1.0 s after CHANGE_LEFT until active.
```
A CHANGE_LEFT decision clears the stored request on arrival (acc:293-306).

- HOLD: the target is pinned at the ego speed when the request was made.
- ACCELERATE: the cap is `v0 + a*t_sw`. It is really an acceleration *limiter* relative to plain ACC.
- YIELD: the target is held `|a|*t_sw` below the current speed and descends at ≤ |a|.

### 4.4 Lane end (lanes 0 to 2 must merge; lane 3 stops) (acc:994-1046, 1193-1245)

```
protection_released = lane_change_active                        # lane_end_hold_until_lane_change_active: true
mergeWaitStop = !released and lane_change_required and lane_change_urgent
                and (targetFrontBlocksMerge  [tf in front and frontGap(tf) <= 11.0]  or  WAIT_FOR_GAP active)
if lane_change_urgent and mergeWaitStop:
   raw = min(raw, dist <= 25 ? 0 : sqrt(2*2.5*(dist - 25)))     # holding line 25 m, comfort decel 2.5
   (raw <= 0.01 and !mergeKeepRolling -> hard_brake_active: instant)
   label lane_end_wait_front_gap | lane_end_wait_no_safe_gap
if emergency_stop_required and mergeWaitStop: same limit, label lane_end_emergency_* or lane_end_wait_no_safe_gap:stop|creep
lane 3 only: if dist <= 45 or ttle <= 4: raw = min(raw, sqrt(2*2.5*max(0, dist - 2)))  "lane_end_stop"
mergeKeepRolling (acc:1094): (intent or laneEndPressure) and !lane3stop and !mergeWaitStop and no lead within 8 m
   and no lead <= 11.1 m/s within 80 m  ->  raw = max(raw, min(11.1, cruise))   (only if !hard)
during active LC near the end: raw = min(raw, clamp((dist - 3)/max(0.5, 4.2 - elapsed), min(3, cruise), cruise)),
   released when |d_t| <= 1.5                                    (acc:1336-1358)
endpoint connector [X] active: raw = min(raw, [7,9,7,7][active path]), accel limit 4.0      (acc:1305-1316)
```
`dist = distance_to_lane_end` from target_selector (rear axle).

### 4.5 During an active lane change

- `current_front` (source) is still a lead.
- `target_front` gets soft authority (≤ 1.5 m/s²) once `|d_t| ≤ 2.7` and hard authority at ≤ 0.75.
- SMPC caps are ignored, and the lane-end wait/stop is released (replaced by the LC lane-end cap).
- The merge-keep-rolling floor is 11.1 m/s.
- The rear-escape accel (`active_rear_escape_max_accel_mps2 4.0`) equals `max_accel 4.0`, so it is a **no-op**.
- After an abort: normal ACC (target_front authority disappears because `|d_t|` grows again and
  `lane_change_active` is false).

### 4.6 Rate limiting, the published value, and the status (acc:1360-1445) [E]

```
dec = hard ? max(7.0, 12.0) : 7.0 + clamp((c_max - 1.0)/5.0, 0, 1)*(11.0 - 7.0)   # closing boost from binding leads
if yield_slew:                    dec = min(dec, |a_yield|)
if soft target_front binding:     dec = min(dec, 1.5)
if lead binds raw and !hard:      dec = min(dec, max(1.5, 1.5*c^2/(2*(g - 5 - 1.2*v_e))))   # only if margin > 0.5
if hard (incl. rapid):            f = raw                                            # bypass filter
else:                             f = clamp(raw, f - dec*dt, f + 4.0*dt)            # max_accel 4.0; connector 4.0
f = clamp(f, 0, cruise);  publish f;  status{reason=active_lead, cap_applied, external, applied_cap, final=f}
```
`f` is internal ACC state. It is **not** reset to the ego speed.

### 4.7 Achieved vehicle longitudinal response [X, documented only by comments]

| quantity | value | source |
|---|---|---|
| MPC pedal PI | kp 0.22, ki 0.02 per m/s; ramp feed-forward `pedal = ramp/3.5` | `mpc_path_tracking/config/mpc_params.yaml:145-169` |
| throttle authority | pedal 0.6 ≈ 1.3 to 1.5 m/s²; pedal 1.0 ≈ 3.3 to 4.9 m/s² (limit 1.0) | mpc_params.yaml:150-152 |
| normal brake | brake ≤ 0.50; brake 0.6 ≈ 6.8 m/s² measured; brake rate ≤ 2.0 /s | mpc_params.yaml:156-160, acc_planner.yaml:57 |
| measured ego decel | p50 2.69, p90 9.28 m/s² | smpc_decision.yaml:159 |
| launch from rest | reaches 0.2 m/s after 0.30 to 0.65 s (M1: median 0.49 s) | supervisor.yaml:65, day1_m1.json |
| legal guard | throttle cut above 16.2 m/s, brake above 16.4 outside exempt links | mpc_params.yaml:173-177 |

---

## 5. Target selector (sel) [E rules, P filters]

- Lane ids (sel:319-334): `current_lane = /gps_state lane` (the active waypoint path) unless a change is active;
  while active it is **forced** to (source, target) from the supervisor status.
  - `target_lane = forced ≥ 0 ? forced : (current < 3 ? current + 1 : current)` (sel:840-843,
    `final_lane_id 3`, `target_lane_offset 1`).
  - The pair changes only at completion (to (tgt, tgt+1)) and at abort (it stays (src, tgt)).
- Lane membership (sel:910-927): `|d_L| - half_d ≤ 2.2 + 0.2`. Representative also requires `|d_L| ≤ 1.35`
  and not a flat long track [P]. If a vehicle is on both the current and the target lane, the one with the
  smaller clearance wins.
- Roles (sel:928-963). `ds` is relative to the ego rear-axle s. All use strict `<` against a running minimum,
  initialised to 120 m (`max_front/rear_range`).

| role | candidate if | key minimised |
|---|---|---|
| current_front | representative current-lane vehicle, `front_edge_ds > 0` | `max(0, rear_edge_ds)` < 120 |
| target_front | representative target-lane vehicle, `front_edge_ds ≥ 0` | `max(0, rear_edge_ds)` < 120 |
| target_rear | representative target-lane vehicle, `rear_edge_ds ≤ 0` | `max(0, -front_edge_ds)` < 120 |
| nearby_* | **every** confirmed vehicle within `|d|-half_d ≤ 3.4` of its best lane, with `front_delta_s ≥ -120` and `rear_delta_s ≤ 150` | role: front (rear edge > 0) / rear (front edge < 0) / overlap; sorted by `|delta_s|`, max 40 (sel:777-826) |

  A vehicle that overlaps the ego longitudinally is **both** target_front and target_rear (gap 0). Ties at
  gap 0 go to the first in tracker order.
- Lane end (sel:976-993):
  - `distance_to_lane_end = max(0, L_current - ego_s_current)`.
  - `ttle = dist / v_e` if `v_e > 0.1`, else inf.
  - `lane_change_required = current < 3`.
  - `prepare = req ∧ (dist ≤ 400 ∨ ttle ≤ 8.0)`.
  - `urgent = req ∧ (dist ≤ 120 ∨ ttle ≤ 5.0)`.
  - `emergency_stop_required = req ∧ (dist ≤ 15 ∨ ttle ≤ 1.0)` (target_selector.yaml:111-117).
- Fields for each vehicle (sel:740-775): `delta_s = s - ego_s`, `half_s/half_d` (footprint on the lane),
  `front/rear_delta_s = delta_s ± half_s`, `v_long/v_lat` projected on the lane tangent, `track_hits`.
- [P]: confidence filters and holds, sticky size hysteresis, yaw stabilisation, `min_tracked_hits 2`, size
  filter, projection extension before the CSV start.

---

## 6. Minimal faithful Python model (baseline arm), about 500 to 900 lines plus the ported risk code

**Update order per 0.05 s tick** (synchronous; each consumer reads the producer's output from the previous
stage of the same tick): perception, then SMPC, then supervisor (decision + rear guard), then ACC, then plant.
Optionally run perception at 10 Hz to mimic the tracker. This changes the rear-abort count and `rate_f`.

### Must implement

1. **Perception builder** (about 80 lines): build `TargetVehicleSet` per section 5.
   - current_front / target_front / target_rear / nearby with exact edge semantics.
   - Lane pair forced during an active change.
   - Lane-end flags, `ego_d_current`, `ego_d_target` from the ego rear axle.
   - Per-id `first_seen` / `last_seen`.
2. **SMPC state** (about 60 lines):
   - `SceneHistory` (2.2), `vehicle_history.first_seen`, ACC-stable timer (2.4).
   - Yield confirmation (2.6), the last published action, and the yield episode start speed.
3. **Ego rollouts and trajectories** (about 120 lines): 1.1 to 1.5.
   - speedTarget, boundedPreparation, blendStage, connectorLaunch (optional).
   - BLEND lateral `d_tgt = -W(1-alpha)` with grid-rounded `tau_m` and `D = 1.04*T_prev`. Yaw = 0. Lane-end
     step and truncation coverage. KEEP invalid past the lane end. `candidateFitsHorizon`.
4. **Ported risk code** (already done): `evaluateTrajectory`, `projectedTargetGapTtcSafe`, `timeIndexedRiskSafe`,
   `clearanceSafe`, modes, `evaluateCandidate`. Feed it with `vehicles = {cf, tf, tr} ∪ {R}`.
5. **Decision** (about 150 lines): 2.1 gates G1 to G8 (including the pass-gap front/scene exceptions and the
   cut-in branch (B)), 2.5 costs and desire, 2.6 tiers plus `ACC_ACK` compatibility, and 2.7 outputs (decision
   bool and request {action, cap, a}).
6. **Supervisor** (about 120 lines): section 3. Covers 3-consecutive acceptance, the 3 s cooldown, completion
   (≥ 2 s, `|d| ≤ 0.75` × 5), the 8 s timeout classification, the full rear abort (remaining, intrudes,
   escapable, launch grace, front_ok, emergency, 3 counts / 0.3 s), and the abort effects.
7. **ACC** (about 130 lines): 4.1 to 4.6. Covers the lead law, lead eligibility (including target_front
   soft/hard by `|d_t|` and the nearby overlap), request application and status/ack, lane-end wait/stop/merge
   rolling/LC cap, and the filter with all decel limits and the hard bypass. Expose `reason` for G8.
8. **Plant** (about 60 lines):
   - Longitudinal: track the ACC target with a PI-like first-order response, bounded at about +3.5 / -5.7 m/s²
     normally and down to about -9 m/s² for hard events. ACC 0 means a full stop. Calibrate to 4.7.
   - Lateral on a path switch: **not** the SMPC's linear model. Use a spatial quintic
     `w(u) = 10u³-15u⁴+6u⁵` over `L = max(20, v*T(v))` plus about 0.3 to 0.5 s of tracking lag.
     Calibrate to M1 (moving: 10/50/90 % progress at 1.22/2.71/4.42 s; `|d| ≤ 0.45` at 4.34 s).
   - Abort: direct switch back, first-order lateral return (M2: about 3 s).

### Safe simplifications (reason)

| simplification | reason |
|---|---|
| Analytic Frenet (s, d) instead of CSV polylines and projections; preview blend = linear lateral offset; ego yaw = lane heading | straight parallel lanes make index-wise blending and projection exact |
| Preview always fresh; BLEND when moving, CONNECTOR only if a lane-end standstill is modelled | freshness and mode handshakes are plumbing [P] |
| Drop held/vanished vehicles, confidence holds, sticky roles, yaw filters, size/flat/min-hit filters, immature-lead floor | perfect perception: no dropouts or ghosts |
| Keep the new-track 15 m/s guard only if vehicles can spawn within 40 m of the ego | otherwise `first_seen` is far in the past |
| Replace `plan_id`, `command_revision` and the previous-revision match with a boolean `ACC_ACK` | synchronous tick order removes the asynchrony it handles |
| Drop the inactive-lane sync, `/gps_state` and `/final_waypoint` acks, path-switch watchdog, `lane_sync_mismatch`, freshness timeouts | instantaneous in the sim |
| Drop lateral TV modes, the cut-in `v_lat` and entry-prediction branches, and branch (A) | disabled by config or never triggered by lane-centred TVs |
| Drop `ego_lane_priority` and `merge_priority` | no effect for `|d| ≤ 1.25` / disabled |
| Drop the dead relaxation params and the decel commit gate (99) | no runtime effect |
| Single `D` per tick from `T(v_e)` or a fixed 4.0 s | the SMPC reads one preview value per tick |

### Open uncertainties to resolve or parameterise

1. `T_prev`: fixed 4.0 s vs the adaptive `T(v)` [X].
2. CONNECTOR trigger and join geometry. The catkin_ws waypoint_system uses 20 m / 10 m, which contradicts the
   baseline comments (20 m connector, 25 m holding line).
3. The targets rate (8 to 10 Hz).
4. Whether a post-abort lane-id flicker resets the scene clock.
5. Plant brake and throttle maps (only comments document them).
