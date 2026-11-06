std_msgs/Header header

# High-level longitudinal/lane-change preparation action.  This message is a
# receding-horizon recommendation; it is not a pre-authorized future lane
# change.  The lane-change supervisor must still receive a fresh
# LaneChangeDecision CHANGE_LEFT request at the actual start time.
uint8 KEEP=0
uint8 CHANGE_NOW=1
uint8 YIELD_DECEL_THEN_CHANGE=2
uint8 HOLD_THEN_CHANGE=3
uint8 ACCEL_THEN_CHANGE=4
# Lane change is desired but no safe start time exists in the current horizon.
# This is a wait/observe state, not a future lane-change reservation.
uint8 WAIT_FOR_GAP=5

uint8 action
uint32 plan_id
# Monotonic revision of the material longitudinal command within this plan.
# A stable plan_id denotes one preparation episode; this changes when the
# published action/lane/timing/speed-cap contract changes materially.
uint32 command_revision
bool now_feasible
bool future_feasible
bool longitudinal_command_enabled

int32 current_lane_id
int32 target_lane_id

# Requested preparation command.  ACC remains the sole publisher of the
# final speed command and clamps this request using its own lead/TTC limits.
float64 preparation_accel_mps2
float64 target_speed_mps

# Predicted timing for the selected candidate, measured from this message's
# header stamp.  These are re-evaluated every planning cycle.
float64 lane_change_start_sec
float64 lane_change_duration_sec
float64 horizon_sec

float64 selected_cost
float64 peak_risk
float64 legacy_risk
# -1.0 means no per-step risk-threshold violation in the horizon.
float64 first_risk_violation_time_sec
float64 min_signed_clearance_m
float64[] risk_by_step
string reason
