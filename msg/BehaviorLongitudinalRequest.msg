std_msgs/Header header

# Narrow execution contract from the behavior planner to ACC.  ACC owns the
# final speed command and treats speed_cap_mps only as an additional cap.
uint8 NONE=0
uint8 YIELD=1
uint8 HOLD=2
uint8 ACCELERATE=3
# Preserve the current longitudinal policy while the planner waits for a gap.
# ACC must not interpret this as a speed cap or a lane-change rejection.
uint8 WAIT_FOR_GAP=4

uint8 action
uint32 plan_id
# Material command revision within plan_id.  ACC echoes this in its status so
# the planner can identify which prior cap produced the current ACC baseline.
uint32 command_revision
bool active
int32 current_lane_id
int32 target_lane_id
float64 speed_cap_mps
float64 requested_accel_mps2
float64 planned_lc_start_sec
float64 candidate_cost
float64 peak_risk
string reason
