std_msgs/Header header

# ACC acknowledgement for a behavior preparation request.  It lets SMPC
# distinguish its own speed cap from an independent lead/lane-end limitation
# on the next receding-horizon cycle.
uint32 plan_id
# Revision of the BehaviorLongitudinalRequest that this status describes.
uint32 command_revision
bool behavior_cap_applied
bool external_speed_limit_active
int32 current_lane_id
int32 target_lane_id
float64 applied_speed_cap_mps
float64 final_target_speed_mps
string reason
