std_msgs/Header header
int32 current_lane_id
int32 target_lane_id

# Ego Frenet state on the current lane CSV.
float64 ego_s_current
float64 ego_d_current
float64 ego_d_target
float64 current_lane_length
float64 distance_to_lane_end
float64 time_to_lane_end

# Lane-end deadline state. final_lane_id (normally 3) has no mandatory left change.
bool lane_change_required
bool lane_change_prepare
bool lane_change_urgent
bool emergency_stop_required

smpc_lane_change/TargetVehicle current_front
smpc_lane_change/TargetVehicle target_front
smpc_lane_change/TargetVehicle target_rear

# All lane-associated candidate vehicles near ego. This intentionally includes
# vehicles beyond the three representative gap vehicles so merging traffic can
# affect ACC and SMPC collision-risk evaluation.
smpc_lane_change/TargetVehicle[] nearby_vehicles
