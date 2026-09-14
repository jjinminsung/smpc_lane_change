bool valid
int32 unique_id
string role
int32 lane_id
float64 x
float64 y
float64 s
float64 d
float64 delta_s
float64 yaw
float64 yaw_rate
float64 v_long
float64 v_lat

# Raw bbox size. length is along yaw, width is perpendicular to yaw.
float64 length
float64 width

# Bbox footprint projected onto this vehicle's assigned lane Frenet frame.
# half_s: half extent along the lane direction.
# half_d: half extent across the lane direction.
# front_delta_s/rear_delta_s: object front/rear edges relative to ego s on that lane.
float64 half_s
float64 half_d
float64 front_delta_s
float64 rear_delta_s

# Tracker maturity: number of frames this track was actually matched to a
# detection.  A ghost that survives on Kalman prediction stays low while a real
# vehicle accumulates quickly (measured over 15 bags: representative frames,
# real median 24-35, ghost median 4-9).  Consumers use it to withhold the most
# aggressive responses from immature tracks, never to discard the obstacle.
int32 track_hits
