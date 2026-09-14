#include <algorithm>
#include <ros/ros.h>
#include <smpc_lane_change/LaneChangeDecision.h>
#include <smpc_lane_change/TargetVehicle.h>
#include <smpc_lane_change/TargetVehicleSet.h>

// Integration-test fallback only. This is NOT SMPC.
class GapTestDecisionNode {
 public:
  GapTestDecisionNode() : nh_(), pnh_("~") {
    pnh_.param("enabled", enabled_, false);
    pnh_.param("min_target_front_gap", min_front_gap_, 20.0);
    pnh_.param("min_target_rear_gap", min_rear_gap_, 15.0);
    pnh_.param("trigger_current_front_gap", trigger_current_gap_, 35.0);
    pnh_.param("target_speed_mps", target_speed_mps_, 10.0);
    pnh_.param("ego_length_m", ego_length_m_, 4.635);
    pnh_.param("ego_front_extent_m", ego_front_extent_m_, 3.845);
    pnh_.param("ego_rear_extent_m", ego_rear_extent_m_, 0.790);
    sub_ = nh_.subscribe("/smpc/targets", 1, &GapTestDecisionNode::callback, this);
    pub_ = nh_.advertise<smpc_lane_change::LaneChangeDecision>("/smpc/decision", 1);
    ROS_WARN_COND(enabled_, "[gap_test_decision] ENABLED: this is wiring test logic, not SMPC");
  }

 private:
  double leadHalfS(const smpc_lane_change::TargetVehicle& v) const {
    return v.half_s > 1e-3 ? v.half_s : 0.5 * v.length;
  }

  double frontEdgeDeltaS(const smpc_lane_change::TargetVehicle& v) const {
    return v.front_delta_s != 0.0 ? v.front_delta_s : v.delta_s + leadHalfS(v);
  }

  double rearEdgeDeltaS(const smpc_lane_change::TargetVehicle& v) const {
    return v.rear_delta_s != 0.0 ? v.rear_delta_s : v.delta_s - leadHalfS(v);
  }

  double frontBumperGap(const smpc_lane_change::TargetVehicle& front) const {
    return rearEdgeDeltaS(front) - ego_front_extent_m_;
  }

  double rearBumperGap(const smpc_lane_change::TargetVehicle& rear) const {
    return -frontEdgeDeltaS(rear) - ego_rear_extent_m_;
  }

  void callback(const smpc_lane_change::TargetVehicleSet::ConstPtr& t) {
    if (!enabled_) return;
    const bool current_blocked = t->current_front.valid &&
        frontBumperGap(t->current_front) < trigger_current_gap_;
    const bool front_safe = !t->target_front.valid ||
        frontBumperGap(t->target_front) > min_front_gap_;
    const bool rear_safe = !t->target_rear.valid ||
        rearBumperGap(t->target_rear) > min_rear_gap_;
    const bool gap_safe = front_safe && rear_safe;
    const bool endpoint_trigger = t->lane_change_prepare || t->lane_change_urgent;

    smpc_lane_change::LaneChangeDecision d;
    d.header = t->header;
    d.current_lane_id = t->current_lane_id;
    d.target_lane_id = t->target_lane_id;
    d.target_speed_mps = target_speed_mps_;
    d.request = (current_blocked || endpoint_trigger) && gap_safe;
    d.mode = d.request ? smpc_lane_change::LaneChangeDecision::CHANGE_LEFT
                       : smpc_lane_change::LaneChangeDecision::KEEP;
    if (d.request && endpoint_trigger) {
      d.reason = "lane_end_deadline_gap_safe";
    } else if (d.request) {
      d.reason = "current_front_blocked_gap_safe";
    } else if (endpoint_trigger && !gap_safe) {
      d.reason = t->emergency_stop_required
          ? "lane_end_emergency_gap_unsafe"
          : "lane_end_wait_gap";
    } else {
      d.reason = "gap_test_keep";
    }
    pub_.publish(d);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  bool enabled_{false};
  double min_front_gap_{20.0}, min_rear_gap_{15.0}, trigger_current_gap_{35.0};
  double target_speed_mps_{10.0};
  double ego_length_m_{4.635};
  double ego_front_extent_m_{3.845};
  double ego_rear_extent_m_{0.790};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "gap_test_decision");
  GapTestDecisionNode node;
  ros::spin();
  return 0;
}
