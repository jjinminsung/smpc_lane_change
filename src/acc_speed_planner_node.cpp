#include <algorithm>
#include <cstdint>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <smpc_lane_change/BehaviorLongitudinalRequest.h>
#include <smpc_lane_change/BehaviorLongitudinalStatus.h>
#include <smpc_lane_change/LaneChangeDecision.h>
#include <smpc_lane_change/LaneChangeStatus.h>
#include <smpc_lane_change/TargetVehicle.h>
#include <smpc_lane_change/TargetVehicleSet.h>
#include <waypoint_maker/State.h>

class AccSpeedPlanner {
 public:
  AccSpeedPlanner() : nh_(), pnh_("~") {
    pnh_.param("ego_length_m", ego_length_m_, 4.635);
    pnh_.param("ego_width_m", ego_width_m_, 1.892);
    pnh_.param("ego_front_extent_m", ego_front_extent_m_, 3.845);
    pnh_.param("ego_rear_extent_m", ego_rear_extent_m_, 0.790);
    ego_length_m_ = ego_front_extent_m_ + ego_rear_extent_m_;
    pnh_.param("standstill_gap_m", standstill_gap_m_, 8.0);
    pnh_.param("time_headway_sec", time_headway_sec_, 2.2);
    pnh_.param("gap_gain", gap_gain_, 0.65);
    pnh_.param("closing_speed_gap_time_sec", closing_speed_gap_time_sec_, 0.8);
    pnh_.param("braking_decel_mps2", braking_decel_mps2_, 4.5);
    pnh_.param("soft_ttc_sec", soft_ttc_sec_, 4.0);
    pnh_.param("hard_ttc_sec", hard_ttc_sec_, 1.8);
    pnh_.param("min_speed_mps", min_speed_mps_, 0.0);
    pnh_.param("max_accel_mps2", max_accel_mps2_, 1.5);
    pnh_.param("max_decel_mps2", max_decel_mps2_, 5.0);
    pnh_.param("closing_decel_boost_enabled",
               closing_decel_boost_enabled_, true);
    pnh_.param("closing_decel_boost_start_mps",
               closing_decel_boost_start_mps_, 5.0);
    pnh_.param("closing_decel_boost_full_mps",
               closing_decel_boost_full_mps_, 15.0);
    pnh_.param("closing_decel_boost_max_decel_mps2",
               closing_decel_boost_max_decel_mps2_, 8.0);
    pnh_.param("hard_brake_decel_mps2", hard_brake_decel_mps2_, 9.0);
    pnh_.param("hard_brake_bypass_filter", hard_brake_bypass_filter_, true);
    pnh_.param("emergency_gap_m", emergency_gap_m_, 8.0);
    pnh_.param("critical_gap_m", critical_gap_m_, 2.5);
    pnh_.param("critical_ttc_sec", critical_ttc_sec_, 0.7);
    pnh_.param("rapid_lead_brake_enabled", rapid_lead_brake_enabled_, true);
    pnh_.param("rapid_lead_ttc_sec", rapid_lead_ttc_sec_, 3.0);
    pnh_.param("rapid_lead_min_closing_speed_mps",
               rapid_lead_min_closing_speed_mps_, 4.0);
    pnh_.param("rapid_lead_required_decel_mps2",
               rapid_lead_required_decel_mps2_, 5.0);
    pnh_.param("max_follow_speed_drop_mps", max_follow_speed_drop_mps_, 5.0);
    // Ordinary (non hard/rapid) lead following: bound how fast the target may
    // fall by what the binding lead actually requires.
    pnh_.param("lead_required_decel_limit_enabled",
               lead_required_decel_limit_enabled_, false);
    pnh_.param("lead_required_decel_limit_gain",
               lead_required_decel_limit_gain_, 1.5);
    pnh_.param("lead_required_decel_limit_min_mps2",
               lead_required_decel_limit_min_mps2_, 1.5);
    pnh_.param("lead_required_decel_limit_headway_sec",
               lead_required_decel_limit_headway_sec_, 1.5);
    pnh_.param("target_timeout_sec", target_timeout_sec_, 0.8);
    pnh_.param("decision_request_timeout_sec", decision_request_timeout_sec_, 0.5);
    pnh_.param("publish_rate_hz", publish_rate_hz_, 20.0);
    pnh_.param<std::string>("mission_speed_topic",
                            mission_speed_topic_,
                            "/control_feedback/mission_speed_target_mps");
    pnh_.param("mission_speed_timeout_sec", mission_speed_timeout_sec_, 0.5);
    pnh_.param("mission_speed_fallback_mps", mission_speed_fallback_mps_, 10.0);
    pnh_.param("behavior_request_enabled", behavior_request_enabled_, false);
    pnh_.param<std::string>("behavior_request_topic", behavior_request_topic_,
                            "/smpc/behavior_longitudinal_request");
    pnh_.param("behavior_request_timeout_sec", behavior_request_timeout_sec_, 0.35);
    pnh_.param<std::string>("behavior_status_topic", behavior_status_topic_,
                            "/smpc/behavior_longitudinal_status");
    pnh_.param("behavior_request_apply_while_lane_change",
               behavior_request_apply_while_lane_change_, false);
    pnh_.param("behavior_request_apply_accel", behavior_request_apply_accel_, false);
    // A path-end connector is only intended for the short, low-speed handoff
    // from an exhausted source CSV to the already-loaded target-lane CSV.
    // Its Bool topic has no header, so freshness is measured at receipt time.
    pnh_.param("endpoint_connector_speed_cap_mps",
               endpoint_connector_speed_cap_mps_, 2.0);
    // Optional per-target-lane override, indexed by target lane id.  Must be
    // identical to SMPC's stopped_launch_connector_speed_cap_by_target_lane_mps.
    pnh_.getParam("endpoint_connector_speed_cap_by_target_lane_mps",
                  endpoint_connector_speed_cap_by_target_lane_mps_);
    pnh_.param("endpoint_connector_max_accel_mps2",
               endpoint_connector_max_accel_mps2_, max_accel_mps2_);
    pnh_.param("endpoint_connector_status_timeout_sec",
               endpoint_connector_status_timeout_sec_, 0.5);
    pnh_.param("lane_change_lane_end_cap_enabled",
               lane_change_lane_end_cap_enabled_, false);
    pnh_.param("lane_change_lane_end_cap_release_abs_d_m",
               lane_change_lane_end_cap_release_abs_d_m_, 0.0);
    pnh_.param("lane_end_hold_until_lane_change_active",
               lane_end_hold_until_lane_change_active_, false);
    pnh_.param("lane_end_hold_wait_for_gap_latch_sec",
               lane_end_hold_wait_for_gap_latch_sec_, 1.0);
    pnh_.param("lane_change_lane_end_margin_m", lane_change_lane_end_margin_m_, 3.0);
    pnh_.param("lane_change_lane_end_min_cap_mps",
               lane_change_lane_end_min_cap_mps_, 3.0);
    pnh_.param("min_target_front_gap_m", min_target_front_gap_m_, 11.0);
    pnh_.param("min_target_rear_gap_m", min_target_rear_gap_m_, 6.0);
    pnh_.param("target_front_acc_requires_target_lane_entry",
               target_front_acc_requires_target_lane_entry_, true);
    pnh_.param("target_front_acc_entry_abs_d_m",
               target_front_acc_entry_abs_d_m_, 0.75);
    pnh_.param("target_front_preentry_soft_acc_enabled",
               target_front_preentry_soft_acc_enabled_, false);
    pnh_.param("target_front_preentry_soft_acc_distance_to_end_m",
               target_front_preentry_soft_acc_distance_to_end_m_, 120.0);
    pnh_.param("target_front_overlap_soft_acc_enabled",
               target_front_overlap_soft_acc_enabled_, false);
    pnh_.param("target_front_overlap_soft_acc_start_abs_d_m",
               target_front_overlap_soft_acc_start_abs_d_m_, 2.7);
    pnh_.param("target_front_overlap_soft_max_decel_mps2",
               target_front_overlap_soft_max_decel_mps2_, 1.5);
    pnh_.param("active_rear_guard_enabled", active_rear_guard_enabled_, false);
    pnh_.param("active_rear_guard_min_gap_m", active_rear_guard_min_gap_m_, 9.0);
    pnh_.param("active_rear_guard_min_ttc_sec", active_rear_guard_min_ttc_sec_, 3.0);
    pnh_.param("active_rear_guard_min_headway_sec",
               active_rear_guard_min_headway_sec_, 1.2);
    pnh_.param("active_rear_guard_abort_before_target_center_abs_d_m",
               active_rear_guard_abort_before_target_center_abs_d_m_, 1.75);
    pnh_.param("active_rear_guard_perception_control_delay_sec",
               active_rear_guard_perception_control_delay_sec_, 0.50);
    pnh_.param("active_rear_guard_stationary_remaining_sec",
               active_rear_guard_stationary_remaining_sec_, 8.0);
    pnh_.param("active_rear_escape_max_accel_mps2",
               active_rear_escape_max_accel_mps2_, 2.0);
    pnh_.param("merge_priority_enabled", merge_priority_enabled_, true);
    pnh_.param("merge_priority_rear_safe_gap_m", merge_priority_rear_safe_gap_m_, 4.5);
    pnh_.param("merge_priority_rear_min_gap_m", merge_priority_rear_min_gap_m_, 4.0);
    pnh_.param("merge_priority_rear_max_closing_mps",
               merge_priority_rear_max_closing_mps_, 3.5);
    pnh_.param("merge_priority_rear_min_ttc_sec",
               merge_priority_rear_min_ttc_sec_, 1.8);
    pnh_.param("ego_lane_priority_enabled", ego_lane_priority_enabled_, true);
    pnh_.param("ego_lane_priority_current_max_abs_d_m",
               ego_lane_priority_current_max_abs_d_m_, 1.25);
    pnh_.param("ego_lane_priority_front_hard_gap_m",
               ego_lane_priority_front_hard_gap_m_, 8.0);
    pnh_.param("lane_end_stop_margin_m", lane_end_stop_margin_m_, 2.0);
    pnh_.param("lane_end_comfort_decel_mps2", lane_end_comfort_decel_mps2_, 2.5);
    pnh_.param("lane_end_merge_creep_speed_mps", lane_end_merge_creep_speed_mps_, 4.0);
    pnh_.param("lane_end_hard_stop_distance_m", lane_end_hard_stop_distance_m_, 6.0);
    pnh_.param("lane_end_merge_wait_stop_enabled",
               lane_end_merge_wait_stop_enabled_, true);
    pnh_.param("lane_end_merge_wait_stop_distance_m",
               lane_end_merge_wait_stop_distance_m_, 35.0);
    pnh_.param("stop_at_lane_end_enabled", stop_at_lane_end_enabled_, true);
    pnh_.param("lane_end_stop_prepare_distance_m", lane_end_stop_prepare_distance_m_, 45.0);
    pnh_.param("lane_end_stop_prepare_time_sec", lane_end_stop_prepare_time_sec_, 4.0);
    pnh_.param("use_nearby_vehicles", use_nearby_vehicles_, true);
    pnh_.param("use_nearby_overlap_as_acc_lead",
               use_nearby_overlap_as_acc_lead_,
               false);
    pnh_.param("nearby_acc_overlap_only", nearby_acc_overlap_only_, false);
    pnh_.param("nearby_overlap_lateral_margin_m",
               nearby_overlap_lateral_margin_m_,
               0.20);
    pnh_.param("lead_requires_center_ahead", lead_requires_center_ahead_, false);
    // 미성숙 트랙(관측 프레임 수가 적은 트랙)에는 완전 정지를 명령하지 않는다.
    // 유령은 칼만 예측으로 연명해 hits 가 낮게 유지되는 반면 실차는 빠르게 쌓인다
    // (15 개 bag, 대표 프레임 기준: 실차 중앙 24~35, 유령 중앙 4~9).
    // lc_gt 2026-09-10-12-18 에서 hits 2~7 의 유령 트랙이 current_front 가 되어
    // speedForLead 의 critical_gap 조기 반환으로 목표속도 0 을 명령했고, 자차가
    // 22.0 -> 14.4 m/s 로 급감속한 뒤 저속에 고착됐다.
    // 감속 자체는 그대로 허용하고 '완전 정지'만 막는다.  장애물을 버리지 않으므로
    // 실차에 대해서도 안전하다.  0 이면 비활성.
    pnh_.param("immature_lead_min_hits", immature_lead_min_hits_, 0);
    pnh_.param("immature_lead_min_speed_mps", immature_lead_min_speed_mps_, 2.0);
    pnh_.param("nearby_acc_lead_max_abs_d_m",
               nearby_acc_lead_max_abs_d_m_,
               1.35);
    pnh_.param("nearby_merge_front_range_m", nearby_merge_front_range_m_, 120.0);
    pnh_.param("nearby_merge_rear_range_m", nearby_merge_rear_range_m_, 20.0);
    pnh_.param("use_all_nearby_when_lane_end_pressure",
               use_all_nearby_when_lane_end_pressure_, true);
    pnh_.param("lane_end_wait_gap_front_only",
               lane_end_wait_gap_front_only_, true);
    pnh_.param("merge_keep_rolling_enabled",
               merge_keep_rolling_enabled_, true);
    pnh_.param("merge_keep_rolling_min_speed_mps",
               merge_keep_rolling_min_speed_mps_, 6.0);
    pnh_.param("merge_keep_rolling_front_block_gap_m",
               merge_keep_rolling_front_block_gap_m_, 8.0);
    pnh_.param("merge_keep_rolling_low_speed_lead_mps",
               merge_keep_rolling_low_speed_lead_mps_, 3.0);
    pnh_.param("merge_keep_rolling_low_speed_lead_max_gap_m",
               merge_keep_rolling_low_speed_lead_max_gap_m_, 80.0);
    if (!pnh_.getParam("lane_end_stop_lane_ids", lane_end_stop_lane_ids_)) {
      lane_end_stop_lane_ids_ = {3};
    }
    std::sort(lane_end_stop_lane_ids_.begin(), lane_end_stop_lane_ids_.end());
    lane_end_stop_lane_ids_.erase(
        std::unique(lane_end_stop_lane_ids_.begin(), lane_end_stop_lane_ids_.end()),
        lane_end_stop_lane_ids_.end());

    targets_sub_ = nh_.subscribe("/smpc/targets", 10, &AccSpeedPlanner::targetsCallback, this);
    odom_sub_ = nh_.subscribe("/odom", 10, &AccSpeedPlanner::odomCallback, this);
    mission_speed_sub_ =
        nh_.subscribe(mission_speed_topic_, 10, &AccSpeedPlanner::missionSpeedCallback, this);
    status_sub_ = nh_.subscribe("/smpc/lane_change_status", 10, &AccSpeedPlanner::statusCallback, this);
    decision_sub_ = nh_.subscribe("/smpc/decision", 10, &AccSpeedPlanner::decisionCallback, this);
    behavior_request_sub_ = nh_.subscribe(
        behavior_request_topic_, 10, &AccSpeedPlanner::behaviorRequestCallback, this);
    endpoint_connector_active_sub_ = nh_.subscribe(
        "/path_switch_connector_active", 10,
        &AccSpeedPlanner::endpointConnectorActiveCallback, this);
    path_number_sub_ = nh_.subscribe(
        "/path_number", 10, &AccSpeedPlanner::pathNumberCallback, this);
    gps_state_sub_ = nh_.subscribe(
        "/gps_state", 10, &AccSpeedPlanner::gpsStateCallback, this);

    speed_pub_ = nh_.advertise<std_msgs::Float64>("/smpc/target_speed_mps", 10);
    raw_speed_pub_ = nh_.advertise<std_msgs::Float64>("/smpc/acc_raw_target_speed_mps", 10);
    active_lead_pub_ = nh_.advertise<std_msgs::String>("/smpc/acc_active_lead", 10);
    behavior_status_pub_ = nh_.advertise<smpc_lane_change::BehaviorLongitudinalStatus>(
        behavior_status_topic_, 10);

    const double period = 1.0 / std::max(1.0, publish_rate_hz_);
    timer_ = nh_.createTimer(ros::Duration(period), &AccSpeedPlanner::timerCallback, this);
    filtered_speed_mps_ = mission_speed_fallback_mps_;
  }

 private:
  void targetsCallback(const smpc_lane_change::TargetVehicleSet::ConstPtr& msg) {
    targets_ = *msg;
    targets_stamp_ = ros::Time::now();
    have_targets_ = true;
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    ego_speed_mps_ = std::hypot(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    have_ego_speed_ = true;
  }

  void missionSpeedCallback(const std_msgs::Float64::ConstPtr& msg) {
    if (!std::isfinite(msg->data)) return;
    mission_speed_target_mps_ = std::max(0.0, msg->data);
    mission_speed_stamp_ = ros::Time::now();
    have_mission_speed_target_ = true;
  }

  void statusCallback(const smpc_lane_change::LaneChangeStatus::ConstPtr& msg) {
    if (msg->active && !lane_change_active_) {
      active_initial_target_abs_d_ =
          std::isfinite(targets_.ego_d_target)
              ? std::max(0.85, std::abs(targets_.ego_d_target)) : 3.5;
    }
    lane_change_active_ = msg->active;
    lane_change_elapsed_sec_ = std::max(0.0, msg->elapsed_sec);
    lane_change_expected_duration_sec_ = std::max(0.1, msg->expected_duration_sec);
  }

  void endpointConnectorActiveCallback(const std_msgs::Bool::ConstPtr& msg) {
    endpoint_connector_active_ = msg->data;
    endpoint_connector_active_stamp_ = ros::Time::now();
    have_endpoint_connector_status_ = true;
  }

  void pathNumberCallback(const std_msgs::Int32::ConstPtr& msg) {
    active_path_number_ = msg->data;
    have_active_path_number_ = true;
  }

  void gpsStateCallback(const waypoint_maker::State::ConstPtr& msg) {
    // /path_number is a one-shot command.  /gps_state is the waypoint system's
    // continuously acknowledged active path, so it also closes the brief
    // command/target delivery race after a path switch.
    active_path_number_ = msg->lane_number;
    have_active_path_number_ = true;
  }

  bool targetsMatchActivePath() const {
    // Preserve startup behavior until the latched path command is available.
    // Once available, it is the authoritative waypoint path epoch.
    return !have_active_path_number_ ||
           targets_.current_lane_id == active_path_number_;
  }

  void decisionCallback(const smpc_lane_change::LaneChangeDecision::ConstPtr& msg) {
    lane_change_requested_ =
        msg->request &&
        msg->mode == smpc_lane_change::LaneChangeDecision::CHANGE_LEFT;
    requested_decision_current_lane_id_ = msg->current_lane_id;
    requested_decision_target_lane_id_ = msg->target_lane_id;
    decision_stamp_ = ros::Time::now();
    // KEEP is a normal receding-horizon result, including WAIT_FOR_GAP and
    // HOLD_THEN_CHANGE preparation.  LaneChangeDecision has no explicit
    // rejection state, so it must never be converted into a lane-end stop.
    // A fresh CHANGE_NOW command supersedes any earlier preparation cap even
    // if the two topics are delivered in a different order.
    if (lane_change_requested_) have_behavior_request_ = false;
  }

  void behaviorRequestCallback(
      const smpc_lane_change::BehaviorLongitudinalRequest::ConstPtr& msg) {
    const ros::Time now = ros::Time::now();
    // An explicit inactive message is cancellation, not a request that may
    // linger until the timeout.  Reject source-stale/future requests here as
    // well as in the predicate below: receiver time alone is unsafe when a
    // queued message arrives late.
    if (!msg->active || msg->header.stamp.isZero() ||
        msg->header.stamp > now + ros::Duration(0.05) ||
        (behavior_request_timeout_sec_ > 0.0 &&
         (now - msg->header.stamp).toSec() > behavior_request_timeout_sec_)) {
      have_behavior_request_ = false;
      return;
    }
    behavior_request_ = *msg;
    behavior_request_stamp_ = now;
    have_behavior_request_ = true;
    if (msg->action ==
        smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP) {
      wait_for_gap_seen_stamp_ = now;
      wait_for_gap_current_lane_id_ = msg->current_lane_id;
      wait_for_gap_target_lane_id_ = msg->target_lane_id;
    }
  }

  bool recentLaneChangeRequest() const {
    if (!lane_change_requested_) return false;
    if (decision_request_timeout_sec_ > 0.0 &&
        (ros::Time::now() - decision_stamp_).toSec() > decision_request_timeout_sec_) {
      return false;
    }
    // A delayed CHANGE_LEFT for a previous lane pair must never suppress
    // current-lane endpoint/merge-stop protection.
    return requested_decision_current_lane_id_ == targets_.current_lane_id &&
           requested_decision_target_lane_id_ == targets_.target_lane_id;
  }

  bool laneChangeIntentActive() const {
    // A preparation plan is deliberately not a lane-change reservation.
    // Endpoint/merge-stop protection remains active until the supervisor has
    // a fresh CHANGE_LEFT command or reports an active maneuver.
    return lane_change_active_ || recentLaneChangeRequest();
  }

  // Endpoint wait/stop protection is released only once the supervisor has
  // actually started the manoeuvre.  Releasing it on a bare CHANGE_LEFT
  // flipped the ACC reason to "none" while the ego was still braking on the
  // wait profile; SMPC read that as unstable ACC and withdrew the request, so
  // the supervisor never saw three consecutive requests (lc_gt
  // 2026-09-15-19-33-03: 7 request/withdraw cycles, 0->1 and 2->3 merges
  // delayed until standstill).  Once active, lane_change_lane_end_cap takes
  // over the endpoint deadline.
  bool laneEndProtectionReleased() const {
    return lane_end_hold_until_lane_change_active_
        ? lane_change_active_
        : laneChangeIntentActive();
  }

  bool behaviorPreparationActive() const {
    if (!behavior_request_enabled_ || !have_behavior_request_ ||
        !behavior_request_.active || !have_targets_) {
      return false;
    }
    if (behavior_request_timeout_sec_ > 0.0 &&
        (ros::Time::now() - behavior_request_stamp_).toSec() >
            behavior_request_timeout_sec_) {
      return false;
    }
    const ros::Time now = ros::Time::now();
    if (behavior_request_.header.stamp.isZero() ||
        behavior_request_.header.stamp > now + ros::Duration(0.05) ||
        (behavior_request_timeout_sec_ > 0.0 &&
         (now - behavior_request_.header.stamp).toSec() >
             behavior_request_timeout_sec_)) {
      return false;
    }
    if (!std::isfinite(behavior_request_.speed_cap_mps) ||
        !std::isfinite(behavior_request_.requested_accel_mps2)) {
      return false;
    }
    if (behavior_request_.current_lane_id != targets_.current_lane_id ||
        behavior_request_.target_lane_id != targets_.target_lane_id) {
      return false;
    }
    if ((lane_change_active_ || recentLaneChangeRequest()) &&
        !behavior_request_apply_while_lane_change_) {
      return false;
    }
    const uint8_t action = behavior_request_.action;
    if (action != smpc_lane_change::BehaviorLongitudinalRequest::YIELD &&
        action != smpc_lane_change::BehaviorLongitudinalRequest::HOLD &&
        action != smpc_lane_change::BehaviorLongitudinalRequest::ACCELERATE &&
        action != smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP) {
      return false;
    }
    if (action == smpc_lane_change::BehaviorLongitudinalRequest::ACCELERATE &&
        !behavior_request_apply_accel_) {
      return false;
    }
    return true;
  }

  bool endpointConnectorActive() const {
    // A missing, false, reset-time, or stale Bool must never retain the
    // connector speed cap.  The publisher therefore needs to refresh true
    // while the generated connector is actually being followed.
    if (!have_endpoint_connector_status_ || !endpoint_connector_active_ ||
        endpoint_connector_status_timeout_sec_ <= 0.0) {
      return false;
    }
    const double age_sec =
        (ros::Time::now() - endpoint_connector_active_stamp_).toSec();
    return age_sec >= 0.0 && age_sec <= endpoint_connector_status_timeout_sec_;
  }

  bool waitForGapActive() const {
    // WAIT_FOR_GAP is a fresh, lane-matched SMPC state: the merge is required
    // but neither CHANGE_NOW nor a safe future t_LC exists yet.  It is not a
    // lead-vehicle ACC command, but it must re-arm the CSV-endpoint fallback
    // once the lane-change deadline becomes urgent.
    if (behaviorPreparationActive() &&
        behavior_request_.action ==
            smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP) {
      return true;
    }
    return waitForGapHeldForPendingChange();
  }

  // A fresh CHANGE_LEFT clears the behavior request on arrival
  // (decisionCallback), which dropped the WAIT_FOR_GAP endpoint hold the
  // moment SMPC asked to change: lane_end_wait_no_safe_gap -> "none", the
  // same request/withdraw loop as the intent release (ACC+SMPC+supervisor
  // chain replay: 4 of 11 loops took this path).  Keep the last lane-matched
  // WAIT_FOR_GAP until the supervisor actually starts the manoeuvre.
  bool waitForGapHeldForPendingChange() const {
    // <= 0 disables the hold outright; with sim time a request and the next
    // planner tick can share a stamp, so "age <= 0" alone is not "off".
    if (!lane_end_hold_until_lane_change_active_ ||
        lane_end_hold_wait_for_gap_latch_sec_ <= 0.0 || lane_change_active_ ||
        !recentLaneChangeRequest() || wait_for_gap_seen_stamp_.isZero()) {
      return false;
    }
    if (wait_for_gap_current_lane_id_ != targets_.current_lane_id ||
        wait_for_gap_target_lane_id_ != targets_.target_lane_id) {
      return false;
    }
    return (ros::Time::now() - wait_for_gap_seen_stamp_).toSec() <=
        std::max(0.0, lane_end_hold_wait_for_gap_latch_sec_);
  }

  const char* behaviorActionName() const {
    switch (behavior_request_.action) {
      case smpc_lane_change::BehaviorLongitudinalRequest::YIELD: return "behavior_yield";
      case smpc_lane_change::BehaviorLongitudinalRequest::HOLD: return "behavior_hold";
      case smpc_lane_change::BehaviorLongitudinalRequest::ACCELERATE: return "behavior_accel";
      case smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP:
        return "behavior_wait_for_gap";
      case smpc_lane_change::BehaviorLongitudinalRequest::NONE:
      default: return "behavior_none";
    }
  }

  double currentCruiseSpeedMps() const {
    if (have_mission_speed_target_ &&
        (mission_speed_timeout_sec_ <= 0.0 ||
         (ros::Time::now() - mission_speed_stamp_).toSec() <= mission_speed_timeout_sec_)) {
      return mission_speed_target_mps_;
    }
    return mission_speed_fallback_mps_;
  }

  double leadHalfS(const smpc_lane_change::TargetVehicle& lead) const {
    return lead.half_s > 1e-3 ? lead.half_s : 0.5 * lead.length;
  }

  double leadFrontDeltaS(const smpc_lane_change::TargetVehicle& lead) const {
    return lead.front_delta_s != 0.0 ? lead.front_delta_s : lead.delta_s + leadHalfS(lead);
  }

  double leadRearDeltaS(const smpc_lane_change::TargetVehicle& lead) const {
    return lead.rear_delta_s != 0.0 ? lead.rear_delta_s : lead.delta_s - leadHalfS(lead);
  }

  double frontBumperGap(const smpc_lane_change::TargetVehicle& lead) const {
    return leadRearDeltaS(lead) - ego_front_extent_m_;
  }

  double rearBumperGap(const smpc_lane_change::TargetVehicle& rear) const {
    return -leadFrontDeltaS(rear) - ego_rear_extent_m_;
  }

  bool leadIsInFront(const smpc_lane_change::TargetVehicle& lead) const {
    if (!lead.valid || leadFrontDeltaS(lead) <= -ego_rear_extent_m_) return false;
    // 앞끝만 자차를 넘은 차량(= 자차와 나란하거나 뒤에 바짝 붙은 차)은 ACC 리드로
    // 쓰지 않는다.  기존 조건은 차량 앞끝만 보므로, 중심이 자차 뒤 1.4 m 인
    // 차량도 앞차가 되어 speedForLead 의 critical_gap 조기 반환(목표속도 0)에
    // 걸렸다.  lc_gt 2026-09-08-14-45: 뒤따르던 HD65 트럭이 delta_s -1.39 m 로
    // current_front 가 되어 73 프레임 목표속도 0, 자차 11.4 -> 0.43 m/s.
    // 뒤차 때문에 급제동하는 것은 후방 추돌을 키우는 방향이다.
    //
    // leadIsInFront 는 current_front / target_front / nearby_overlap 세 경로가
    // 모두 통과하는 유일한 지점이라, 여기서 한 번만 막으면 경로를 바꿔 다시
    // 들어오지 않는다.  대표 차량 선정(target_selector)은 건드리지 않으므로
    // SMPC 의 게이트가 보는 차량 집합은 그대로다.
    if (lead_requires_center_ahead_ && std::isfinite(lead.delta_s) &&
        lead.delta_s <= 0.0) {
      return false;
    }
    return true;
  }

  double leadSpeedMps(const smpc_lane_change::TargetVehicle& lead,
                      double cruise_speed_mps) const {
    const double lead_speed =
        std::isfinite(lead.v_long) ? lead.v_long : ego_speed_mps_;
    return std::clamp(lead_speed, 0.0, std::max(0.0, cruise_speed_mps));
  }

  double leadTtcSec(const smpc_lane_change::TargetVehicle& lead) const {
    const double lead_speed = std::isfinite(lead.v_long) ? lead.v_long : ego_speed_mps_;
    const double closing_speed = ego_speed_mps_ - std::max(0.0, lead_speed);
    if (closing_speed <= 0.1) return std::numeric_limits<double>::infinity();
    return frontBumperGap(lead) / closing_speed;
  }

  double leadClosingSpeedMps(const smpc_lane_change::TargetVehicle& lead) const {
    if (!leadIsInFront(lead)) return 0.0;
    const double lead_speed = std::isfinite(lead.v_long) ? lead.v_long : ego_speed_mps_;
    return std::max(0.0, ego_speed_mps_ - std::max(0.0, lead_speed));
  }

  bool hardBrakeRequiredForLead(const smpc_lane_change::TargetVehicle& lead) const {
    if (!leadIsInFront(lead)) return false;
    if (frontBumperGap(lead) <= critical_gap_m_) return true;
    return leadTtcSec(lead) <= critical_ttc_sec_;
  }

  double requiredDecelForLead(const smpc_lane_change::TargetVehicle& lead) const {
    if (!leadIsInFront(lead)) return 0.0;
    const double closing_speed = leadClosingSpeedMps(lead);
    const double clearance = frontBumperGap(lead) - standstill_gap_m_;
    if (closing_speed <= 1e-3) return 0.0;
    if (clearance <= 1e-3) return std::numeric_limits<double>::infinity();
    return closing_speed * closing_speed / (2.0 * clearance);
  }

  // Deceleration that removes the closing speed before the gap shrinks to
  // standstill_gap + headway * ego speed, assuming a constant-speed lead.
  // Infinite once that margin is used up (the existing limits then apply).
  double leadRequiredDecelForHeadway(const smpc_lane_change::TargetVehicle& lead) const {
    const double closing_speed = leadClosingSpeedMps(lead);
    if (closing_speed <= 1e-3) return 0.0;
    const double margin = frontBumperGap(lead) - standstill_gap_m_ -
        std::max(0.0, lead_required_decel_limit_headway_sec_) * ego_speed_mps_;
    if (margin <= 0.5) return std::numeric_limits<double>::infinity();
    return closing_speed * closing_speed / (2.0 * margin);
  }

  // The lead whose follow speed last lowered raw_target.
  struct LeadBinding {
    bool valid{false};
    double target_mps{0.0};
    double required_decel_mps2{0.0};
  };

  bool rapidBrakeRequiredForLead(const smpc_lane_change::TargetVehicle& lead) const {
    // A newly appearing lead has no previous ACC gap history.  If maintaining
    // the standstill gap already needs a strong brake, bypass the target-speed
    // slew limiter immediately instead of waiting for the normal critical TTC.
    if (!rapid_lead_brake_enabled_ || !leadIsInFront(lead)) return false;
    const double closing_speed = leadClosingSpeedMps(lead);
    if (closing_speed < rapid_lead_min_closing_speed_mps_) return false;
    if (leadTtcSec(lead) <= rapid_lead_ttc_sec_) return true;
    return requiredDecelForLead(lead) >= rapid_lead_required_decel_mps2_;
  }

  // 미성숙 트랙이면 완전 정지 대신 이 하한까지만 내린다.
  // The endpoint connector always drives onto the waypoint system's active
  // path (switched when the connector is armed), so that path is the merge
  // lane: /gps_state lane_number matched it in 1920/1920 connector frames of
  // 18 lc_gt connectors.  targets_.target_lane_id follows supervisor state and
  // flipped to the next lane in 5/597 frames (4 in the last 0.35 s), which
  // would briefly drop a 0->1 cap to the lane-2 value while merging.
  double endpointConnectorSpeedCapMps() const {
    const auto& by_lane = endpoint_connector_speed_cap_by_target_lane_mps_;
    const int lane = have_active_path_number_ ? active_path_number_
                                              : targets_.target_lane_id;
    if (lane >= 0 && lane < static_cast<int>(by_lane.size()) && by_lane[lane] > 0.0) {
      return by_lane[lane];
    }
    return endpoint_connector_speed_cap_mps_;
  }

  double leadStopFloorMps(const smpc_lane_change::TargetVehicle& lead) const {
    if (immature_lead_min_hits_ <= 0) return 0.0;
    if (lead.track_hits >= immature_lead_min_hits_) return 0.0;
    return std::max(0.0, immature_lead_min_speed_mps_);
  }

  double speedForLead(const smpc_lane_change::TargetVehicle& lead,
                      double cruise_speed_mps) const {
    if (!leadIsInFront(lead)) return cruise_speed_mps;

    const double stop_floor = std::min(leadStopFloorMps(lead), cruise_speed_mps);
    const double bumper_gap = frontBumperGap(lead);
    if (bumper_gap <= critical_gap_m_) return stop_floor;

    const double lead_speed = leadSpeedMps(lead, cruise_speed_mps);
    const double closing_speed = std::max(0.0, ego_speed_mps_ - lead_speed);
    const double closing_brake_gap =
        (braking_decel_mps2_ > 1e-3)
            ? (closing_speed * closing_speed) / (2.0 * braking_decel_mps2_)
            : 0.0;
    const double desired_gap = standstill_gap_m_ +
        time_headway_sec_ * ego_speed_mps_ +
        closing_speed_gap_time_sec_ * closing_speed +
        closing_brake_gap;
    const double gap_error = bumper_gap - desired_gap;
    double commanded = lead_speed + gap_gain_ * gap_error;

    // Normal ACC should follow the lead vehicle instead of commanding an
    // immediate full stop whenever the desired time-headway gap is not met.
    // Keep hard stops for truly critical gap/TTC only; otherwise allow the
    // controller to trail the lead speed by a bounded amount to open the gap.
    const double follow_floor =
        std::max(0.0, lead_speed - std::max(0.0, max_follow_speed_drop_mps_));
    commanded = std::max(commanded, follow_floor);

    // Braking-distance guard: do not command a speed that would require more
    // than braking_decel_mps2 to settle behind the lead vehicle.
    if (braking_decel_mps2_ > 1e-3) {
      const double braking_clearance = std::max(0.0, bumper_gap - standstill_gap_m_);
      const double braking_limit =
          std::sqrt(lead_speed * lead_speed +
                    2.0 * braking_decel_mps2_ * braking_clearance);
      commanded = std::min(commanded, braking_limit);
    }

    // TTC guard: if ego is closing, keep predicted TTC above soft_ttc_sec.
    // If critical_ttc_sec is already violated, request an immediate stop.
    // hard_ttc_sec only pulls the command toward lead speed; it no longer
    // creates stop-and-go behavior by itself.
    const double ttc = leadTtcSec(lead);
    if (std::isfinite(ttc)) {
      if (ttc <= critical_ttc_sec_) return stop_floor;
      const double ttc_clearance = std::max(0.0, bumper_gap - emergency_gap_m_);
      const double ttc_limit = lead_speed + ttc_clearance / std::max(0.1, soft_ttc_sec_);
      commanded = std::min(commanded, ttc_limit);
      if (ttc <= hard_ttc_sec_) {
        commanded = std::min(commanded, lead_speed);
      }
    }

    return std::clamp(commanded, min_speed_mps_, cruise_speed_mps);
  }

  bool isRearVehicle(const smpc_lane_change::TargetVehicle& vehicle) const {
    return vehicle.role.find("rear") != std::string::npos;
  }

  bool isNearbyOverlapVehicle(const smpc_lane_change::TargetVehicle& vehicle) const {
    return vehicle.role.find("nearby_overlap") != std::string::npos;
  }

  // nearby_overlap denotes longitudinal bbox overlap on the lane to which the
  // object was projected.  It must also overlap ego laterally before it is
  // treated as an ACC lead; otherwise a vehicle beside ego in an adjacent lane
  // could cause unnecessary braking.
  bool nearbyOverlapOccupiesEgoCorridor(
      const smpc_lane_change::TargetVehicle& vehicle) const {
    if (!isNearbyOverlapVehicle(vehicle) || !std::isfinite(vehicle.d)) return false;

    double ego_d = std::numeric_limits<double>::quiet_NaN();
    if (vehicle.lane_id == targets_.current_lane_id) {
      ego_d = targets_.ego_d_current;
    } else if (vehicle.lane_id == targets_.target_lane_id) {
      ego_d = targets_.ego_d_target;
    }
    if (!std::isfinite(ego_d)) return false;

    const double lateral_overlap_extent =
        0.5 * ego_width_m_ + std::max(0.0, vehicle.half_d) +
        std::max(0.0, nearby_overlap_lateral_margin_m_);
    return std::abs(vehicle.d - ego_d) <= lateral_overlap_extent;
  }

  bool laneEndPressureActive() const {
    return targets_.lane_change_prepare ||
           targets_.lane_change_urgent ||
           targets_.emergency_stop_required;
  }

  bool mergePriorityRearEligible(const smpc_lane_change::TargetVehicle& rear) const {
    if (!merge_priority_enabled_ || !laneEndPressureActive() || !rear.valid) return false;
    if (!isRearVehicle(rear)) return false;

    const double gap = rearBumperGap(rear);
    if (!std::isfinite(gap) || gap < merge_priority_rear_min_gap_m_) return false;

    const double rear_speed = std::isfinite(rear.v_long) ? rear.v_long : 0.0;
    const double closing = std::max(0.0, rear_speed - ego_speed_mps_);
    const double ttc = closing > 0.1
        ? gap / closing
        : std::numeric_limits<double>::infinity();
    if (closing > merge_priority_rear_max_closing_mps_ &&
        ttc < merge_priority_rear_min_ttc_sec_) {
      return false;
    }
    return true;
  }

  double effectiveTargetRearGapRequirement() const {
    if (!mergePriorityRearEligible(targets_.target_rear)) return min_target_rear_gap_m_;
    return std::min(min_target_rear_gap_m_,
                    std::max(0.0, merge_priority_rear_safe_gap_m_));
  }

  bool targetLaneEntryEstablished() const {
    if (targets_.current_lane_id == targets_.target_lane_id) return true;
    if (!std::isfinite(targets_.ego_d_target)) return false;
    return std::abs(targets_.ego_d_target) <=
           std::max(0.0, target_front_acc_entry_abs_d_m_);
  }

  bool targetFrontHardAccAllowed() const {
    if (!target_front_acc_requires_target_lane_entry_) return true;
    return targetLaneEntryEstablished();
  }

  // ego_d_target is measured from the target-lane center.  For a 3.5 m lane
  // and an IONIQ 5 width of about 1.9 m, |d_target| <= 2.7 m means the ego
  // footprint has started to occupy the target lane.  This is deliberately
  // distinct from targetLaneEntryEstablished(): it enables gentle following
  // as the maneuver starts, while full TTC/hard-brake authority remains
  // reserved for the later target-lane-entry phase.
  bool targetLaneFootprintOverlapsTargetLane() const {
    if (targets_.current_lane_id == targets_.target_lane_id) return true;
    if (!std::isfinite(targets_.ego_d_target)) return false;
    return std::abs(targets_.ego_d_target) <=
           std::max(0.0, target_front_overlap_soft_acc_start_abs_d_m_);
  }

  bool targetFrontOverlapSoftAccAllowed() const {
    return target_front_overlap_soft_acc_enabled_ && lane_change_active_ &&
           !targetLaneEntryEstablished() &&
           targetLaneFootprintOverlapsTargetLane();
  }

  bool targetFrontPreEntrySoftAccAllowed() const {
    if (!target_front_preentry_soft_acc_enabled_) return false;
    if (targetLaneEntryEstablished()) return true;
    if (!laneEndPressureActive()) return false;
    return targets_.distance_to_lane_end <=
           std::max(0.0, target_front_preentry_soft_acc_distance_to_end_m_);
  }

  bool targetFrontAccAllowed() const {
    return targetFrontHardAccAllowed() || targetFrontOverlapSoftAccAllowed() ||
           targetFrontPreEntrySoftAccAllowed();
  }

  bool isTargetLaneRearOrOverlap(
      const smpc_lane_change::TargetVehicle& vehicle) const {
    return vehicle.valid && vehicle.lane_id == targets_.target_lane_id &&
        vehicle.rear_delta_s <= 0.0;
  }

  double activeRearRemainingSec() const {
    if (!lane_change_active_) return 0.0;
    if (!std::isfinite(targets_.ego_d_target)) {
      return std::max(0.0,
          lane_change_expected_duration_sec_ - lane_change_elapsed_sec_);
    }
    const double remaining_lateral = std::max(
        0.0, std::abs(targets_.ego_d_target) - target_front_acc_entry_abs_d_m_);
    const double initial_span = std::max(
        0.1, active_initial_target_abs_d_ - target_front_acc_entry_abs_d_m_);
    double remaining = lane_change_expected_duration_sec_ *
        std::clamp(remaining_lateral / initial_span, 0.0, 1.0);
    if (ego_speed_mps_ < 0.20) {
      remaining = std::max(remaining, active_rear_guard_stationary_remaining_sec_);
    }
    return remaining;
  }

  bool activeRearIntrusion() const {
    if (!active_rear_guard_enabled_ || !lane_change_active_) return false;
    const double min_gap = std::max(0.0, active_rear_guard_min_gap_m_);
    const double min_ttc = std::max(0.0, active_rear_guard_min_ttc_sec_);
    const double remaining_sec = activeRearRemainingSec();
    const auto intrudes = [this, min_gap, min_ttc, remaining_sec](
        const smpc_lane_change::TargetVehicle& rear) {
      if (!isTargetLaneRearOrOverlap(rear)) return false;
      const double gap = rearBumperGap(rear);
      const double rear_speed = std::max(0.0, rear.v_long);
      const double closing = std::max(0.0, rear_speed - ego_speed_mps_);
      const double required_gap = min_gap + closing *
          (remaining_sec +
           std::max(0.0, active_rear_guard_perception_control_delay_sec_));
      const double ttc = closing > 0.1
          ? gap / closing
          : std::numeric_limits<double>::infinity();
      return gap <= required_gap || (std::isfinite(ttc) && ttc <= min_ttc);
    };

    if (intrudes(targets_.target_rear)) return true;
    for (const auto& nearby : targets_.nearby_vehicles) {
      if (intrudes(nearby)) return true;
    }
    return false;
  }

  bool activeRearEscapeAccelerationAllowed() const {
    if (!activeRearIntrusion()) return false;
    if (targets_.target_front.valid) {
      const double front_gap = frontBumperGap(targets_.target_front);
      const double required = std::max(
          min_target_front_gap_m_,
          time_headway_sec_ * std::max(
              ego_speed_mps_, std::max(0.0, targets_.target_front.v_long)));
      if (front_gap < required ||
          hardBrakeRequiredForLead(targets_.target_front) ||
          rapidBrakeRequiredForLead(targets_.target_front)) {
        return false;
      }
    }
    return true;
  }

  bool suppressTargetFrontSoftAccForRearIntrusion() const {
    if (!targetFrontOverlapSoftAccAllowed() || !activeRearIntrusion()) return false;
    // Before the target-lane centre, supervisor can still return to source if
    // the bounded escape is infeasible.  Past that point retain ordinary
    // target-front following rather than masking a front constraint.
    if (!std::isfinite(targets_.ego_d_target) ||
        std::abs(targets_.ego_d_target) <=
            std::max(0.0, active_rear_guard_abort_before_target_center_abs_d_m_)) {
      return false;
    }
    // Never suppress an actual front collision response.  This applies only to
    // the pre-entry, limited-deceleration target_front following branch.
    return !hardBrakeRequiredForLead(targets_.target_front) &&
        !rapidBrakeRequiredForLead(targets_.target_front);
  }

  bool shouldUseNearbyLead(const smpc_lane_change::TargetVehicle& lead) const {
    if (!use_nearby_vehicles_ || !lead.valid || isRearVehicle(lead)) return false;
    const bool nearby_overlap = isNearbyOverlapVehicle(lead);
    if (nearby_overlap && !use_nearby_overlap_as_acc_lead_) {
      return false;
    }
    if (nearby_acc_overlap_only_ && !nearby_overlap) return false;
    if (lead.role.find("nearby") != std::string::npos &&
        !nearby_overlap &&
        nearby_acc_lead_max_abs_d_m_ > 0.0 &&
        std::abs(lead.d) > nearby_acc_lead_max_abs_d_m_) {
      return false;
    }
    if (leadFrontDeltaS(lead) <= -ego_rear_extent_m_) return false;

    // A longitudinally overlapping nearby vehicle is a lead only when its
    // lateral bbox reaches ego's current physical corridor. This bypasses
    // lane-ID/pre-entry gating because it is an immediate overlap hazard.
    if (nearby_overlap) return nearbyOverlapOccupiesEgoCorridor(lead);

    if (lead.lane_id == targets_.current_lane_id) {
      return !egoLanePrioritySuppressesLead(lead);
    }
    if (lane_change_active_ && lead.lane_id == targets_.target_lane_id) {
      return targetFrontAccAllowed();
    }

    if (use_all_nearby_when_lane_end_pressure_ && laneEndPressureActive()) {
      if (egoLanePrioritySuppressesLead(lead)) return false;
      return lead.delta_s >= -nearby_merge_rear_range_m_ &&
             lead.delta_s <= nearby_merge_front_range_m_;
    }
    return false;
  }

  bool egoLanePrioritySuppressesLead(const smpc_lane_change::TargetVehicle& lead) const {
    if (!ego_lane_priority_enabled_ || !lead.valid) return false;

    // Lane-ownership priority is only for ordinary merge traffic.  A lead
    // that has already crossed the critical gap/TTC boundary must always be
    // passed to ACC, even if its lateral projection is slightly off-center.
    if (hardBrakeRequiredForLead(lead)) return false;

    const bool current_front = lead.role.find("current_front") != std::string::npos;
    const bool nearby = lead.role.find("nearby") != std::string::npos;
    if (!current_front && !nearby) return false;

    const double front_gap = frontBumperGap(lead);
    if (std::isfinite(front_gap) &&
        front_gap <= ego_lane_priority_front_hard_gap_m_) {
      return false;
    }

    if (current_front) {
      return std::abs(lead.d) > ego_lane_priority_current_max_abs_d_m_;
    }

    if (lead.lane_id == targets_.current_lane_id &&
        std::abs(lead.d) <= ego_lane_priority_current_max_abs_d_m_) {
      return false;
    }
    if (lead.lane_id == targets_.current_lane_id &&
        std::abs(lead.d) > ego_lane_priority_current_max_abs_d_m_) {
      return true;
    }
    if (lead.lane_id == targets_.target_lane_id &&
        lead.lane_id != targets_.current_lane_id) {
      return false;
    }

    return laneEndPressureActive();
  }

  void applyLeadLimit(const smpc_lane_change::TargetVehicle& lead,
                      const std::string& label,
                      double cruise_speed_mps,
                      double& raw_target,
                      std::string& active_lead,
                      bool& hard_brake_active,
                      double& max_active_closing_speed_mps,
                      bool allow_hard_brake = true,
                      bool soft_target_front_acc = false,
                      bool* soft_target_front_limited_speed = nullptr,
                      LeadBinding* binding = nullptr) const {
    const double candidate_speed = speedForLead(lead, cruise_speed_mps);
    if (lead.valid && candidate_speed < raw_target) {
      raw_target = candidate_speed;
      if (binding != nullptr) {
        binding->valid = true;
        binding->target_mps = candidate_speed;
        binding->required_decel_mps2 = leadRequiredDecelForHeadway(lead);
      }
      max_active_closing_speed_mps =
          std::max(max_active_closing_speed_mps, leadClosingSpeedMps(lead));
      const bool hard_brake = allow_hard_brake && hardBrakeRequiredForLead(lead);
      const bool rapid_brake = allow_hard_brake && !hard_brake &&
          rapidBrakeRequiredForLead(lead);
      hard_brake_active = hard_brake_active || hard_brake || rapid_brake;
      if (soft_target_front_limited_speed != nullptr) {
        *soft_target_front_limited_speed = soft_target_front_acc;
      }
      active_lead = label + ":" + std::to_string(lead.unique_id);
      if (hard_brake) active_lead += ":hard";
      if (rapid_brake) active_lead += ":rapid";
      if (soft_target_front_acc) {
        active_lead += ":overlap_soft";
      } else if (!allow_hard_brake && hardBrakeRequiredForLead(lead)) {
        active_lead += ":preentry_soft";
      }
    }
  }

  double closingAwareDecelLimit(double base_decel_mps2,
                                double closing_speed_mps) const {
    const double base = std::max(0.0, base_decel_mps2);
    if (!closing_decel_boost_enabled_) return base;
    if (closing_speed_mps <= closing_decel_boost_start_mps_) return base;

    const double full =
        std::max(closing_decel_boost_start_mps_ + 1e-3,
                 closing_decel_boost_full_mps_);
    const double ratio = std::clamp(
        (closing_speed_mps - closing_decel_boost_start_mps_) /
            (full - closing_decel_boost_start_mps_),
        0.0,
        1.0);
    const double boosted =
        base + ratio * (closing_decel_boost_max_decel_mps2_ - base);
    return std::max(base, boosted);
  }


  bool targetGapSafe() const {
    const double rear_gap_requirement = effectiveTargetRearGapRequirement();
    const bool front_safe = !targets_.target_front.valid ||
        frontBumperGap(targets_.target_front) > min_target_front_gap_m_;
    const bool rear_safe = !targets_.target_rear.valid ||
        rearBumperGap(targets_.target_rear) > rear_gap_requirement;
    return front_safe && rear_safe;
  }

  bool targetFrontBlocksMerge() const {
    return targets_.target_front.valid &&
           leadIsInFront(targets_.target_front) &&
           frontBumperGap(targets_.target_front) <= min_target_front_gap_m_;
  }

  bool shouldWaitAtLaneEndForGap() const {
    if (lane_end_wait_gap_front_only_) {
      return targetFrontBlocksMerge();
    }
    return !targetGapSafe();
  }

  // On lanes that must merge before their CSV endpoint, stop at a safe buffer
  // when SMPC cannot approve the merge.  This preserves enough local-path
  // points for MPC while target gaps and trajectory risk continue updating.
  bool mergeWaitStopRequired() const {
    if (!lane_end_merge_wait_stop_enabled_ || laneEndProtectionReleased()) return false;
    if (!targets_.lane_change_required || !targets_.lane_change_urgent) return false;
    // Ordinary KEEP_CRUISE is not a rejection.  A fresh WAIT_FOR_GAP is
    // different: no safe merge exists and the current CSV must not be run
    // through at cruise speed.  Start the existing endpoint speed profile in
    // the urgent zone; a new HOLD/CHANGE_NOW cancels this on the next cycle.
    return shouldWaitAtLaneEndForGap() || waitForGapActive();
  }

  double laneEndSpeedLimit() const {
    const double usable_distance = std::max(
        0.0, targets_.distance_to_lane_end - lane_end_stop_margin_m_);
    return std::sqrt(2.0 * lane_end_comfort_decel_mps2_ * usable_distance);
  }

  double mandatoryMergeLaneEndLimit() const {
    const double stop_distance = lane_end_merge_wait_stop_enabled_
        ? std::max(0.0, lane_end_merge_wait_stop_distance_m_)
        : std::max(0.0, lane_end_hard_stop_distance_m_);
    const double usable_distance =
        std::max(0.0, targets_.distance_to_lane_end - stop_distance);
    const double endpoint_limit =
        std::sqrt(2.0 * lane_end_comfort_decel_mps2_ * usable_distance);
    if (targets_.distance_to_lane_end <= stop_distance) {
      return 0.0;
    }
    return endpoint_limit;
  }

  bool shouldStopAtLaneEnd() const {
    if (!stop_at_lane_end_enabled_ || !std::isfinite(targets_.distance_to_lane_end)) return false;
    if (lane_end_stop_lane_ids_.empty()) return true;
    return std::find(lane_end_stop_lane_ids_.begin(), lane_end_stop_lane_ids_.end(),
                     targets_.current_lane_id) != lane_end_stop_lane_ids_.end();
  }

  bool vehicleBlocksMergeRolling(const smpc_lane_change::TargetVehicle& lead) const {
    if (!lead.valid || isRearVehicle(lead) || !leadIsInFront(lead)) return false;
    const double front_gap = frontBumperGap(lead);
    return std::isfinite(front_gap) &&
           front_gap <= merge_keep_rolling_front_block_gap_m_;
  }

  bool mergeRollingFrontBlocked() const {
    if (!egoLanePrioritySuppressesLead(targets_.current_front) &&
        vehicleBlocksMergeRolling(targets_.current_front)) {
      return true;
    }
    if (vehicleBlocksMergeRolling(targets_.target_front)) {
      return true;
    }
    for (const auto& lead : targets_.nearby_vehicles) {
      if (shouldUseNearbyLead(lead) && vehicleBlocksMergeRolling(lead)) {
        return true;
      }
    }
    return false;
  }

  bool lowSpeedLeadBlocksMergeRolling(const smpc_lane_change::TargetVehicle& lead) const {
    if (!lead.valid || isRearVehicle(lead) || !leadIsInFront(lead)) return false;
    const double front_gap = frontBumperGap(lead);
    if (!std::isfinite(front_gap) || front_gap < 0.0 ||
        front_gap > merge_keep_rolling_low_speed_lead_max_gap_m_) {
      return false;
    }
    const double lead_speed = std::isfinite(lead.v_long) ? lead.v_long : ego_speed_mps_;
    return std::max(0.0, lead_speed) <= merge_keep_rolling_low_speed_lead_mps_;
  }

  bool mergeRollingLowSpeedLeadBlocked() const {
    // A stopped/slow current-lane vehicle must be allowed to pull ACC below
    // the merge rolling floor well before the close-gap emergency threshold.
    if (lowSpeedLeadBlocksMergeRolling(targets_.current_front)) return true;
    if (lowSpeedLeadBlocksMergeRolling(targets_.target_front)) return true;
    for (const auto& lead : targets_.nearby_vehicles) {
      if (shouldUseNearbyLead(lead) && lowSpeedLeadBlocksMergeRolling(lead)) {
        return true;
      }
    }
    return false;
  }

  bool mergeKeepRollingActive() const {
    if (!merge_keep_rolling_enabled_) return false;
    if (merge_keep_rolling_min_speed_mps_ <= 1e-3) return false;
    if (!(laneChangeIntentActive() || laneEndPressureActive())) return false;
    if (shouldStopAtLaneEnd()) return false;
    if (mergeWaitStopRequired()) return false;
    return !mergeRollingFrontBlocked() && !mergeRollingLowSpeedLeadBlocked();
  }

  void applyMergeRollingFloor(double cruise_speed_mps,
                              double& raw_target,
                              std::string& active_lead,
                              bool& soft_target_front_limited_speed) const {
    if (!mergeKeepRollingActive()) return;
    const double rolling_floor =
        std::min(std::max(0.0, merge_keep_rolling_min_speed_mps_),
                 std::max(0.0, cruise_speed_mps));
    if (rolling_floor <= 1e-3 || raw_target >= rolling_floor) return;
    raw_target = rolling_floor;
    soft_target_front_limited_speed = false;
    active_lead = active_lead == "none"
        ? "merge_keep_rolling"
        : active_lead + ":merge_keep_rolling";
  }

  void timerCallback(const ros::TimerEvent& event) {
    if (!have_ego_speed_) return;

    const double cruise_speed_mps = currentCruiseSpeedMps();
    double raw_target = cruise_speed_mps;
    std::string active_lead = "none";
    bool hard_brake_active = false;
    double max_active_closing_speed_mps = 0.0;
    bool behavior_yield_slew_active = false;
    double behavior_yield_decel_limit_mps2 = 0.0;
    bool behavior_cap_applied = false;
    bool external_speed_limit_active = false;
    bool soft_target_front_limited_speed = false;
    double applied_behavior_cap_mps = std::numeric_limits<double>::quiet_NaN();
    LeadBinding lead_binding;

    const bool fresh_targets = have_targets_ &&
        (ros::Time::now() - targets_stamp_).toSec() <= target_timeout_sec_;
    const bool behavior_preparation_active =
        fresh_targets && behaviorPreparationActive();
    const bool endpoint_connector_active = endpointConnectorActive();
    const bool lane_sync_mismatch =
        fresh_targets && !targetsMatchActivePath();

    if (fresh_targets) {
      // current_front is the representative vehicle on the source/current
      // lane and remains an ACC lead throughout the lane change.
      applyLeadLimit(targets_.current_front, "current_front",
                     cruise_speed_mps, raw_target, active_lead, hard_brake_active,
                     max_active_closing_speed_mps, true, false,
                     &soft_target_front_limited_speed, &lead_binding);

      if ((laneChangeIntentActive() || behavior_preparation_active ||
           laneEndPressureActive()) &&
          targetFrontAccAllowed()) {
        const bool target_front_hard_acc = targetFrontHardAccAllowed();
        const bool target_front_soft_acc = !target_front_hard_acc;
        if (target_front_soft_acc && suppressTargetFrontSoftAccForRearIntrusion()) {
          ROS_WARN_THROTTLE(
              0.5,
              "[smpc_acc_speed_planner] holding merge momentum for target-lane rear "
              "intrusion; target_front soft ACC suppressed");
        } else {
          applyLeadLimit(targets_.target_front, "target_front",
                         cruise_speed_mps, raw_target, active_lead, hard_brake_active,
                         max_active_closing_speed_mps,
                         target_front_hard_acc, target_front_soft_acc,
                         &soft_target_front_limited_speed, &lead_binding);
        }
      }

      for (const auto& lead : targets_.nearby_vehicles) {
        if (!shouldUseNearbyLead(lead)) continue;
        const bool nearby_overlap =
            isNearbyOverlapVehicle(lead) && nearbyOverlapOccupiesEgoCorridor(lead);
        const bool target_lane_preentry =
            lead.lane_id == targets_.target_lane_id &&
            targets_.target_lane_id != targets_.current_lane_id &&
            !targetFrontHardAccAllowed();
        // A real bbox overlap is a collision hazard now, not merely a
        // target-lane pre-entry case.  It must retain normal hard/rapid brake
        // authority even before ego reaches the target-lane center.
        const bool allow_hard_brake = nearby_overlap || !target_lane_preentry;
        applyLeadLimit(lead, lead.role, cruise_speed_mps,
                       raw_target, active_lead, hard_brake_active,
                       max_active_closing_speed_mps,
                       allow_hard_brake, false,
                       &soft_target_front_limited_speed, &lead_binding);
      }

      // A lane endpoint is a merge deadline for lanes 0,1,2. If target-front
      // space is blocked, or SMPC explicitly reports WAIT_FOR_GAP, follow the
      // existing endpoint profile to the holding line.  WAIT_FOR_GAP keeps
      // normal ACC running upstream; this only starts in the urgent zone.
      if (!laneEndProtectionReleased() &&
          !lane_sync_mismatch &&
          targets_.lane_change_urgent &&
          mergeWaitStopRequired()) {
        const double endpoint_limit = mandatoryMergeLaneEndLimit();
        if (endpoint_limit < raw_target) {
          raw_target = endpoint_limit;
          soft_target_front_limited_speed = false;
          active_lead = targetFrontBlocksMerge()
              ? "lane_end_wait_front_gap"
              : "lane_end_wait_no_safe_gap";
          if (raw_target <= 0.01 && !mergeKeepRollingActive()) {
            hard_brake_active = true;
          }
        }
      }

      // Treat configured CSV endpoints as stop targets, independent of the
      // lane-change deadline logic.
      if (!laneEndProtectionReleased() && shouldStopAtLaneEnd() &&
          !lane_sync_mismatch &&
          (targets_.distance_to_lane_end <= lane_end_stop_prepare_distance_m_ ||
           targets_.time_to_lane_end <= lane_end_stop_prepare_time_sec_)) {
        const double endpoint_limit = laneEndSpeedLimit();
        if (endpoint_limit < raw_target) {
          raw_target = endpoint_limit;
          soft_target_front_limited_speed = false;
          active_lead = "lane_end_stop";
        }
      }

      if (!laneEndProtectionReleased() &&
          !lane_sync_mismatch &&
          targets_.emergency_stop_required &&
          mergeWaitStopRequired()) {
        const double endpoint_limit = mandatoryMergeLaneEndLimit();
        if (endpoint_limit < raw_target) {
          raw_target = endpoint_limit;
          soft_target_front_limited_speed = false;
          if (targetFrontBlocksMerge()) {
            active_lead = raw_target <= 0.01
                ? "lane_end_emergency_stop"
                : "lane_end_emergency_creep";
          } else {
            active_lead = raw_target <= 0.01
                ? "lane_end_wait_no_safe_gap:stop"
                : "lane_end_wait_no_safe_gap:creep";
          }
          if (raw_target <= 0.01 && !mergeKeepRollingActive()) {
            hard_brake_active = true;
          }
        }
      }

      if (!hard_brake_active) {
        applyMergeRollingFloor(cruise_speed_mps, raw_target, active_lead,
                                soft_target_front_limited_speed);
      }

      // A path switch can arrive one target-selector cycle before the new
      // lane projection.  Keep all physical lead/overlap/TTC limits above,
      // but never let the exhausted source lane's endpoint reset the speed
      // filter to zero during that short epoch mismatch.
      if (lane_sync_mismatch && active_lead == "none") {
        active_lead = "lane_sync_wait";
      }

      // Capture the non-behavior arbitration result before a preparation cap
      // changes its debug label.  This acknowledgement is fed back to SMPC so
      // it never mistakes a genuine lead/lane-end brake for its own YIELD.
      external_speed_limit_active =
          active_lead != "none" && active_lead != "merge_keep_rolling";

      // This is deliberately after the merge rolling floor.  A selected
      // YIELD/HOLD candidate must be able to lower speed even when ordinary
      // merge logic prefers to preserve momentum.  It can never relax an
      // existing lead, TTC, hard-brake, lane-end, or mission-speed limit.
    if (behavior_preparation_active &&
        behavior_request_.action !=
            smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP) {
        const double behavior_cap = std::clamp(
            behavior_request_.speed_cap_mps, min_speed_mps_, cruise_speed_mps);
        if (behavior_cap < raw_target) {
          behavior_cap_applied = true;
          applied_behavior_cap_mps = behavior_cap;
          // If this extra cap is the only source of deceleration, honor the
          // planner's requested YIELD rate.  Existing lead/lane-end/rapid
          // braking keeps its original (possibly stronger) safety authority.
          const bool no_external_speed_limit =
              active_lead == "none" || active_lead == "merge_keep_rolling";
          const bool requested_yield =
              behavior_request_.action ==
                  smpc_lane_change::BehaviorLongitudinalRequest::YIELD &&
              behavior_request_.requested_accel_mps2 < -1e-3;
          if (!hard_brake_active && no_external_speed_limit && requested_yield) {
            behavior_yield_slew_active = true;
            behavior_yield_decel_limit_mps2 =
                std::abs(behavior_request_.requested_accel_mps2);
          }
          raw_target = behavior_cap;
          soft_target_front_limited_speed = false;
          active_lead = active_lead == "none"
              ? behaviorActionName()
              : active_lead + ":" + behaviorActionName();
        }
      }
    }

    // This cap is applied after all ordinary ACC lead/TTC/lane-end arbitration
    // and its merge rolling floor.  It is only a minimum operation: a stricter
    // normal safety target (including a hard stop) always remains in force.
    // Do not retain it after the Bool stops being refreshed by the path node.
    if (endpoint_connector_active) {
      const double connector_cap = std::clamp(
          endpointConnectorSpeedCapMps(), min_speed_mps_, cruise_speed_mps);
      if (connector_cap < raw_target) {
        raw_target = connector_cap;
        soft_target_front_limited_speed = false;
        active_lead = active_lead == "none"
            ? "endpoint_connector"
            : active_lead + ":endpoint_connector";
      }
    }

    // A moving lane change that starts near the source CSV end must not reach
    // that end before the waypoint blend finishes.  The blend pairs source and
    // target points by index and appends unblended target points past the
    // source end, so arriving early leaves a (1 - alpha) * lane-offset step in
    // the reference (5.3 m/s at 29 m, accelerating: alpha ~0.8 -> ~0.7 m).
    // Cap speed so the remaining source length lasts the remaining change
    // time.  The stopped-launch connector keeps its own cap.
    // 차체가 이미 목표 차선에 들어온 뒤에는 이 상한을 푼다.  상한의 목적은 blend 가
    // 끝나기 전에 원래 차선 끝에 닿아 경로가 튀는 것을 막는 것인데, 횡방향 진행이
    // 사실상 끝난 뒤에는 막을 대상이 없다.  lc_gt 2026-09-17 백 3개: 목표차선 중심까지
    // 0.52~0.85 m 인 상태에서 남은 원래 차선 길이(2~5 m) 때문에 상한이 하한 3.0 m/s
    // 까지 떨어져, 8~10 m/s 로 달리던 자차가 브레이크 0.35~0.53 을 밟았다.  합류 후반
    // 1.5~2 s 동안 목표가 8~12 m/s 로 묶였다 (ACC 단독 A/B: 같은 입력에서 이 상한만
    // 끄면 목표가 20~23 m/s 로 유지되고 감속 지시가 사라진다).  0 이면 해제하지 않음.
    const bool lane_end_cap_released =
        lane_change_lane_end_cap_release_abs_d_m_ > 0.0 &&
        std::isfinite(targets_.ego_d_target) &&
        std::abs(targets_.ego_d_target) <=
            lane_change_lane_end_cap_release_abs_d_m_;
    if (lane_change_lane_end_cap_enabled_ && lane_change_active_ &&
        !lane_end_cap_released &&
        !endpoint_connector_active && fresh_targets &&
        std::isfinite(targets_.distance_to_lane_end)) {
      const double remaining_sec = std::max(
          0.5, lane_change_expected_duration_sec_ - lane_change_elapsed_sec_);
      const double usable_m = targets_.distance_to_lane_end -
          std::max(0.0, lane_change_lane_end_margin_m_);
      const double lane_end_cap = std::clamp(
          usable_m / remaining_sec,
          std::min(std::max(0.0, lane_change_lane_end_min_cap_mps_), cruise_speed_mps),
          cruise_speed_mps);
      if (lane_end_cap < raw_target) {
        raw_target = lane_end_cap;
        soft_target_front_limited_speed = false;
        active_lead = active_lead == "none"
            ? "lane_change_lane_end_cap"
            : active_lead + ":lane_change_lane_end_cap";
      }
    }

    double dt = (event.current_real - event.last_real).toSec();
    if (!(dt > 0.0 && dt < 1.0)) dt = 1.0 / std::max(1.0, publish_rate_hz_);

    const bool rear_escape_accel_active = fresh_targets &&
        activeRearEscapeAccelerationAllowed() &&
        raw_target > filtered_speed_mps_ + 1e-3;
    if (rear_escape_accel_active && active_lead == "none") {
      active_lead = "target_rear_escape_accel";
    }

    const double decel_limit =
        hard_brake_active ? std::max(max_decel_mps2_, hard_brake_decel_mps2_)
                          : closingAwareDecelLimit(max_decel_mps2_,
                                                    max_active_closing_speed_mps);
    double effective_decel_limit = decel_limit;
    if (behavior_yield_slew_active) {
      effective_decel_limit = std::min(
          effective_decel_limit, std::max(0.0, behavior_yield_decel_limit_mps2));
    }
    if (soft_target_front_limited_speed && !hard_brake_active) {
      effective_decel_limit = std::min(
          effective_decel_limit,
          std::max(0.0, target_front_overlap_soft_max_decel_mps2_));
    }
    // Ordinary lead following brakes only as hard as that lead requires.  Only
    // while the lead still sets raw_target: a lane-end, SMPC or connector cap
    // below it keeps the existing limits.  hard/rapid bypass the filter below.
    if (lead_required_decel_limit_enabled_ && !hard_brake_active &&
        lead_binding.valid &&
        std::abs(raw_target - lead_binding.target_mps) <= 1e-6 &&
        std::isfinite(lead_binding.required_decel_mps2)) {
      effective_decel_limit = std::min(
          effective_decel_limit,
          std::max(std::max(0.0, lead_required_decel_limit_min_mps2_),
                   std::max(0.0, lead_required_decel_limit_gain_) *
                       lead_binding.required_decel_mps2));
    }
    if (hard_brake_active && hard_brake_bypass_filter_) {
      filtered_speed_mps_ = raw_target;
    } else {
      // The endpoint connector is a short, prevalidated low-speed path.
      // Boost only its recovery slew; all ordinary ACC acceleration retains
      // max_accel_mps2_, while every front/TTC limit still lowers raw_target.
      double accel_limit = endpoint_connector_active
          ? std::max(0.0, endpoint_connector_max_accel_mps2_)
          : std::max(0.0, max_accel_mps2_);
      if (rear_escape_accel_active) {
        accel_limit = std::max(
            accel_limit, std::max(0.0, active_rear_escape_max_accel_mps2_));
      }
      const double upper = filtered_speed_mps_ + accel_limit * dt;
      const double lower = filtered_speed_mps_ - effective_decel_limit * dt;
      filtered_speed_mps_ = std::clamp(raw_target, lower, upper);
    }
    filtered_speed_mps_ = std::clamp(filtered_speed_mps_, min_speed_mps_, cruise_speed_mps);

    std_msgs::Float64 speed_msg;
    speed_msg.data = filtered_speed_mps_;
    speed_pub_.publish(speed_msg);

    std_msgs::Float64 raw_msg;
    raw_msg.data = raw_target;
    raw_speed_pub_.publish(raw_msg);

    std_msgs::String lead_msg;
    lead_msg.data = active_lead;
    active_lead_pub_.publish(lead_msg);

    smpc_lane_change::BehaviorLongitudinalStatus behavior_status;
    behavior_status.header.stamp = ros::Time::now();
    behavior_status.plan_id = behavior_preparation_active ? behavior_request_.plan_id : 0;
    behavior_status.command_revision =
        behavior_preparation_active ? behavior_request_.command_revision : 0;
    behavior_status.behavior_cap_applied = behavior_cap_applied;
    behavior_status.external_speed_limit_active = external_speed_limit_active;
    behavior_status.current_lane_id = fresh_targets ? targets_.current_lane_id : -1;
    behavior_status.target_lane_id = fresh_targets ? targets_.target_lane_id : -1;
    behavior_status.applied_speed_cap_mps = applied_behavior_cap_mps;
    behavior_status.final_target_speed_mps = filtered_speed_mps_;
    behavior_status.reason = active_lead;
    behavior_status_pub_.publish(behavior_status);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber targets_sub_, odom_sub_, mission_speed_sub_, status_sub_, decision_sub_;
  ros::Subscriber behavior_request_sub_, endpoint_connector_active_sub_, path_number_sub_;
  ros::Subscriber gps_state_sub_;
  ros::Publisher speed_pub_, raw_speed_pub_, active_lead_pub_, behavior_status_pub_;
  ros::Timer timer_;

  smpc_lane_change::TargetVehicleSet targets_;
  ros::Time targets_stamp_;
  bool have_targets_{false};
  bool have_ego_speed_{false};
  bool lane_change_active_{false};
  bool lane_change_requested_{false};
  bool have_mission_speed_target_{false};
  bool have_behavior_request_{false};
  bool have_endpoint_connector_status_{false};
  bool endpoint_connector_active_{false};
  bool have_active_path_number_{false};
  int active_path_number_{-1};
  double ego_speed_mps_{0.0};
  double filtered_speed_mps_{0.0};

  std::string mission_speed_topic_{"/control_feedback/mission_speed_target_mps"};
  double mission_speed_timeout_sec_{0.5};
  double mission_speed_fallback_mps_{10.0};
  double mission_speed_target_mps_{10.0};
  ros::Time mission_speed_stamp_;
  smpc_lane_change::BehaviorLongitudinalRequest behavior_request_;
  ros::Time behavior_request_stamp_;
  bool behavior_request_enabled_{false};
  std::string behavior_request_topic_{"/smpc/behavior_longitudinal_request"};
  double behavior_request_timeout_sec_{0.35};
  std::string behavior_status_topic_{"/smpc/behavior_longitudinal_status"};
  bool behavior_request_apply_while_lane_change_{false};
  bool behavior_request_apply_accel_{false};
  double endpoint_connector_speed_cap_mps_{2.0};
  std::vector<double> endpoint_connector_speed_cap_by_target_lane_mps_;
  double endpoint_connector_max_accel_mps2_{1.5};
  double endpoint_connector_status_timeout_sec_{0.5};
  bool lane_change_lane_end_cap_enabled_{false};
  double lane_change_lane_end_cap_release_abs_d_m_{0.0};
  bool lane_end_hold_until_lane_change_active_{false};
  double lane_end_hold_wait_for_gap_latch_sec_{1.0};
  ros::Time wait_for_gap_seen_stamp_{0.0};
  int wait_for_gap_current_lane_id_{-1};
  int wait_for_gap_target_lane_id_{-1};
  double lane_change_lane_end_margin_m_{3.0};
  double lane_change_lane_end_min_cap_mps_{3.0};
  ros::Time endpoint_connector_active_stamp_;
  double ego_length_m_{4.635};
  double ego_width_m_{1.892};
  double ego_front_extent_m_{3.845};
  double ego_rear_extent_m_{0.790};
  double standstill_gap_m_{8.0};
  double time_headway_sec_{2.2};
  double gap_gain_{0.65};
  double closing_speed_gap_time_sec_{0.8};
  double braking_decel_mps2_{4.5};
  double soft_ttc_sec_{4.0};
  double hard_ttc_sec_{1.8};
  double min_speed_mps_{0.0};
  double max_accel_mps2_{1.5};
  double max_decel_mps2_{5.0};
  bool closing_decel_boost_enabled_{true};
  double closing_decel_boost_start_mps_{5.0};
  double closing_decel_boost_full_mps_{15.0};
  double closing_decel_boost_max_decel_mps2_{8.0};
  double hard_brake_decel_mps2_{9.0};
  bool hard_brake_bypass_filter_{true};
  double emergency_gap_m_{8.0};
  double critical_gap_m_{2.5};
  double critical_ttc_sec_{0.7};
  bool rapid_lead_brake_enabled_{true};
  double rapid_lead_ttc_sec_{3.0};
  double rapid_lead_min_closing_speed_mps_{4.0};
  double rapid_lead_required_decel_mps2_{5.0};
  double max_follow_speed_drop_mps_{5.0};
  bool lead_required_decel_limit_enabled_{false};
  double lead_required_decel_limit_gain_{1.5};
  double lead_required_decel_limit_min_mps2_{1.5};
  double lead_required_decel_limit_headway_sec_{1.5};
  double target_timeout_sec_{0.8};
  double decision_request_timeout_sec_{0.5};
  double publish_rate_hz_{20.0};
  ros::Time decision_stamp_;
  int requested_decision_current_lane_id_{-1};
  int requested_decision_target_lane_id_{-1};
  double min_target_front_gap_m_{11.0};
  double min_target_rear_gap_m_{6.0};
  bool target_front_acc_requires_target_lane_entry_{true};
  double target_front_acc_entry_abs_d_m_{0.75};
  bool target_front_preentry_soft_acc_enabled_{false};
  double target_front_preentry_soft_acc_distance_to_end_m_{120.0};
  bool target_front_overlap_soft_acc_enabled_{false};
  double target_front_overlap_soft_acc_start_abs_d_m_{2.7};
  double target_front_overlap_soft_max_decel_mps2_{1.5};
  bool active_rear_guard_enabled_{false};
  double active_rear_guard_min_gap_m_{9.0};
  double active_rear_guard_min_ttc_sec_{3.0};
  double active_rear_guard_min_headway_sec_{1.2};
  double active_rear_guard_abort_before_target_center_abs_d_m_{1.75};
  double active_rear_guard_perception_control_delay_sec_{0.50};
  double active_rear_guard_stationary_remaining_sec_{8.0};
  double active_rear_escape_max_accel_mps2_{2.0};
  double active_initial_target_abs_d_{3.5};
  double lane_change_elapsed_sec_{0.0};
  double lane_change_expected_duration_sec_{4.0};
  bool merge_priority_enabled_{true};
  double merge_priority_rear_safe_gap_m_{4.5};
  double merge_priority_rear_min_gap_m_{4.0};
  double merge_priority_rear_max_closing_mps_{3.5};
  double merge_priority_rear_min_ttc_sec_{1.8};
  bool ego_lane_priority_enabled_{true};
  double ego_lane_priority_current_max_abs_d_m_{1.25};
  double ego_lane_priority_front_hard_gap_m_{8.0};
  double lane_end_stop_margin_m_{2.0};
  double lane_end_comfort_decel_mps2_{2.5};
  double lane_end_merge_creep_speed_mps_{4.0};
  double lane_end_hard_stop_distance_m_{6.0};
  bool lane_end_merge_wait_stop_enabled_{true};
  double lane_end_merge_wait_stop_distance_m_{35.0};
  bool stop_at_lane_end_enabled_{true};
  std::vector<int> lane_end_stop_lane_ids_{3};
  double lane_end_stop_prepare_distance_m_{45.0};
  double lane_end_stop_prepare_time_sec_{4.0};
  bool use_nearby_vehicles_{true};
  bool use_nearby_overlap_as_acc_lead_{false};
  bool nearby_acc_overlap_only_{false};
  double nearby_overlap_lateral_margin_m_{0.20};
  bool lead_requires_center_ahead_{false};
  int immature_lead_min_hits_{0};
  double immature_lead_min_speed_mps_{2.0};
  double nearby_acc_lead_max_abs_d_m_{1.35};
  double nearby_merge_front_range_m_{120.0};
  double nearby_merge_rear_range_m_{20.0};
  bool use_all_nearby_when_lane_end_pressure_{true};
  bool lane_end_wait_gap_front_only_{true};
  bool merge_keep_rolling_enabled_{true};
  double merge_keep_rolling_min_speed_mps_{6.0};
  double merge_keep_rolling_front_block_gap_m_{8.0};
  double merge_keep_rolling_low_speed_lead_mps_{3.0};
  double merge_keep_rolling_low_speed_lead_max_gap_m_{80.0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_acc_speed_planner");
  AccSpeedPlanner node;
  ros::spin();
  return 0;
}
