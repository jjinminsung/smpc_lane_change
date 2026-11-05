#include <algorithm>
#include <array>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <geometry_msgs/Point.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <smpc_lane_change/BehaviorLongitudinalRequest.h>
#include <smpc_lane_change/BehaviorLongitudinalStatus.h>
#include <smpc_lane_change/BehaviorPlan.h>
#include <smpc_lane_change/LaneChangeDecision.h>
#include <smpc_lane_change/LaneChangeStatus.h>
#include <smpc_lane_change/TargetVehicle.h>
#include <smpc_lane_change/TargetVehicleSet.h>
#include <smpc_lane_change/lane_path.hpp>
#include <waypoint_system/PathSwitchPreview.h>

namespace fs = std::filesystem;

// SMPC-inspired lane-change decision node v2.
//
// This is still competition-practical rather than a full Gurobi/SOCP
// implementation of the paper.  The important upgrade from v1 is that the node
// no longer evaluates only scalar gap evolution.  It now:
//   1) loads the same lane CSVs as target_selector,
//   2) rolls out ego KEEP and CHANGE_LEFT candidate trajectories in map/UTM,
//   3) predicts target vehicles in Frenet s,d with multiple modes,
//   4) evaluates bbox collision probability across the whole horizon, and
//   5) publishes the lower-cost safe decision.
class SmpcDecisionNode {
 public:
  SmpcDecisionNode() : nh_(), pnh_("~") {
    pnh_.param("enabled", enabled_, true);
    pnh_.param("publish_rate_hz", publish_rate_hz_, 20.0);
    pnh_.param("target_timeout_sec", target_timeout_sec_, 0.5);
    pnh_.param<std::string>("waypoint_directory", waypoint_directory_, "");

    pnh_.param("horizon_steps", horizon_steps_, 40);
    pnh_.param("prediction_dt_sec", prediction_dt_sec_, 0.2);
    pnh_.param("risk_epsilon", risk_epsilon_, 0.04);
    pnh_.param("urgent_risk_epsilon", urgent_risk_epsilon_, 0.15);
    // Keep /smpc/targets.nearby_vehicles available for RViz/diagnostics, but
    // allow A/B testing whether they participate in behavior-risk evaluation.
    pnh_.param("risk_include_nearby_vehicles", risk_include_nearby_vehicles_, true);

    pnh_.param("ego_length_m", ego_length_m_, 4.635);
    pnh_.param("ego_width_m", ego_width_m_, 1.892);
    // /odom is the rear-axle pose, not the geometric centre.
    pnh_.param("ego_front_extent_m", ego_front_extent_m_, 3.845);
    pnh_.param("ego_rear_extent_m", ego_rear_extent_m_, 0.790);
    ego_length_m_ = ego_front_extent_m_ + ego_rear_extent_m_;
    pnh_.param<std::string>("mission_speed_topic",
                            mission_speed_topic_,
                            "/control_feedback/mission_speed_target_mps");
    pnh_.param("mission_speed_timeout_sec", mission_speed_timeout_sec_, 0.5);
    pnh_.param("mission_speed_fallback_mps", mission_speed_fallback_mps_, 10.0);
    pnh_.param("ego_prediction_use_acc_target_speed",
               ego_prediction_use_acc_target_speed_, true);
    pnh_.param<std::string>("ego_prediction_acc_target_speed_topic",
                            ego_prediction_acc_target_speed_topic_,
                            "/smpc/target_speed_mps");
    pnh_.param("ego_prediction_acc_target_speed_timeout_sec",
               ego_prediction_acc_target_speed_timeout_sec_, 0.5);
    pnh_.param("standstill_gap_m", standstill_gap_m_, 5.0);
    pnh_.param("time_headway_sec", time_headway_sec_, 1.0);
    pnh_.param("current_front_trigger_gap_m", current_front_trigger_gap_m_, 35.0);
    pnh_.param("current_front_safe_gap_m", current_front_safe_gap_m_, 7.0);
    pnh_.param("target_front_safe_gap_m", target_front_safe_gap_m_, 14.0);
    pnh_.param("target_rear_safe_gap_m", target_rear_safe_gap_m_, 9.0);
    pnh_.param("prepare_merge_front_safe_gap_m",
               prepare_merge_front_safe_gap_m_,
               12.0);
    pnh_.param("prepare_merge_rear_safe_gap_m",
               prepare_merge_rear_safe_gap_m_,
               7.0);
    pnh_.param("target_front_min_ttc_sec", target_front_min_ttc_sec_, 3.0);
    pnh_.param("target_front_min_headway_sec", target_front_min_headway_sec_, 1.2);
    pnh_.param("target_front_headway_use_closing_speed",
               target_front_headway_use_closing_speed_, false);
    pnh_.param("pass_gap_headway_use_closing_speed",
               pass_gap_headway_use_closing_speed_, false);
    pnh_.param("change_rollout_post_completion_use_pre_cap_floor",
               change_rollout_post_completion_use_pre_cap_floor_, true);
    pnh_.param("change_rollout_blend_speed_cap_enabled",
               change_rollout_blend_speed_cap_enabled_, false);
    pnh_.param("change_rollout_blend_speed_interp_enabled",
               change_rollout_blend_speed_interp_enabled_, false);
    pnh_.param("change_rollout_blend_stage_enabled",
               change_rollout_blend_stage_enabled_, false);
    pnh_.param("change_rollout_soft_acc_start_abs_d_m",
               change_rollout_soft_acc_start_abs_d_m_, 2.7);
    pnh_.param("change_rollout_launch_speed_threshold_mps",
               change_rollout_launch_speed_threshold_mps_, 1.0);
    pnh_.param("change_rollout_blend_cap_decel_mps2",
               change_rollout_blend_cap_decel_mps2_, 4.0);
    pnh_.param("target_rear_min_ttc_sec", target_rear_min_ttc_sec_, 3.0);
    pnh_.param("target_rear_min_headway_sec", target_rear_min_headway_sec_, 1.2);
    pnh_.param("change_commit_scene_observation_sec",
               change_commit_scene_observation_sec_, 1.5);
    pnh_.param("change_commit_vehicle_observation_sec",
               change_commit_vehicle_observation_sec_, 0.6);
    pnh_.param("change_commit_observation_gap_timeout_sec",
               change_commit_observation_gap_timeout_sec_, 0.35);
    pnh_.param("change_commit_acc_stable_sec",
               change_commit_acc_stable_sec_, 0.50);
    pnh_.param("change_commit_acc_speed_error_mps",
               change_commit_acc_speed_error_mps_, 1.50);
    pnh_.param("change_commit_acc_max_decel_mps2",
               change_commit_acc_max_decel_mps2_, 1.00);
    pnh_.param("change_commit_allow_lane_end_wait_decel",
               change_commit_allow_lane_end_wait_decel_, false);
    pnh_.param("vanished_target_hold_enabled", vanished_target_hold_enabled_, false);
    pnh_.param("vanished_target_hold_sec", vanished_target_hold_sec_, 0.50);
    pnh_.param("vanished_target_hold_min_hits", vanished_target_hold_min_hits_, 20);
    pnh_.param("vanished_target_hold_zone_rear_m",
               vanished_target_hold_zone_rear_m_, 60.0);
    pnh_.param("vanished_target_hold_zone_front_m",
               vanished_target_hold_zone_front_m_, 30.0);
    pnh_.param("vanished_target_hold_margin_m", vanished_target_hold_margin_m_, 1.5);
    pnh_.param("vanished_target_hold_alongside_sec",
               vanished_target_hold_alongside_sec_, 0.0);
    pnh_.param("vanished_target_hold_alongside_ds_m",
               vanished_target_hold_alongside_ds_m_, 4.0);
    pnh_.param("pass_gap_enabled", pass_gap_enabled_, true);
    pnh_.param("pass_gap_min_relative_speed_mps",
               pass_gap_min_relative_speed_mps_, 2.0);
    pnh_.param("pass_gap_front_max_distance_m",
               pass_gap_front_max_distance_m_, 40.0);
    pnh_.param("pass_gap_entry_front_clearance_m",
               pass_gap_entry_front_clearance_m_, 2.0);
    pnh_.param("pass_gap_entry_relative_speed_time_sec",
               pass_gap_entry_relative_speed_time_sec_, 0.0);
    pnh_.param("pass_gap_entry_length_margin_m",
               pass_gap_entry_length_margin_m_, 0.0);
    pnh_.param("pass_gap_lane_pair_observation_sec",
               pass_gap_lane_pair_observation_sec_, 1.5);
    pnh_.param("rear_gap_perception_control_delay_sec",
               rear_gap_perception_control_delay_sec_, 0.50);
    pnh_.param("new_rear_track_guard_sec", new_rear_track_guard_sec_, 0.80);
    pnh_.param("new_rear_track_max_gap_m", new_rear_track_max_gap_m_, 40.0);
    pnh_.param("new_rear_track_closing_upper_mps",
               new_rear_track_closing_upper_mps_, 15.0);
    pnh_.param("tv_lane_projection_extension_before_m",
               tv_lane_projection_extension_before_m_, 100.0);
    pnh_.param("observed_cutin_safety_enabled", observed_cutin_safety_enabled_, true);
    pnh_.param("observed_cutin_assume_no_lane_change",
               observed_cutin_assume_no_lane_change_, false);
    pnh_.param("observed_cutin_min_lateral_speed_mps",
               observed_cutin_min_lateral_speed_mps_, 0.5);
    pnh_.param("observed_cutin_max_entry_time_sec",
               observed_cutin_max_entry_time_sec_, 4.5);
    pnh_.param("observed_cutin_max_rear_distance_m",
               observed_cutin_max_rear_distance_m_, 60.0);
    pnh_.param("urgent_merge_front_safe_gap_m", urgent_merge_front_safe_gap_m_, 9.0);
    pnh_.param("urgent_merge_rear_safe_gap_m", urgent_merge_rear_safe_gap_m_, 5.5);
    pnh_.param("emergency_merge_front_safe_gap_m", emergency_merge_front_safe_gap_m_, 6.5);
    pnh_.param("emergency_merge_rear_safe_gap_m", emergency_merge_rear_safe_gap_m_, 4.5);
    pnh_.param("emergency_merge_risk_epsilon", emergency_merge_risk_epsilon_, 0.25);
    pnh_.param("merge_priority_enabled", merge_priority_enabled_, true);
    pnh_.param("merge_priority_rear_safe_gap_m", merge_priority_rear_safe_gap_m_, 4.5);
    pnh_.param("merge_priority_rear_min_gap_m", merge_priority_rear_min_gap_m_, 4.0);
    pnh_.param("merge_priority_rear_max_closing_mps",
               merge_priority_rear_max_closing_mps_, 3.5);
    pnh_.param("merge_priority_rear_min_ttc_sec",
               merge_priority_rear_min_ttc_sec_, 1.8);
    pnh_.param("merge_priority_rear_risk_scale", merge_priority_rear_risk_scale_, 0.10);
    pnh_.param("merge_priority_nearby_rear_risk_scale",
               merge_priority_nearby_rear_risk_scale_, 0.05);
    pnh_.param("low_speed_rear_safety_enabled", low_speed_rear_safety_enabled_, true);
    pnh_.param("low_speed_rear_ego_speed_mps", low_speed_rear_ego_speed_mps_, 2.0);
    pnh_.param("low_speed_rear_safe_gap_m", low_speed_rear_safe_gap_m_, 10.0);
    pnh_.param("low_speed_rear_min_ttc_sec", low_speed_rear_min_ttc_sec_, 3.0);
    pnh_.param("ego_lane_priority_enabled", ego_lane_priority_enabled_, true);
    pnh_.param("ego_lane_priority_current_max_abs_d_m",
               ego_lane_priority_current_max_abs_d_m_, 1.25);
    pnh_.param("ego_lane_priority_front_hard_gap_m",
               ego_lane_priority_front_hard_gap_m_, 8.0);
    pnh_.param("ego_lane_priority_side_risk_scale",
               ego_lane_priority_side_risk_scale_, 0.08);

    pnh_.param("lane_width_m", lane_width_m_, 3.5);
    pnh_.param("lane_change_duration_sec", lane_change_duration_sec_, 2.5);
    pnh_.param("use_waypoint_linear_blend", use_waypoint_linear_blend_, true);
    pnh_.param("waypoint_blend_steps", waypoint_blend_steps_, 17);
    pnh_.param("waypoint_blend_rate_hz", waypoint_blend_rate_hz_, 25.0);
    pnh_.param("waypoint_blend_duration_sec", waypoint_blend_duration_sec_, 0.0);
    pnh_.param("lane_center_decay_tau_sec", lane_center_decay_tau_sec_, 1.2);

    // waypoint_system is the owner of the reference that the low-level MPC
    // will receive.  For a lane-change candidate, consume its preview instead
    // of independently blending two CSV lane centre-lines in this node.
    pnh_.param("ego_reference_preview_enabled", ego_reference_preview_enabled_, true);
    pnh_.param("ego_reference_preview_required_for_change",
               ego_reference_preview_required_for_change_, true);
    pnh_.param<std::string>("ego_reference_preview_topic",
                            ego_reference_preview_topic_, "/path_switch_preview");
    pnh_.param("ego_reference_preview_timeout_sec",
               ego_reference_preview_timeout_sec_, 0.50);
    pnh_.param("ego_reference_preview_duration_scale",
               ego_reference_preview_duration_scale_, 1.0);
    pnh_.param("ego_reference_preview_motion_start_speed_mps",
               ego_reference_preview_motion_start_speed_mps_, 0.20);
    pnh_.param("ego_reference_preview_motion_start_distance_m",
               ego_reference_preview_motion_start_distance_m_, 0.30);
    pnh_.param("stopped_launch_connector_speed_cap_mps",
               stopped_launch_connector_speed_cap_mps_, 2.0);
    // Optional per-target-lane override, indexed by target lane id.  Must be
    // identical to ACC's endpoint_connector_speed_cap_by_target_lane_mps.
    pnh_.getParam("stopped_launch_connector_speed_cap_by_target_lane_mps",
                  stopped_launch_connector_speed_cap_by_target_lane_mps_);
    pnh_.param("stopped_launch_connector_max_accel_mps2",
               stopped_launch_connector_max_accel_mps2_, 1.5);
    pnh_.param("stopped_launch_connector_max_preswitch_distance_m",
               stopped_launch_connector_max_preswitch_distance_m_, 0.50);
    // A change candidate is not safe merely because its frozen source-lane
    // trajectory has low risk. It must reach the target lane within the
    // evaluated horizon. This rejects a stopped normal-BLEND whose clock has
    // not started; stopped endpoint changes instead use CONNECTOR preview.
    pnh_.param("ego_reference_preview_require_target_completion",
               ego_reference_preview_require_target_completion_, true);
    pnh_.param("ego_reference_preview_target_completion_max_abs_d_m",
               ego_reference_preview_target_completion_max_abs_d_m_, 0.75);
    // preview 폴리라인 끝을 넘어선 horizon 스텝에서 궤적 전체를 버리지 않고
    // 마지막 유효 스텝까지 잘라 쓴다.  false 로 두면 기존(전체 폐기) 동작.
    pnh_.param("ego_reference_preview_allow_truncation",
               ego_reference_preview_allow_truncation_, true);
    pnh_.param("change_commit_scene_rear_max_range_m",
               change_commit_scene_rear_max_range_m_, 0.0);
    pnh_.param("pass_gap_skip_ramp_while_receding",
               pass_gap_skip_ramp_while_receding_, false);
    pnh_.param("ego_brake_reaction_enabled",
               ego_brake_reaction_enabled_, false);
    pnh_.param("ego_brake_reaction_delay_sec",
               ego_brake_reaction_delay_sec_, 0.5);
    pnh_.param("ego_brake_reaction_max_decel_mps2",
               ego_brake_reaction_max_decel_mps2_, 2.5);
    ego_reference_preview_timeout_sec_ =
        std::max(0.0, ego_reference_preview_timeout_sec_);
    ego_reference_preview_motion_start_speed_mps_ =
        std::max(0.0, ego_reference_preview_motion_start_speed_mps_);
    ego_reference_preview_motion_start_distance_m_ =
        std::max(0.0, ego_reference_preview_motion_start_distance_m_);
    stopped_launch_connector_speed_cap_mps_ =
        std::max(0.0, stopped_launch_connector_speed_cap_mps_);
    stopped_launch_connector_max_accel_mps2_ =
        std::max(0.0, stopped_launch_connector_max_accel_mps2_);
    stopped_launch_connector_max_preswitch_distance_m_ =
        std::max(0.0, stopped_launch_connector_max_preswitch_distance_m_);

    // High-level preparation candidates.  These are evaluated in the
    // decision node, while the low-level longitudinal command remains owned
    // by ACC through BehaviorLongitudinalRequest.
    pnh_.param("behavior_planner_enabled", behavior_planner_enabled_, true);
    pnh_.param("behavior_execute_decel_hold", behavior_execute_decel_hold_, false);
    pnh_.param("behavior_execute_accel", behavior_execute_accel_, false);
    pnh_.param("behavior_require_legacy_risk_gate",
               behavior_require_legacy_risk_gate_, false);
    pnh_.param("behavior_min_signed_clearance_m",
               behavior_min_signed_clearance_m_, 0.0);
    pnh_.param("behavior_execution_delay_sec", behavior_execution_delay_sec_, 0.20);
    pnh_.param("behavior_post_merge_buffer_sec",
               behavior_post_merge_buffer_sec_, 2.0);
    pnh_.param("behavior_decel_accel_mps2", behavior_decel_accel_mps2_, -1.0);
    pnh_.param("behavior_yield_comfort_max_decel_mps2",
               behavior_yield_comfort_max_decel_mps2_, 0.8);
    pnh_.param("behavior_yield_max_speed_drop_mps",
               behavior_yield_max_speed_drop_mps_, 3.0);
    pnh_.param("behavior_yield_confirm_sec", behavior_yield_confirm_sec_, 0.6);
    pnh_.param("behavior_hold_accel_mps2", behavior_hold_accel_mps2_, 0.0);
    pnh_.param("behavior_accel_accel_mps2", behavior_accel_accel_mps2_, 1.0);
    pnh_.param("behavior_rear_accel_min_closing_mps",
               behavior_rear_accel_min_closing_mps_, 1.0);
    pnh_.param("behavior_rear_accel_gain",
               behavior_rear_accel_gain_, 0.25);
    pnh_.param("behavior_rear_accel_min_mps2",
               behavior_rear_accel_min_mps2_, 0.40);
    pnh_.param("behavior_rear_accel_emergency_ttc_sec",
               behavior_rear_accel_emergency_ttc_sec_, 2.0);
    pnh_.param("behavior_delay_cost_weight", behavior_delay_cost_weight_, 1.0);
    pnh_.param("behavior_accel_cost_weight", behavior_accel_cost_weight_, 0.25);
    pnh_.param<std::string>("behavior_status_topic", behavior_status_topic_,
                            "/smpc/behavior_longitudinal_status");
    pnh_.param("behavior_status_timeout_sec", behavior_status_timeout_sec_, 0.35);
    if (!pnh_.getParam("behavior_yield_decel_candidates_mps2",
                       behavior_yield_decel_candidates_mps2_)) {
      behavior_yield_decel_candidates_mps2_ = {behavior_decel_accel_mps2_};
    }
    behavior_yield_comfort_max_decel_mps2_ =
        std::max(0.05, behavior_yield_comfort_max_decel_mps2_);
    behavior_yield_max_speed_drop_mps_ =
        std::max(0.0, behavior_yield_max_speed_drop_mps_);
    behavior_yield_confirm_sec_ = std::max(0.0, behavior_yield_confirm_sec_);
    behavior_yield_decel_candidates_mps2_.erase(
        std::remove_if(behavior_yield_decel_candidates_mps2_.begin(),
                       behavior_yield_decel_candidates_mps2_.end(),
                       [](double a) { return !std::isfinite(a) || a >= -1e-3; }),
        behavior_yield_decel_candidates_mps2_.end());
    for (double& accel : behavior_yield_decel_candidates_mps2_) {
      accel = std::max(accel, -behavior_yield_comfort_max_decel_mps2_);
    }
    std::sort(behavior_yield_decel_candidates_mps2_.begin(),
              behavior_yield_decel_candidates_mps2_.end(), std::greater<double>());
    behavior_yield_decel_candidates_mps2_.erase(
        std::unique(behavior_yield_decel_candidates_mps2_.begin(),
                    behavior_yield_decel_candidates_mps2_.end()),
        behavior_yield_decel_candidates_mps2_.end());
    if (behavior_yield_decel_candidates_mps2_.empty()) {
      behavior_yield_decel_candidates_mps2_ = {
          -behavior_yield_comfort_max_decel_mps2_};
    }
    if (!pnh_.getParam("behavior_lc_start_times_sec", behavior_lc_start_times_sec_)) {
      behavior_lc_start_times_sec_ = {0.4, 0.8, 1.2, 1.6, 2.0};
    }
    behavior_lc_start_times_sec_.erase(
        std::remove_if(behavior_lc_start_times_sec_.begin(),
                       behavior_lc_start_times_sec_.end(),
                       [](double t) { return !std::isfinite(t) || t <= 0.0; }),
        behavior_lc_start_times_sec_.end());
    std::sort(behavior_lc_start_times_sec_.begin(), behavior_lc_start_times_sec_.end());
    behavior_lc_start_times_sec_.erase(
        std::unique(behavior_lc_start_times_sec_.begin(),
                    behavior_lc_start_times_sec_.end()),
        behavior_lc_start_times_sec_.end());

    pnh_.param("front_brake_probability", front_brake_probability_, 0.25);
    pnh_.param("rear_accel_probability", rear_accel_probability_, 0.25);
    pnh_.param("min_front_brake_probability", min_front_brake_probability_, 0.05);
    pnh_.param("max_front_brake_probability", max_front_brake_probability_, 0.65);
    pnh_.param("min_rear_accel_probability", min_rear_accel_probability_, 0.05);
    pnh_.param("max_rear_accel_probability", max_rear_accel_probability_, 0.65);
    pnh_.param("front_brake_accel_mps2", front_brake_accel_mps2_, -2.0);
    pnh_.param("rear_accel_mps2", rear_accel_mps2_, 1.5);
    pnh_.param("rear_accel_max_speed_enabled",
               rear_accel_max_speed_enabled_, false);
    pnh_.param("rear_accel_max_speed_mps", rear_accel_max_speed_mps_, -1.0);
    pnh_.param("rear_accel_max_speed_margin_mps",
               rear_accel_max_speed_margin_mps_, 0.0);
    pnh_.param("rear_accel_probability_by_accel_enabled",
               rear_accel_probability_by_accel_enabled_, false);
    pnh_.getParam("rear_accel_probability_accel_breaks_mps2",
                  rear_accel_probability_accel_breaks_mps2_);
    pnh_.getParam("rear_accel_probability_by_accel",
                  rear_accel_probability_by_accel_);
    pnh_.param("nominal_target_accel_mps2", nominal_target_accel_mps2_, 0.0);
    pnh_.param("ego_assumed_accel_mps2", ego_assumed_accel_mps2_, 0.0);
    pnh_.param("ego_prediction_use_mission_speed_rollout",
               ego_prediction_use_mission_speed_rollout_,
               true);
    pnh_.param("ego_prediction_max_accel_mps2", ego_prediction_max_accel_mps2_, 2.5);
    pnh_.param("ego_prediction_max_decel_mps2", ego_prediction_max_decel_mps2_, 2.0);
    pnh_.param("change_rollout_target_lane_speed_enabled",
               change_rollout_target_lane_speed_enabled_, false);
    pnh_.param("change_rollout_release_fraction",
               change_rollout_release_fraction_, 1.0);
    pnh_.param("probability_ttc_safe_sec", probability_ttc_safe_sec_, 5.0);
    pnh_.param("probability_ttc_critical_sec", probability_ttc_critical_sec_, 1.5);
    pnh_.param("probability_rel_speed_scale_mps", probability_rel_speed_scale_mps_, 8.0);
    pnh_.param("probability_accel_scale_mps2", probability_accel_scale_mps2_, 3.0);
    pnh_.param("observed_accel_alpha", observed_accel_alpha_, 0.35);
    pnh_.param("max_observed_accel_mps2", max_observed_accel_mps2_, 6.0);
    // 트리 플래너 (research). 기본 꺼짐 = 기존 SMPC 그대로.
    pnh_.param("tree_planner_enabled", tree_planner_enabled_, false);
    pnh_.param("tree_keep_gates", tree_keep_gates_, true);
    pnh_.param("tree_monitor_enabled", tree_monitor_enabled_, true);
    // 2×2 비교용, 기존 판단 경로(tree_planner_enabled=false)에서만 쓴다. 기본값 = 기존 SMPC 그대로.
    // baseline_keep_gates=false: G3/G5 와 모든-mode clearance 를 걷어낸다(시뮬레이터 baseline_gt_nogate).
    // baseline_tree_monitor=true: 차선 변경 중 트리 모니터를 돌려 /smpc/tree_abort 를 낸다(tree_gt_nogate_bdec).
    pnh_.param("baseline_keep_gates", baseline_keep_gates_, true);
    pnh_.param("baseline_tree_monitor", baseline_tree_monitor_, false);
    pnh_.getParam("tree_onsets_sec", tree_onsets_sec_);
    pnh_.param("tree_rear_detect_delay_sec", tree_rear_detect_delay_sec_, 0.5);
    pnh_.param("tree_front_detect_delay_sec", tree_front_detect_delay_sec_, 0.3);
    pnh_.param("tree_escape_accel_mps2", tree_escape_accel_mps2_, 1.5);
    pnh_.param("tree_return_time_full_sec", tree_return_time_full_sec_, 3.1);
    pnh_.param("tree_return_decel_mps2", tree_return_decel_mps2_, -1.5);
    pnh_.param("tree_brake_decel_mps2", tree_brake_decel_mps2_, -3.0);
    pnh_.param("tree_abort_before_abs_d_m", tree_abort_before_abs_d_m_, 1.75);
    pnh_.getParam("tree_response_cost", tree_response_cost_);
    if (tree_response_cost_.size() != 5) tree_response_cost_ = {0.0, 1.0, 30.0, 35.0, 5.0};
    // research/docs/plan_v2.md 반응형 트리
    pnh_.param("reactive_tree_enabled", reactive_tree_enabled_, false);
    pnh_.param("reactive_drop_common", reactive_drop_common_, true);
    pnh_.param("reactive_single", reactive_single_, false);
    pnh_.param("reactive/connector_cap", rx_.connector_cap, rx_.connector_cap);
    pnh_.param("reactive/monitor_tree", rx_.monitor_tree, rx_.monitor_tree);
    pnh_.param("reactive/passed_front_as_rear", rx_.passed_front_as_rear, rx_.passed_front_as_rear);
    pnh_.param("reactive/merge_floor", rx_.merge_floor, rx_.merge_floor);
    pnh_.param("reactive/floor_urgent_distance_m", rx_.floor_urgent_distance_m, rx_.floor_urgent_distance_m);
    pnh_.param("reactive/floor_urgent_time_s", rx_.floor_urgent_time_s, rx_.floor_urgent_time_s);
    pnh_.param("reactive/plant_tau", rx_.plant_tau, rx_.plant_tau);
    pnh_.param("reactive/plant_a_max", rx_.plant_a_max, rx_.plant_a_max);
    pnh_.param("reactive/plant_a_min", rx_.plant_a_min, rx_.plant_a_min);
    pnh_.param("reactive/plant_a_stop", rx_.plant_a_stop, rx_.plant_a_stop);
    pnh_.param("reactive/plant_jerk", rx_.plant_jerk, rx_.plant_jerk);
    pnh_.param("reactive/acc_up", rx_.acc_up, rx_.acc_up);
    pnh_.param("reactive/acc_dn", rx_.acc_dn, rx_.acc_dn);
    pnh_.param("reactive/mission_ramp", rx_.mission_ramp, rx_.mission_ramp);
    pnh_.param("reactive/cruise_max", rx_.cruise_max, rx_.cruise_max);
    pnh_.param("reactive/lane_end_decel", rx_.lane_end_decel, rx_.lane_end_decel);
    pnh_.param("reactive/rear_react_intrusion_m", rx_.rear_react_intrusion_m, rx_.rear_react_intrusion_m);
    pnh_.param("reactive/rear_react_delay_s", rx_.rear_react_delay_s, rx_.rear_react_delay_s);
    pnh_.param("reactive/rear_detect_s", rx_.rear_detect_s, rx_.rear_detect_s);
    pnh_.param("reactive/rear_idm_s0", rx_.rear_idm_s0, rx_.rear_idm_s0);
    pnh_.param("reactive/rear_idm_T", rx_.rear_idm_T, rx_.rear_idm_T);
    pnh_.param("reactive/rear_idm_a", rx_.rear_idm_a, rx_.rear_idm_a);
    pnh_.param("reactive/rear_idm_b", rx_.rear_idm_b, rx_.rear_idm_b);
    pnh_.param("reactive/rear_max_brake", rx_.rear_max_brake, rx_.rear_max_brake);
    pnh_.param("reactive/front_brake", rx_.front_brake, rx_.front_brake);
    pnh_.param("reactive/front_accel", rx_.front_accel, rx_.front_accel);
    pnh_.param("reactive/front_detect_s", rx_.front_detect_s, rx_.front_detect_s);
    pnh_.param("reactive/accel_decay_s", rx_.accel_decay_s, rx_.accel_decay_s);
    pnh_.param("reactive/front_idm_s0", rx_.front_idm_s0, rx_.front_idm_s0);
    pnh_.param("reactive/front_idm_T", rx_.front_idm_T, rx_.front_idm_T);
    pnh_.param("reactive/front_idm_a", rx_.front_idm_a, rx_.front_idm_a);
    pnh_.param("reactive/front_idm_b", rx_.front_idm_b, rx_.front_idm_b);
    pnh_.param("reactive/p_exc_floor", rx_.p_exc_floor, rx_.p_exc_floor);
    pnh_.param("reactive/follow_sigma_cap_s", rx_.follow_sigma_cap_s, rx_.follow_sigma_cap_s);
    pnh_.param("reactive/abort_before_abs_d", rx_.abort_before_abs_d, rx_.abort_before_abs_d);
    pnh_.param("acc/standstill_gap_m", rx_acc_.standstill_gap_m, rx_acc_.standstill_gap_m);
    pnh_.param("acc/time_headway_sec", rx_acc_.time_headway_sec, rx_acc_.time_headway_sec);
    pnh_.param("acc/closing_speed_gap_time_sec", rx_acc_.closing_speed_gap_time_sec, rx_acc_.closing_speed_gap_time_sec);
    pnh_.param("acc/gap_gain", rx_acc_.gap_gain, rx_acc_.gap_gain);
    pnh_.param("acc/max_follow_speed_drop_mps", rx_acc_.max_follow_speed_drop_mps, rx_acc_.max_follow_speed_drop_mps);
    pnh_.param("acc/emergency_gap_m", rx_acc_.emergency_gap_m, rx_acc_.emergency_gap_m);
    pnh_.param("acc/critical_gap_m", rx_acc_.critical_gap_m, rx_acc_.critical_gap_m);
    pnh_.param("acc/critical_ttc_sec", rx_acc_.critical_ttc_sec, rx_acc_.critical_ttc_sec);
    pnh_.param("acc/soft_ttc_sec", rx_acc_.soft_ttc_sec, rx_acc_.soft_ttc_sec);
    pnh_.param("acc/hard_ttc_sec", rx_acc_.hard_ttc_sec, rx_acc_.hard_ttc_sec);
    pnh_.param("acc/target_front_acc_entry_abs_d_m", rx_acc_.target_front_acc_entry_abs_d_m, rx_acc_.target_front_acc_entry_abs_d_m);
    pnh_.param("acc/target_front_overlap_soft_acc_start_abs_d_m", rx_acc_.target_front_overlap_soft_acc_start_abs_d_m, rx_acc_.target_front_overlap_soft_acc_start_abs_d_m);
    pnh_.param("acc/lane_end_merge_wait_stop_distance_m", rx_acc_.lane_end_merge_wait_stop_distance_m, rx_acc_.lane_end_merge_wait_stop_distance_m);
    pnh_.param("acc/lane_change_lane_end_margin_m", rx_acc_.lane_change_lane_end_margin_m, rx_acc_.lane_change_lane_end_margin_m);
    pnh_.param("acc/lane_change_lane_end_min_cap_mps", rx_acc_.lane_change_lane_end_min_cap_mps, rx_acc_.lane_change_lane_end_min_cap_mps);
    pnh_.param("acc/lane_change_lane_end_cap_release_abs_d_m", rx_acc_.lane_change_lane_end_cap_release_abs_d_m, rx_acc_.lane_change_lane_end_cap_release_abs_d_m);
    pnh_.param("acc/closing_decel_boost_start_mps", rx_acc_.closing_decel_boost_start_mps, rx_acc_.closing_decel_boost_start_mps);
    pnh_.param("acc/closing_decel_boost_full_mps", rx_acc_.closing_decel_boost_full_mps, rx_acc_.closing_decel_boost_full_mps);
    pnh_.param("acc/lead_required_decel_limit_gain", rx_acc_.lead_required_decel_limit_gain, rx_acc_.lead_required_decel_limit_gain);
    pnh_.param("acc/lead_required_decel_limit_headway_sec", rx_acc_.lead_required_decel_limit_headway_sec, rx_acc_.lead_required_decel_limit_headway_sec);
    pnh_.param("acc/rapid_lead_ttc_sec", rx_acc_.rapid_lead_ttc_sec, rx_acc_.rapid_lead_ttc_sec);
    pnh_.param("acc/rapid_lead_min_closing_speed_mps", rx_acc_.rapid_lead_min_closing_speed_mps, rx_acc_.rapid_lead_min_closing_speed_mps);
    pnh_.param("acc/lead_required_decel_limit_enabled", rx_acc_.lead_required_decel_limit_enabled, rx_acc_.lead_required_decel_limit_enabled);
    pnh_.param("acc/hard_brake_bypass_filter", rx_acc_.hard_brake_bypass_filter, rx_acc_.hard_brake_bypass_filter);
    pnh_.param("acc/rapid_lead_brake_enabled", rx_acc_.rapid_lead_brake_enabled, rx_acc_.rapid_lead_brake_enabled);
    pnh_.param("acc/merge_keep_rolling_enabled", rx_acc_.merge_keep_rolling_enabled, rx_acc_.merge_keep_rolling_enabled);
    pnh_.param("acc/merge_keep_rolling_min_speed_mps", rx_acc_.merge_keep_rolling_min_speed_mps, rx_acc_.merge_keep_rolling_min_speed_mps);
    pnh_.param("acc/merge_keep_rolling_front_block_gap_m", rx_acc_.merge_keep_rolling_front_block_gap_m, rx_acc_.merge_keep_rolling_front_block_gap_m);
    pnh_.param("acc/merge_keep_rolling_low_speed_lead_mps", rx_acc_.merge_keep_rolling_low_speed_lead_mps, rx_acc_.merge_keep_rolling_low_speed_lead_mps);
    pnh_.param("acc/merge_keep_rolling_low_speed_lead_max_gap_m", rx_acc_.merge_keep_rolling_low_speed_lead_max_gap_m, rx_acc_.merge_keep_rolling_low_speed_lead_max_gap_m);
    pnh_.param("acc/min_target_front_gap_m", rx_acc_.min_target_front_gap_m, rx_acc_.min_target_front_gap_m);
    pnh_.param("acc/lane_end_stop_prepare_distance_m", rx_acc_.lane_end_stop_prepare_distance_m, rx_acc_.lane_end_stop_prepare_distance_m);
    pnh_.param("acc/lane_end_stop_prepare_time_sec", rx_acc_.lane_end_stop_prepare_time_sec, rx_acc_.lane_end_stop_prepare_time_sec);
    pnh_.param("acc/lane_end_stop_lane_ids", rx_acc_.lane_end_stop_lane_ids, rx_acc_.lane_end_stop_lane_ids);
    pnh_.param("tree_risk_weight", tree_risk_weight_, 80.0);
    pnh_.param("tree_abort_risk_factor", tree_abort_risk_factor_, 2.0);
    // 실행 계층이 실제로 하는 대응만(계속·복귀·제동). 계속+가속, 복귀+감속은 아무도 명령하지 않는다.
    pnh_.param("tree_executable_responses_only", tree_executable_responses_only_, true);
    // 복귀 횡이동 = 측정 진행률 곡선(시뮬레이터 plant 와 같은 표). false 면 예전 선형 모델.
    pnh_.param("tree_plant_return", tree_plant_return_, true);

    pnh_.param("lateral_mode_enabled", lateral_mode_enabled_, true);
    pnh_.param("lateral_mode_probability", lateral_mode_probability_, 0.03);
    pnh_.param("max_lateral_mode_probability", max_lateral_mode_probability_, 0.08);
    pnh_.param("lateral_mode_v_threshold_mps", lateral_mode_v_threshold_mps_, 0.90);
    pnh_.param("yaw_lateral_threshold_rad", yaw_lateral_threshold_rad_, 0.28);
    pnh_.param("yaw_lateral_probability_gain", yaw_lateral_probability_gain_, 0.04);
    pnh_.param("max_lateral_prediction_mps", max_lateral_prediction_mps_, 0.45);
    pnh_.param("lateral_mode_allow_nearby", lateral_mode_allow_nearby_, false);
    pnh_.param("lateral_mode_allow_rear", lateral_mode_allow_rear_, false);
    pnh_.param("lateral_mode_allow_yaw_only", lateral_mode_allow_yaw_only_, false);
    pnh_.param("lateral_mode_min_abs_d_m", lateral_mode_min_abs_d_m_, 1.40);
    pnh_.param("lateral_mode_require_yaw_agreement",
               lateral_mode_require_yaw_agreement_, true);
    pnh_.param("lateral_mode_require_away_from_center",
               lateral_mode_require_away_from_center_, true);
    pnh_.param("tv_yaw_rate_weight", tv_yaw_rate_weight_, 0.05);
    pnh_.param("tv_csv_yaw_weight", tv_csv_yaw_weight_, 0.90);
    pnh_.param("tv_no_lateral_probability_floor",
               tv_no_lateral_probability_floor_, 0.90);
    pnh_.param("tv_yaw_rate_horizon_sec", tv_yaw_rate_horizon_sec_, 0.8);
    pnh_.param("max_abs_tv_yaw_rate", max_abs_tv_yaw_rate_, 0.6);

    pnh_.param("prediction_sigma_base_m", prediction_sigma_base_m_, 0.5);
    pnh_.param("prediction_sigma_growth_mps", prediction_sigma_growth_mps_, 0.35);
    pnh_.param("prediction_nominal_sigma_base_m",
               prediction_nominal_sigma_base_m_, prediction_sigma_base_m_);
    pnh_.param("prediction_nominal_sigma_growth_mps",
               prediction_nominal_sigma_growth_mps_, prediction_sigma_growth_mps_);
    pnh_.param("prediction_front_brake_sigma_base_m",
               prediction_front_brake_sigma_base_m_, prediction_sigma_base_m_);
    pnh_.param("prediction_front_brake_sigma_growth_mps",
               prediction_front_brake_sigma_growth_mps_, prediction_sigma_growth_mps_);
    pnh_.param("prediction_rear_accel_sigma_base_m",
               prediction_rear_accel_sigma_base_m_, prediction_sigma_base_m_);
    pnh_.param("prediction_rear_accel_sigma_growth_mps",
               prediction_rear_accel_sigma_growth_mps_, prediction_sigma_growth_mps_);
    pnh_.param("prediction_lateral_sigma_base_m", prediction_lateral_sigma_base_m_, 0.25);
    pnh_.param("prediction_lateral_sigma_growth_mps", prediction_lateral_sigma_growth_mps_, 0.12);
    pnh_.param("collision_margin_m", collision_margin_m_, 0.45);

    pnh_.param("lane_change_base_cost", lane_change_base_cost_, 0.5);
    pnh_.param("risk_cost_weight", risk_cost_weight_, 80.0);
    pnh_.param("gap_cost_weight", gap_cost_weight_, 2.5);
    pnh_.param("blocked_cost_weight", blocked_cost_weight_, 35.0);
    pnh_.param("lane_end_prepare_cost", lane_end_prepare_cost_, 60.0);
    pnh_.param("lane_end_urgent_cost", lane_end_urgent_cost_, 180.0);
    pnh_.param("lane_end_emergency_cost", lane_end_emergency_cost_, 500.0);
    pnh_.param("decision_hysteresis_cost", decision_hysteresis_cost_, 0.1);
    pnh_.param("debug_publish_marker", debug_publish_marker_, true);
    pnh_.param("debug_marker_lifetime_sec", debug_marker_lifetime_sec_, 0.5);
    pnh_.param("debug_marker_height_m", debug_marker_height_m_, 4.0);
    pnh_.param("debug_prediction_publish_markers", debug_prediction_publish_markers_, true);
    pnh_.param("debug_prediction_marker_lifetime_sec", debug_prediction_marker_lifetime_sec_, 0.5);
    pnh_.param("debug_prediction_line_width_m", debug_prediction_line_width_m_, 0.25);
    pnh_.param("debug_prediction_z_m", debug_prediction_z_m_, 0.35);
    pnh_.param("debug_prediction_footprint_height_m", debug_prediction_footprint_height_m_, 0.15);
    pnh_.param("debug_prediction_ego_footprint_stride", debug_prediction_ego_footprint_stride_, 5);
    pnh_.param("debug_prediction_target_footprint_stride", debug_prediction_target_footprint_stride_, 5);
    pnh_.param("debug_prediction_target_max_vehicles", debug_prediction_target_max_vehicles_, 20);
    pnh_.param("debug_prediction_show_target_modes", debug_prediction_show_target_modes_, true);
    pnh_.param("debug_prediction_show_labels", debug_prediction_show_labels_, true);

    lanes_loaded_ = loadLanes();
    if (!lanes_loaded_) {
      ROS_WARN("[smpc_decision] failed to load lane CSVs from '%s'; v2 decision will publish KEEP only",
               waypoint_directory_.c_str());
    }

    targets_sub_ = nh_.subscribe("/smpc/targets", 10, &SmpcDecisionNode::targetsCallback, this);
    odom_sub_ = nh_.subscribe("/odom", 10, &SmpcDecisionNode::odomCallback, this);
    mission_speed_sub_ =
        nh_.subscribe(mission_speed_topic_, 10, &SmpcDecisionNode::missionSpeedCallback, this);
    acc_target_speed_sub_ = nh_.subscribe(
        ego_prediction_acc_target_speed_topic_, 10,
        &SmpcDecisionNode::accTargetSpeedCallback, this);
    behavior_status_sub_ = nh_.subscribe(
        behavior_status_topic_, 10, &SmpcDecisionNode::behaviorStatusCallback, this);
    reference_preview_sub_ = nh_.subscribe(
        ego_reference_preview_topic_, 10,
        &SmpcDecisionNode::referencePreviewCallback, this);
    decision_pub_ = nh_.advertise<smpc_lane_change::LaneChangeDecision>("/smpc/decision", 10);
    behavior_plan_pub_ = nh_.advertise<smpc_lane_change::BehaviorPlan>("/smpc/behavior_plan", 10);
    behavior_request_pub_ = nh_.advertise<smpc_lane_change::BehaviorLongitudinalRequest>(
        "/smpc/behavior_longitudinal_request", 10);
    debug_pub_ = nh_.advertise<std_msgs::String>("/smpc/smpc_decision_debug", 10);
    debug_marker_pub_ = nh_.advertise<visualization_msgs::Marker>("/smpc/decision_debug_marker", 1);
    prediction_marker_pub_ =
        nh_.advertise<visualization_msgs::MarkerArray>("/smpc/prediction_markers", 1);
    keep_path_pub_ = nh_.advertise<nav_msgs::Path>("/smpc/ego_keep_prediction", 1);
    change_path_pub_ = nh_.advertise<nav_msgs::Path>("/smpc/ego_change_prediction", 1);
    tree_abort_pub_ = nh_.advertise<std_msgs::Bool>("/smpc/tree_abort", 10);
    tree_debug_pub_ = nh_.advertise<std_msgs::String>("/smpc/tree_debug", 10);
    lc_status_sub_ = nh_.subscribe("/smpc/lane_change_status", 10,
                                   &SmpcDecisionNode::laneChangeStatusCallback, this);

    const double period = 1.0 / std::max(1.0, publish_rate_hz_);
    timer_ = nh_.createTimer(ros::Duration(period), &SmpcDecisionNode::timerCallback, this);

    const double horizon_sec = std::max(1, horizon_steps_) * prediction_dt_sec_;
    const double max_lc_start = behavior_lc_start_times_sec_.empty()
        ? 0.0
        : behavior_lc_start_times_sec_.back();
    const double required_candidate_horizon =
        behavior_execution_delay_sec_ + max_lc_start + egoChangeBlendDuration() +
        behavior_post_merge_buffer_sec_;
    if (behavior_planner_enabled_ && horizon_sec + 1e-6 < required_candidate_horizon) {
      ROS_WARN("[smpc_decision] behavior horizon %.2fs is shorter than the requested "
               "candidate coverage %.2fs; late t_LC candidates will be skipped",
               horizon_sec, required_candidate_horizon);
    }
    ROS_INFO("[smpc_decision] v3 enabled=%s lanes=%zu horizon=%d dt=%.2f eps=%.3f behavior=%s risk_nearby=%s ref_preview=%s%s",
             enabled_ ? "true" : "false", lanes_.size(), horizon_steps_, prediction_dt_sec_,
             risk_epsilon_, behavior_planner_enabled_ ? "true" : "false",
             risk_include_nearby_vehicles_ ? "true" : "false",
             ego_reference_preview_enabled_ ? "true" : "false",
             ego_reference_preview_required_for_change_ ? "/required" : "");
  }

 private:
  struct TvMode {
    std::string name;
    double probability{1.0};
    double accel_mps2{0.0};
    double lateral_rate_mps{0.0};
    // 이 모드에서 TV 가 낼 수 있는 최대 속도.  무한대면 기존 동작(상한 없음).
    double max_speed_mps{std::numeric_limits<double>::infinity()};
    // 트리 플래너(research): 이 시각까지는 등속, 이후 accel_mps2. 기존 mode 는 0 이다.
    double onset_sec{0.0};
  };

  struct VehicleHistory {
    bool initialized{false};
    double v_long{0.0};
    double filtered_accel{0.0};
    ros::Time first_seen_stamp;
    ros::Time last_seen_stamp;
    ros::Time stamp;
  };

  // `target_rear.valid == false` means no representative rear vehicle was
  // found in this target-lane scene, not that a merge may be committed from
  // the first LiDAR frame.  Keep a short, continuous scene history so a
  // newly appearing rear track is observed before it can be treated as a
  // safe gap.  This is a pre-commit perception guard; it does not replace
  // the per-step multimodal chance constraints below.
  struct TargetLaneSceneHistory {
    bool initialized{false};
    int current_lane_id{-1};
    int target_lane_id{-1};
    bool front_present{false};
    int front_unique_id{-1};
    bool rear_present{false};
    int rear_unique_id{-1};
    ros::Time stable_since_stamp;
    // Unlike stable_since_stamp, this does not reset when one continuously
    // tracked vehicle crosses ego and changes target_rear -> target_front.
    // It proves that the lane pair itself has been observed continuously.
    ros::Time lane_pair_stable_since_stamp;
    ros::Time last_seen_stamp;
  };

  struct PredPose {
    bool valid{false};
    double t{0.0};
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    double s{0.0};
    double d{0.0};
    double v{0.0};
  };

  struct LongitudinalRollout {
    double v{0.0};
    double ds{0.0};
  };

  struct ReferencePolyline {
    std::vector<geometry_msgs::Pose> poses;
    std::vector<double> arc_m;

    bool hasPoint() const {
      return !poses.empty() && poses.size() == arc_m.size();
    }

    bool valid() const {
      return poses.size() >= 2 && poses.size() == arc_m.size() &&
          std::isfinite(arc_m.back()) && arc_m.back() > 1e-4;
    }
  };

  struct ReferencePreview {
    bool valid{false};
    uint8_t mode{waypoint_system::PathSwitchPreview::INVALID};
    int source_lane_id{-1};
    int target_lane_id{-1};
    double blend_duration_sec{0.0};
    uint32_t blend_max_points{0};
    double connector_join_arc_m{std::numeric_limits<double>::quiet_NaN()};
    ReferencePolyline source;
    ReferencePolyline target;
    ReferencePolyline connector;
    ros::Time header_stamp;
    ros::Time received_stamp;
  };

  struct OrientedBox {
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    double half_l{0.0};
    double half_w{0.0};
  };

  struct CollisionEval {
    double probability{0.0};
    double signed_clearance{std::numeric_limits<double>::infinity()};
  };

  enum class BehaviorAction {
    kKeep,
    kWaitForGap,
    kChangeNow,
    kYieldDecelThenChange,
    kHoldThenChange,
    kAccelThenChange,
  };

  struct ActionSpec {
    BehaviorAction action{BehaviorAction::kKeep};
    double prep_accel_mps2{0.0};
    double lane_change_start_sec{-1.0};
    double lane_change_duration_sec{0.0};
    bool change_left{false};
    std::string name{"keep"};
  };

  struct ModeRiskTrace {
    std::string name;
    double probability{0.0};
    std::vector<double> conditional_risk_by_step;
  };

  struct VehicleRiskTrace {
    int unique_id{-1};
    std::string role;
    double priority_scale{1.0};
    std::vector<double> mixed_risk_by_step;
    std::vector<ModeRiskTrace> modes;
  };

  struct TrajectoryEval {
    bool valid{false};
    // Legacy risk preserves the previous mode-max aggregation for shadow
    // comparison and conservative rollout gating during the first stage.
    double risk{1.0};
    double peak_step_risk{1.0};
    std::size_t peak_step{0};
    double min_signed_clearance{std::numeric_limits<double>::infinity()};
    std::vector<double> risk_by_step;
    std::vector<double> min_signed_clearance_by_step;
    std::vector<VehicleRiskTrace> vehicle_risk_traces;
    std::string detail;
  };

  struct CandidateEval {
    ActionSpec spec;
    TrajectoryEval trajectory;
    double target_speed_mps{0.0};
    double cost{std::numeric_limits<double>::infinity()};
    bool time_risk_safe{false};
    bool legacy_risk_safe{false};
    bool clearance_safe{false};
    bool projected_gap_ttc_safe{false};
    bool longitudinal_model_compatible{false};
    bool feasible{false};
  };

  // A future YIELD must describe the same target-lane gap for more than one
  // LiDAR update before it may change the ACC cap.  This is deliberately
  // separate from the general vehicle-history filter: a known track can be
  // re-associated to the target lane for only one frame during a lane-boundary
  // crossing or an ID handoff.
  struct YieldCandidateConfirmation {
    bool active{false};
    int current_lane_id{std::numeric_limits<int>::min()};
    int target_lane_id{std::numeric_limits<int>::min()};
    int target_front_id{-1};
    int target_rear_id{-1};
    ros::Time stable_since_stamp;
  };

  void targetsCallback(const smpc_lane_change::TargetVehicleSet::ConstPtr& msg) {
    // 이력/장면 판단은 실제 관측만 쓴다.  유지 차량은 판정용 사본에만 넣는다.
    updateVehicleHistories(*msg);
    updateTargetLaneSceneHistory(*msg);
    smpc_lane_change::TargetVehicleSet held = *msg;
    updateHeldTargetVehicles(held);
    targets_ = held;
    targets_stamp_ = ros::Time::now();
    have_targets_ = true;
  }

  // 목표 차선 판단 구간에서 관측이 끊긴 차량을 짧게 유지한다 (D2-lite).
  // 트래커 출력 공백은 차가 사라졌다는 뜻이 아니다: lc_gt 백 15개에서 GT 실차
  // 트랙의 공백 383건 중 289건이 0.3 s 이하, 89% 가 0.5 s 이하, 1 s 초과는 없다.
  // lc_gt 2026-09-10-13-33-17 207.1 s 에서 42 m 뒤차(21.6 m/s)가 0.35 s 사라진
  // 사이에 승인이 나 GT 최소여유 -0.88 m 가 됐다.  등속 예측 오차(1 s 중앙 2.4 m)
  // 때문에 정확한 간격 계산용이 아니라 차단용으로만 쓰고, 자차 쪽으로 여유를 준다.
  void updateHeldTargetVehicles(smpc_lane_change::TargetVehicleSet& t) {
    // 디버그의 held= 는 이번 프레임에 실제로 주입한 대수를 뜻한다.  맵 크기를
    // 찍으면 관측 중인 차까지 세어 유지가 걸린 것처럼 보인다.
    held_injected_count_ = 0;
    if (!vanished_target_hold_enabled_) {
      held_target_vehicles_.clear();
      return;
    }
    const ros::Time stamp = t.header.stamp.isZero() ? ros::Time::now() : t.header.stamp;
    const double hold = std::max(0.0, vanished_target_hold_sec_);
    const double zone_front = std::max(0.0, vanished_target_hold_zone_front_m_);
    const double zone_rear = std::max(0.0, vanished_target_hold_zone_rear_m_);
    std::vector<int> seen;
    const auto consider = [&](const smpc_lane_change::TargetVehicle& v) {
      if (!v.valid || v.unique_id < 0) return;
      seen.push_back(v.unique_id);
      if (v.lane_id != t.target_lane_id) return;
      if (v.track_hits < vanished_target_hold_min_hits_) return;
      if (v.delta_s > zone_front || v.delta_s < -zone_rear) return;
      auto& h = held_target_vehicles_[v.unique_id];
      h.vehicle = v;
      h.last_seen_stamp = stamp;
      h.current_lane_id = t.current_lane_id;
      h.target_lane_id = t.target_lane_id;
    };
    consider(t.current_front);
    consider(t.target_front);
    consider(t.target_rear);
    for (const auto& v : t.nearby_vehicles) consider(v);

    for (auto it = held_target_vehicles_.begin(); it != held_target_vehicles_.end();) {
      const double age = (stamp - it->second.last_seen_stamp).toSec();
      const bool observed_now =
          std::find(seen.begin(), seen.end(), it->first) != seen.end();
      // 이번 프레임에 관측된 차량은 consider() 가 방금 최신값으로 갱신한 것이다.
      // 여기서 지우면 다음 프레임에 유지할 대상이 남지 않는다 (held 가 항상 0 이던 원인).
      if (observed_now) {
        ++it;
        continue;
      }
      // 자차 옆에서 사라진 차는 더 오래 유지한다.  옆은 라이다 사각지대라 트랙이
      // 죽고 다른 번호로 재획득되며, 공백이 같은 번호로 돌아오는 경우(실측 최대
      // 0.77 s)보다 길다: lc_gt 2026-09-16-13-07-43 에서 -3.2 m 옆 차가 29.06~
      // 30.92 s (1.86 s) 사라져 3차선이 비어 보였고 30.67 s 에 승인 -> GT -0.38 m,
      // 0.25 s 뒤 긴급 중단.  백 19개 측정(±4 m / 2.0 s): 위험 시작 1건 차단,
      // 유령 지연 0건, 안전한 시작 지연 0건 (차가 다시 잡히면 유지는 즉시 풀린다).
      const double entry_hold =
          std::abs(it->second.vehicle.delta_s) <=
                  std::max(0.0, vanished_target_hold_alongside_ds_m_)
              ? std::max(hold, std::max(0.0, vanished_target_hold_alongside_sec_))
              : hold;
      if (age < 0.0 || age > entry_hold ||
          it->second.current_lane_id != t.current_lane_id ||
          it->second.target_lane_id != t.target_lane_id) {
        it = held_target_vehicles_.erase(it);
        continue;
      }
      smpc_lane_change::TargetVehicle v = it->second.vehicle;
      const double shift = (v.v_long - ego_speed_mps_) * age;
      const double margin = std::max(0.0, vanished_target_hold_margin_m_);
      const double bias = v.delta_s >= 0.0 ? -margin : margin;
      v.delta_s += shift + bias;
      v.front_delta_s += shift + bias;
      v.rear_delta_s += shift + bias;
      v.s += v.v_long * age;
      v.x += v.v_long * std::cos(v.yaw) * age;
      v.y += v.v_long * std::sin(v.yaw) * age;
      if (v.delta_s > zone_front || v.delta_s < -zone_rear) {
        it = held_target_vehicles_.erase(it);
        continue;
      }
      const bool rear_side = v.rear_delta_s <= 0.0;
      v.role = rear_side ? "nearby_rear_held" : "nearby_front_held";
      t.nearby_vehicles.push_back(v);
      ++held_injected_count_;
      if (rear_side && !t.target_rear.valid) {
        smpc_lane_change::TargetVehicle r = v;
        r.role = "target_rear";
        t.target_rear = r;
      } else if (!rear_side && !t.target_front.valid) {
        smpc_lane_change::TargetVehicle f = v;
        f.role = "target_front";
        t.target_front = f;
      }
      ++it;
    }
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    ego_x_ = msg->pose.pose.position.x;
    ego_y_ = msg->pose.pose.position.y;
    const auto& q = msg->pose.pose.orientation;
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    ego_yaw_ = std::atan2(siny_cosp, cosy_cosp);
    const double new_speed =
        std::hypot(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    const ros::Time stamp = msg->header.stamp.isZero()
        ? ros::Time::now() : msg->header.stamp;
    if (have_ego_ && !ego_speed_stamp_.isZero()) {
      const double dt = (stamp - ego_speed_stamp_).toSec();
      if (dt > 1e-3 && dt < 1.0) {
        const double raw_accel = std::clamp(
            (new_speed - ego_speed_mps_) / dt, -12.0, 6.0);
        ego_longitudinal_accel_mps2_ =
            0.25 * raw_accel + 0.75 * ego_longitudinal_accel_mps2_;
      }
    }
    ego_speed_mps_ = new_speed;
    ego_speed_stamp_ = stamp;
    have_ego_ = true;
  }

  void missionSpeedCallback(const std_msgs::Float64::ConstPtr& msg) {
    if (!std::isfinite(msg->data)) return;
    mission_speed_target_mps_ = std::max(0.0, msg->data);
    mission_speed_stamp_ = ros::Time::now();
    have_mission_speed_target_ = true;
  }

  void accTargetSpeedCallback(const std_msgs::Float64::ConstPtr& msg) {
    if (!std::isfinite(msg->data)) return;
    acc_target_speed_mps_ = std::max(0.0, msg->data);
    acc_target_speed_stamp_ = ros::Time::now();
    have_acc_target_speed_ = true;
  }

  void behaviorStatusCallback(
      const smpc_lane_change::BehaviorLongitudinalStatus::ConstPtr& msg) {
    behavior_status_ = *msg;
    behavior_status_stamp_ = ros::Time::now();
    have_behavior_status_ = true;
  }

  static ReferencePolyline makeReferencePolyline(
      const std::vector<geometry_msgs::Pose>& poses) {
    ReferencePolyline out;
    out.poses = poses;
    out.arc_m.resize(out.poses.size(), 0.0);
    if (out.poses.size() < 2) return out;
    for (std::size_t i = 1; i < out.poses.size(); ++i) {
      const double dx = out.poses[i].position.x - out.poses[i - 1].position.x;
      const double dy = out.poses[i].position.y - out.poses[i - 1].position.y;
      const double ds = std::hypot(dx, dy);
      if (!std::isfinite(ds)) {
        out.poses.clear();
        out.arc_m.clear();
        return out;
      }
      out.arc_m[i] = out.arc_m[i - 1] + ds;
    }
    return out;
  }

  void referencePreviewCallback(
      const waypoint_system::PathSwitchPreview::ConstPtr& msg) {
    ReferencePreview preview;
    preview.valid = msg->valid;
    preview.mode = msg->mode;
    preview.source_lane_id = msg->source_path_number;
    preview.target_lane_id = msg->target_path_number;
    preview.blend_duration_sec = std::max(0.0, msg->blend_duration_sec);
    preview.blend_max_points = msg->blend_max_points;
    preview.connector_join_arc_m = msg->connector_join_arc_m;
    preview.source = makeReferencePolyline(msg->source_path);
    preview.target = makeReferencePolyline(msg->target_path);
    preview.connector = makeReferencePolyline(msg->connector_path);
    preview.header_stamp = msg->header.stamp;
    preview.received_stamp = ros::Time::now();
    reference_preview_ = std::move(preview);
    have_reference_preview_ = true;
  }

  double currentTargetSpeedMps() const {
    if (have_mission_speed_target_ &&
        (mission_speed_timeout_sec_ <= 0.0 ||
         (ros::Time::now() - mission_speed_stamp_).toSec() <= mission_speed_timeout_sec_)) {
      return mission_speed_target_mps_;
    }
    return mission_speed_fallback_mps_;
  }

  // ACC can be holding ego at the pre-end wait line while mission speed still
  // asks for cruise.  Use the lower fresh target for safety rollout so a
  // stopped ego is not modelled as immediately accelerating into a fast rear.
  double egoPredictionTargetSpeedMps() const {
    const double mission_target = currentTargetSpeedMps();
    if (!ego_prediction_use_acc_target_speed_ || !have_acc_target_speed_) {
      return mission_target;
    }
    if (ego_prediction_acc_target_speed_timeout_sec_ > 0.0 &&
        (ros::Time::now() - acc_target_speed_stamp_).toSec() >
            ego_prediction_acc_target_speed_timeout_sec_) {
      return mission_target;
    }
    return std::min(mission_target, acc_target_speed_mps_);
  }

  void updateOneVehicleHistory(const smpc_lane_change::TargetVehicle& v,
                               const ros::Time& stamp) {
    if (!v.valid || v.unique_id < 0) return;
    auto& h = vehicle_history_[v.unique_id];
    const bool discontinuous = h.initialized && !h.last_seen_stamp.isZero() &&
        (stamp - h.last_seen_stamp).toSec() >
            std::max(0.0, change_commit_observation_gap_timeout_sec_);
    if (!h.initialized || discontinuous) {
      h.initialized = true;
      h.v_long = v.v_long;
      h.filtered_accel = 0.0;
      h.first_seen_stamp = stamp;
      h.last_seen_stamp = stamp;
      h.stamp = stamp;
      return;
    }

    const double dt = (stamp - h.stamp).toSec();
    if (dt > 1e-3 && dt < 1.0) {
      const double raw_accel = std::clamp((v.v_long - h.v_long) / dt,
                                          -max_observed_accel_mps2_,
                                          max_observed_accel_mps2_);
      const double alpha = std::clamp(observed_accel_alpha_, 0.0, 1.0);
      h.filtered_accel = alpha * raw_accel + (1.0 - alpha) * h.filtered_accel;
    }

    h.v_long = v.v_long;
    h.last_seen_stamp = stamp;
    h.stamp = stamp;
  }

  void updateVehicleHistories(const smpc_lane_change::TargetVehicleSet& t) {
    const ros::Time stamp = t.header.stamp.isZero() ? ros::Time::now() : t.header.stamp;
    const smpc_lane_change::TargetVehicle candidates[3] = {
        t.current_front, t.target_front, t.target_rear};
    for (const auto& v : candidates) updateOneVehicleHistory(v, stamp);
    for (const auto& v : t.nearby_vehicles) updateOneVehicleHistory(v, stamp);
  }

  void updateTargetLaneSceneHistory(const smpc_lane_change::TargetVehicleSet& t) {
    const ros::Time stamp = t.header.stamp.isZero() ? ros::Time::now() : t.header.stamp;
    const bool front_present = t.target_front.valid && t.target_front.unique_id >= 0;
    const int front_id = front_present ? t.target_front.unique_id : -1;
    const bool rear_present = t.target_rear.valid && t.target_rear.unique_id >= 0 &&
        withinSceneObservationRange(t.target_rear);
    const int rear_id = rear_present ? t.target_rear.unique_id : -1;
    const bool stale = target_lane_scene_history_.initialized &&
        !target_lane_scene_history_.last_seen_stamp.isZero() &&
        (stamp - target_lane_scene_history_.last_seen_stamp).toSec() >
            std::max(0.0, change_commit_observation_gap_timeout_sec_);
    const bool lane_pair_changed = !target_lane_scene_history_.initialized || stale ||
        target_lane_scene_history_.current_lane_id != t.current_lane_id ||
        target_lane_scene_history_.target_lane_id != t.target_lane_id;
    const bool changed = lane_pair_changed ||
        target_lane_scene_history_.front_present != front_present ||
        (front_present && target_lane_scene_history_.front_unique_id != front_id) ||
        target_lane_scene_history_.rear_present != rear_present ||
        (rear_present && target_lane_scene_history_.rear_unique_id != rear_id);
    if (changed) {
      target_lane_scene_history_.initialized = true;
      target_lane_scene_history_.current_lane_id = t.current_lane_id;
      target_lane_scene_history_.target_lane_id = t.target_lane_id;
      target_lane_scene_history_.front_present = front_present;
      target_lane_scene_history_.front_unique_id = front_id;
      target_lane_scene_history_.rear_present = rear_present;
      target_lane_scene_history_.rear_unique_id = rear_id;
      target_lane_scene_history_.stable_since_stamp = stamp;
    }
    if (lane_pair_changed) {
      target_lane_scene_history_.lane_pair_stable_since_stamp = stamp;
    }
    target_lane_scene_history_.last_seen_stamp = stamp;
  }

  bool loadLanes() {
    lanes_.clear();
    if (waypoint_directory_.empty() || !fs::is_directory(waypoint_directory_)) return false;

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(waypoint_directory_)) {
      if (entry.is_regular_file() && entry.path().extension() == ".csv") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
      try {
        return std::stoi(a.stem().string()) < std::stoi(b.stem().string());
      } catch (...) {
        return a.filename().string() < b.filename().string();
      }
    });

    for (const auto& file : files) {
      int lane_id = -1;
      try {
        lane_id = std::stoi(file.stem().string());
      } catch (...) {
        continue;
      }
      smpc_lane_change::LanePath lane;
      if (lane.loadCsv(file.string())) lanes_[lane_id] = std::move(lane);
    }
    return !lanes_.empty();
  }

  const smpc_lane_change::LanePath* laneFor(int lane_id) const {
    const auto it = lanes_.find(lane_id);
    return it == lanes_.end() ? nullptr : &it->second;
  }

  static double normalCdf(double z) {
    return 0.5 * std::erfc(-z / std::sqrt(2.0));
  }

  static double clampProbability(double p) {
    return std::clamp(p, 0.0, 1.0);
  }

  static double smoothstep(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
  }

  static double angleLerp(double a, double b, double alpha) {
    return smpc_lane_change::wrapToPi(a + alpha * smpc_lane_change::wrapToPi(b - a));
  }

  static double projectPolylineArc(const ReferencePolyline& path,
                                   double x,
                                   double y) {
    if (!path.hasPoint()) return std::numeric_limits<double>::quiet_NaN();
    if (path.poses.size() == 1) return 0.0;
    double best_sq = std::numeric_limits<double>::infinity();
    double best_arc = 0.0;
    for (std::size_t i = 1; i < path.poses.size(); ++i) {
      const auto& a = path.poses[i - 1].position;
      const auto& b = path.poses[i].position;
      const double dx = b.x - a.x;
      const double dy = b.y - a.y;
      const double len_sq = dx * dx + dy * dy;
      if (len_sq < 1e-8) continue;
      const double u = std::clamp(((x - a.x) * dx + (y - a.y) * dy) / len_sq,
                                  0.0, 1.0);
      const double px = a.x + u * dx;
      const double py = a.y + u * dy;
      const double sq = (x - px) * (x - px) + (y - py) * (y - py);
      if (sq < best_sq) {
        best_sq = sq;
        best_arc = path.arc_m[i - 1] +
            u * (path.arc_m[i] - path.arc_m[i - 1]);
      }
    }
    return std::isfinite(best_sq) ? best_arc : std::numeric_limits<double>::quiet_NaN();
  }

  static PredPose samplePolyline(const ReferencePolyline& path, double arc_m) {
    PredPose out;
    if (!path.hasPoint() || !std::isfinite(arc_m) || arc_m < -1e-4 ||
        arc_m > path.arc_m.back() + 1e-4) {
      return out;
    }
    if (path.poses.size() == 1) {
      if (std::abs(arc_m) > 1e-4) return out;
      const auto& pose = path.poses.front();
      out.valid = true;
      out.x = pose.position.x;
      out.y = pose.position.y;
      const auto& q = pose.orientation;
      out.yaw = std::atan2(2.0 * (q.w * q.z + q.x * q.y),
                           1.0 - 2.0 * (q.y * q.y + q.z * q.z));
      return out;
    }
    arc_m = std::clamp(arc_m, 0.0, path.arc_m.back());
    const auto it = std::upper_bound(path.arc_m.begin(), path.arc_m.end(), arc_m);
    const std::size_t hi = std::clamp<std::size_t>(
        static_cast<std::size_t>(std::distance(path.arc_m.begin(), it)),
        1, path.arc_m.size() - 1);
    const std::size_t lo = hi - 1;
    const double segment = std::max(1e-6, path.arc_m[hi] - path.arc_m[lo]);
    const double u = std::clamp((arc_m - path.arc_m[lo]) / segment, 0.0, 1.0);
    const auto& a = path.poses[lo].position;
    const auto& b = path.poses[hi].position;
    out.valid = true;
    out.x = (1.0 - u) * a.x + u * b.x;
    out.y = (1.0 - u) * a.y + u * b.y;
    // Use the geometric tangent.  The same tangent is written by
    // local_path_publisher after it blends x/y, so yaw cannot describe a
    // different curve from the collision footprint.
    out.yaw = std::atan2(b.y - a.y, b.x - a.x);
    return out;
  }

  static ReferencePolyline blendReferencePreview(const ReferencePreview& preview,
                                                  double alpha) {
    ReferencePolyline out;
    if (!preview.source.valid() || !preview.target.valid()) return out;
    alpha = std::clamp(alpha, 0.0, 1.0);
    const std::size_t common = std::min(preview.source.poses.size(),
                                        preview.target.poses.size());
    const std::size_t max_points = preview.blend_max_points > 0
        ? static_cast<std::size_t>(preview.blend_max_points) : common;
    const std::size_t blend_size = std::min(common, max_points);
    if (blend_size < 2) return out;

    out.poses.reserve(preview.target.poses.size());
    for (std::size_t i = 0; i < blend_size; ++i) {
      geometry_msgs::Pose p = preview.target.poses[i];
      p.position.x = (1.0 - alpha) * preview.source.poses[i].position.x +
          alpha * preview.target.poses[i].position.x;
      p.position.y = (1.0 - alpha) * preview.source.poses[i].position.y +
          alpha * preview.target.poses[i].position.y;
      p.position.z = (1.0 - alpha) * preview.source.poses[i].position.z +
          alpha * preview.target.poses[i].position.z;
      out.poses.push_back(p);
    }
    for (std::size_t i = blend_size; i < preview.target.poses.size(); ++i) {
      out.poses.push_back(preview.target.poses[i]);
    }
    out.arc_m.resize(out.poses.size(), 0.0);
    for (std::size_t i = 1; i < out.poses.size(); ++i) {
      const double dx = out.poses[i].position.x - out.poses[i - 1].position.x;
      const double dy = out.poses[i].position.y - out.poses[i - 1].position.y;
      const double ds = std::hypot(dx, dy);
      if (!std::isfinite(ds)) {
        out.poses.clear();
        out.arc_m.clear();
        return out;
      }
      out.arc_m[i] = out.arc_m[i - 1] + ds;
    }
    return out;
  }

  bool referencePreviewFreshFor(const smpc_lane_change::TargetVehicleSet& t) const {
    if (!ego_reference_preview_enabled_) return false;
    if (!have_reference_preview_ || !reference_preview_.valid) return false;
    const ros::Time now = ros::Time::now();
    const auto fresh = [&](const ros::Time& stamp) {
      return ego_reference_preview_timeout_sec_ <= 0.0 ||
          (!stamp.isZero() && (now - stamp).toSec() <= ego_reference_preview_timeout_sec_);
    };
    if (!fresh(reference_preview_.received_stamp)) return false;
    if (!reference_preview_.header_stamp.isZero() &&
        !fresh(reference_preview_.header_stamp)) return false;
    if (reference_preview_.source_lane_id != t.current_lane_id ||
        reference_preview_.target_lane_id != t.target_lane_id) return false;
    if (reference_preview_.mode == waypoint_system::PathSwitchPreview::BLEND) {
      return reference_preview_.source.valid() && reference_preview_.target.valid() &&
          reference_preview_.blend_duration_sec > 1e-3;
    }
    if (reference_preview_.mode == waypoint_system::PathSwitchPreview::CONNECTOR) {
      return reference_preview_.source.hasPoint() && reference_preview_.connector.valid() &&
          std::isfinite(reference_preview_.connector_join_arc_m) &&
          reference_preview_.connector_join_arc_m > 1e-3 &&
          reference_preview_.connector_join_arc_m <=
              reference_preview_.connector.arc_m.back() + 1e-3;
    }
    if (reference_preview_.mode == waypoint_system::PathSwitchPreview::DIRECT) {
      return reference_preview_.source.valid() && reference_preview_.target.valid();
    }
    return false;
  }

  static const char* referencePreviewModeName(uint8_t mode) {
    switch (mode) {
      case waypoint_system::PathSwitchPreview::BLEND:
        return "blend";
      case waypoint_system::PathSwitchPreview::CONNECTOR:
        return "connector";
      case waypoint_system::PathSwitchPreview::DIRECT:
        return "direct";
      default:
        return "invalid";
    }
  }

  static double constantAccelDistance(double v0, double accel, double tau) {
    v0 = std::max(0.0, v0);
    tau = std::max(0.0, tau);
    if (accel < -1e-6) {
      const double t_stop = v0 / -accel;
      const double used_t = std::min(tau, t_stop);
      return std::max(0.0, v0 * used_t + 0.5 * accel * used_t * used_t);
    }
    return std::max(0.0, v0 * tau + 0.5 * accel * tau * tau);
  }

  LongitudinalRollout egoLongitudinalRollout(double tau,
                                             double target_speed_mps) const {
    LongitudinalRollout out;
    const double v0 = std::max(0.0, ego_speed_mps_);
    tau = std::max(0.0, tau);

    if (!ego_prediction_use_mission_speed_rollout_) {
      out.v = std::max(0.0, v0 + ego_assumed_accel_mps2_ * tau);
      out.ds = constantAccelDistance(v0, ego_assumed_accel_mps2_, tau);
      return out;
    }

    return speedTargetRollout(v0, tau, target_speed_mps);
  }

  // Speed-target rollout from an arbitrary initial speed: accelerate or brake
  // at the prediction limits until v_ref, then hold it.
  LongitudinalRollout speedTargetRollout(double v0, double tau,
                                         double target_speed_mps) const {
    LongitudinalRollout out;
    v0 = std::max(0.0, v0);
    tau = std::max(0.0, tau);
    const double v_ref = std::max(0.0, target_speed_mps);
    constexpr double kSpeedEps = 0.05;

    double accel = 0.0;
    if (v_ref > v0 + kSpeedEps) {
      accel = std::max(0.0, ego_prediction_max_accel_mps2_);
    } else if (v_ref < v0 - kSpeedEps) {
      accel = -std::max(0.0, ego_prediction_max_decel_mps2_);
    }

    if (std::abs(accel) < 1e-6) {
      out.v = v0;
      out.ds = v0 * tau;
      return out;
    }

    const double t_to_ref = std::max(0.0, (v_ref - v0) / accel);
    const double accel_t = std::min(tau, t_to_ref);
    out.v = std::max(0.0, v0 + accel * accel_t);
    out.ds = std::max(0.0, v0 * accel_t + 0.5 * accel * accel_t * accel_t);

    if (tau > accel_t) {
      out.v = v_ref;
      out.ds += v_ref * (tau - accel_t);
    }

    return out;
  }

  double egoChangeBlendDuration() const {
    if (ego_reference_preview_enabled_ && have_reference_preview_ &&
        reference_preview_.valid &&
        reference_preview_.mode == waypoint_system::PathSwitchPreview::BLEND &&
        reference_preview_.blend_duration_sec > 1e-3) {
      return reference_preview_.blend_duration_sec *
          std::max(0.1, ego_reference_preview_duration_scale_);
    }
    if (waypoint_blend_duration_sec_ > 1e-3) {
      return waypoint_blend_duration_sec_;
    }
    if (use_waypoint_linear_blend_ && waypoint_blend_steps_ > 0 && waypoint_blend_rate_hz_ > 1e-3) {
      return static_cast<double>(waypoint_blend_steps_) / waypoint_blend_rate_hz_;
    }
    return lane_change_duration_sec_;
  }

  double egoChangeAlpha(double tau, double start_sec, double duration_sec) const {
    const double duration = std::max(0.1, duration_sec);
    const double raw = std::clamp((tau - std::max(0.0, start_sec)) / duration, 0.0, 1.0);
    return use_waypoint_linear_blend_ ? raw : smoothstep(raw);
  }

  double effectiveLaneChangeStartSec(const ActionSpec& spec) const {
    if (!spec.change_left) return std::numeric_limits<double>::infinity();
    return std::max(0.0, spec.lane_change_start_sec) +
           std::max(0.0, behavior_execution_delay_sec_);
  }

  // A YIELD command is refreshed at 20 Hz while its future lane change is
  // repeatedly re-evaluated.  Anchor the speed floor to the beginning of
  // that continuous episode, rather than to the current speed every cycle;
  // otherwise a harmless, short yield can ratchet a highway-speed vehicle all
  // the way down to a crawl without a new safety reason.
  double behaviorYieldSpeedFloorMps(const ActionSpec& spec) const {
    if (spec.action != BehaviorAction::kYieldDecelThenChange) return 0.0;
    const double reference_speed =
        (last_behavior_action_ == BehaviorAction::kYieldDecelThenChange &&
         std::isfinite(behavior_yield_episode_start_speed_mps_))
            ? behavior_yield_episode_start_speed_mps_
            : std::max(0.0, ego_speed_mps_);
    return std::max(0.0, reference_speed - behavior_yield_max_speed_drop_mps_);
  }

  void clearYieldCandidateConfirmation() {
    yield_candidate_confirmation_ = YieldCandidateConfirmation{};
  }

  bool yieldCandidateConfirmed(const smpc_lane_change::TargetVehicleSet& t) {
    if (behavior_yield_confirm_sec_ <= 1e-6) return true;

    const int front_id = t.target_front.valid ? t.target_front.unique_id : -1;
    const int rear_id = t.target_rear.valid ? t.target_rear.unique_id : -1;
    const bool same_gap = yield_candidate_confirmation_.active &&
        yield_candidate_confirmation_.current_lane_id == t.current_lane_id &&
        yield_candidate_confirmation_.target_lane_id == t.target_lane_id &&
        yield_candidate_confirmation_.target_front_id == front_id &&
        yield_candidate_confirmation_.target_rear_id == rear_id;
    const ros::Time now = ros::Time::now();
    if (!same_gap) {
      yield_candidate_confirmation_.active = true;
      yield_candidate_confirmation_.current_lane_id = t.current_lane_id;
      yield_candidate_confirmation_.target_lane_id = t.target_lane_id;
      yield_candidate_confirmation_.target_front_id = front_id;
      yield_candidate_confirmation_.target_rear_id = rear_id;
      yield_candidate_confirmation_.stable_since_stamp = now;
      return false;
    }
    return !yield_candidate_confirmation_.stable_since_stamp.isZero() &&
        (now - yield_candidate_confirmation_.stable_since_stamp).toSec() >=
            behavior_yield_confirm_sec_;
  }

  LongitudinalRollout boundedPreparationRollout(double duration_sec,
                                                double requested_accel_mps2,
                                                double speed_cap_mps,
                                                double minimum_speed_mps) const {
    LongitudinalRollout out;
    const double v0 = std::max(0.0, ego_speed_mps_);
    const double duration = std::max(0.0, duration_sec);
    const double speed_cap = std::max(0.0, speed_cap_mps);
    const double accel = requested_accel_mps2;

    if (duration <= 1e-6) {
      out.v = v0;
      return out;
    }

    // HOLD means "do not add a longitudinal maneuver", not an instantaneous
    // jump to the latest ACC reference.  The candidate eligibility check
    // below rejects a HOLD/YIELD plan when normal ACC is already braking, so
    // keeping v0 here matches the separate behavior-rate contract in ACC.
    if (std::abs(accel) <= 1e-6) {
      out.v = v0;
      out.ds = v0 * duration;
      return out;
    }

    if (accel > 0.0) {
      if (v0 >= speed_cap) {
        out.v = speed_cap;
        out.ds = speed_cap * duration;
        return out;
      }
      const double t_to_cap = (speed_cap - v0) / accel;
      const double accel_time = std::clamp(t_to_cap, 0.0, duration);
      out.v = std::min(speed_cap, v0 + accel * accel_time);
      out.ds = v0 * accel_time + 0.5 * accel * accel_time * accel_time;
      if (duration > accel_time) out.ds += out.v * (duration - accel_time);
      return out;
    }

    // For a renewed YIELD request, the cap advances with each receding-
    // horizon cycle.  Its conservative closed-loop model is therefore a
    // persistent requested deceleration, limited only by zero speed.  The
    // normal ACC reference is checked separately before such a plan can be
    // executed, so it must not become an instantaneous lower-speed jump here.
    const double speed_floor = std::clamp(minimum_speed_mps, 0.0, v0);
    const double t_to_floor = (v0 - speed_floor) / std::max(1e-6, -accel);
    const double accel_time = std::min(duration, t_to_floor);
    out.v = std::max(speed_floor, v0 + accel * accel_time);
    out.ds = std::max(0.0, v0 * accel_time + 0.5 * accel * accel_time * accel_time);
    if (duration > accel_time) out.ds += out.v * (duration - accel_time);
    return out;
  }

  // ACC keeps the current-lane lead until the supervisor completes the lane
  // change (target_selector holds the source lane while active).  After that
  // the ego is no longer behind it and recovers toward the target-lane flow:
  // never faster than the target-lane front vehicle, never slower than the
  // pre-change cap.
  double changePostCompletionSpeedMps(const smpc_lane_change::TargetVehicleSet& t,
                                      double pre_cap_mps) const {
    double post = currentTargetSpeedMps();
    bool have_target_front_speed = false;
    if (t.target_front.valid && std::isfinite(t.target_front.v_long)) {
      post = std::min(post, std::max(0.0, t.target_front.v_long));
      have_target_front_speed = true;
    }
    // lc_gt 2026-09-17-12-23-58 차선2->3 (20.6~32.3 s) 측정: 목표차선 앞차가
    // 내내 유효하고 16~17 m/s 인데, 자차 ACC 목표가 그 속도를 넘어서는 24.4 s
    // 부터 pre_cap 바닥값이 앞차 속도를 덮어 v_post 가 25.0 으로 고정됐다.
    // 차선변경 후 그 앞차를 따라갈 것이 자명한데도 정속 25 로 달린다고 보아
    // 요구 전방간격이 19.8 -> 30.0 m 까지 올랐고, 실제 간격은 최대 14.6 m 라
    // 정상 게이트로는 끝내 통과하지 못했다(실제 합류는 pass_gap 예외로 32.34 s).
    // 앞차 속도를 실제로 관측한 경우에는 바닥값을 적용하지 않는다.  앞차가
    // 없을 때는 따라갈 대상이 없으므로 기존대로 pre_cap 을 유지한다.
    if (change_rollout_post_completion_use_pre_cap_floor_ ||
        !have_target_front_speed) {
      return std::max(pre_cap_mps, post);
    }
    return post;
  }

  // CHANGE_NOW rollout: the pre-change cap until release (start + fraction of
  // the lane-change duration), then speedTargetRollout toward the
  // post-completion speed.  Preparation candidates keep their persistent
  // worst-case model.
  // 합류 상한 롤아웃 (2026-09-21).  v_ref 를 향해 굴리되 감속률만 ACC 실제값에
  // 가깝게 제한한다.  기존 speedTargetRollout 의 감속 한계 9.0 m/s^2 는 실제 ACC
  // 제동(3.8~4.0)보다 훨씬 빨라 예측 주행거리를 과소평가하고 간격을 낙관한다.
  LongitudinalRollout blendCapRollout(double v0, double tau, double v_ref,
                                      double decel) const {
    LongitudinalRollout out;
    v0 = std::max(0.0, v0);
    tau = std::max(0.0, tau);
    v_ref = std::max(0.0, v_ref);
    constexpr double kEps = 0.05;
    double a = 0.0;
    if (v_ref > v0 + kEps) a = std::max(0.0, ego_prediction_max_accel_mps2_);
    else if (v_ref < v0 - kEps) a = -std::max(0.1, decel);
    if (std::abs(a) < 1e-6) {
      out.v = v0;
      out.ds = v0 * tau;
      return out;
    }
    const double t_ref = std::max(0.0, (v_ref - v0) / a);
    const double ta = std::min(tau, t_ref);
    out.v = std::max(0.0, v0 + a * ta);
    out.ds = std::max(0.0, v0 * ta + 0.5 * a * ta * ta);
    if (tau > ta) {
      out.v = v_ref;
      out.ds += v_ref * (tau - ta);
    }
    return out;
  }

  // 차선 변경 중 기준속도 보간 (2026-09-21, 사용자 설계).
  //   현재 차선 구간:  목표 속도(pre_cap)
  //   블렌드 구간:     목표 속도 -> 합류 후 ACC 추종속도 선형 보간
  //   완료 이후:       합류 후 ACC 추종속도 = min(순항, 목표차선 앞차속도)
  // 앞차가 자차보다 빠르면 min 이 순항을 고르므로 목표 속도로 주행하고,
  // 느리면 앞차 속도를 따른다 -- ACC 의 정상 동작과 같다.
  LongitudinalRollout blendInterpRollout(const smpc_lane_change::TargetVehicleSet& t,
                                         double tau, double pre_cap_mps,
                                         const ActionSpec& spec) const {
    const double lc_start = effectiveLaneChangeStartSec(spec);
    const double duration = std::max(0.1, spec.lane_change_duration_sec > 1e-3
        ? spec.lane_change_duration_sec : egoChangeBlendDuration());
    const double v_post = changePostCompletionSpeedMps(t, pre_cap_mps);
    const double dt = std::max(0.02, prediction_dt_sec_);
    double v = std::max(0.0, ego_speed_mps_);
    double ds = 0.0;
    for (double s = 0.0; s + 1e-9 < tau; s += dt) {
      const double step = std::min(dt, tau - s);
      const double alpha = std::clamp((s - lc_start) / duration, 0.0, 1.0);
      const double v_ref = (1.0 - alpha) * std::max(0.0, pre_cap_mps) +
          alpha * std::max(0.0, v_post);
      double v_next = v;
      if (v_ref > v + 0.05) {
        v_next = std::min(v_ref, v + std::max(0.0, ego_prediction_max_accel_mps2_) * step);
      } else if (v_ref < v - 0.05) {
        v_next = std::max(v_ref,
            v - std::max(0.1, change_rollout_blend_cap_decel_mps2_) * step);
      }
      ds += 0.5 * (v + v_next) * step;
      v = std::max(0.0, v_next);
    }
    LongitudinalRollout out;
    out.v = v;
    out.ds = ds;
    return out;
  }

  // ACC 3단계 모델 (2026-09-21, 사용자 설계).  ACC 가 실제로 하는 동작을 그대로 옮긴다:
  //   |d_target| > soft_start : 아직 소스 차선 기준 -> pre_cap
  //   |d_target| <= soft_start: 목표차선 앞차 추종  -> v_post = min(순항, 앞차속도)
  // (ACC: targetFrontOverlapSoftAccAllowed 는 |d|<=2.7, HardAcc 는 |d|<=0.75 에서 켜진다)
  // 블렌드는 선형이므로 |d(tau)| = (1-alpha)*|d0| 로 환산해 전환 시각을 구한다.
  // 정지/저속 대기에서는 pre_cap 이 0 이라 그대로 쓰면 종방향 이동이 없는(옆으로만
  // 미끄러지는) 궤적이 된다.  ACC 는 차선변경이 시작되면 정지 홀드를 풀고 출발하므로
  // 정지 출발 캡으로 대체한다.  BLEND 프리뷰 + 정지/저속은 백 7개에서 109 프레임 존재.
  LongitudinalRollout blendStageRollout(const smpc_lane_change::TargetVehicleSet& t,
                                        double tau, double pre_cap_mps,
                                        const ActionSpec& spec) const {
    const double lc_start = effectiveLaneChangeStartSec(spec);
    const double duration = std::max(0.1, spec.lane_change_duration_sec > 1e-3
        ? spec.lane_change_duration_sec : egoChangeBlendDuration());
    const double soft = std::max(0.0, change_rollout_soft_acc_start_abs_d_m_);
    const double d0 = std::isfinite(t.ego_d_target) ? std::abs(t.ego_d_target) : 0.0;
    double switch_sec = lc_start;
    if (d0 > soft + 1e-3) switch_sec = lc_start + duration * (1.0 - soft / d0);
    const double v_post = changePostCompletionSpeedMps(t, pre_cap_mps);
    const bool launching = ego_speed_mps_ <
        std::max(0.0, change_rollout_launch_speed_threshold_mps_);
    const double stage1_ref = launching
        ? std::max(pre_cap_mps, stoppedLaunchConnectorCapMps(t.target_lane_id))
        : pre_cap_mps;
    const double accel = launching
        ? std::max(0.0, stopped_launch_connector_max_accel_mps2_)
        : std::max(0.0, ego_prediction_max_accel_mps2_);
    const double decel = std::max(0.1, change_rollout_blend_cap_decel_mps2_);
    const double dt = std::max(0.02, prediction_dt_sec_);
    double v = std::max(0.0, ego_speed_mps_);
    double ds = 0.0;
    for (double sec = 0.0; sec + 1e-9 < tau; sec += dt) {
      const double step = std::min(dt, tau - sec);
      const double ref = (sec < switch_sec) ? stage1_ref : v_post;
      double vn = v;
      if (ref > v + 0.05) vn = std::min(ref, v + accel * step);
      else if (ref < v - 0.05) vn = std::max(ref, v - decel * step);
      ds += 0.5 * (v + vn) * step;
      v = std::max(0.0, vn);
    }
    LongitudinalRollout out;
    out.v = v;
    out.ds = ds;
    return out;
  }

  LongitudinalRollout changeAwareRollout(const smpc_lane_change::TargetVehicleSet& t,
                                         double tau, double pre_cap_mps,
                                         const ActionSpec& spec) const {
    if (!change_rollout_target_lane_speed_enabled_ ||
        !ego_prediction_use_mission_speed_rollout_ || !spec.change_left ||
        spec.action != BehaviorAction::kChangeNow) {
      return actionLongitudinalRollout(tau, pre_cap_mps, spec);
    }
    const double duration = spec.lane_change_duration_sec > 1e-3
        ? spec.lane_change_duration_sec
        : egoChangeBlendDuration();
    const double release_sec = effectiveLaneChangeStartSec(spec) +
        std::clamp(change_rollout_release_fraction_, 0.0, 1.0) * duration;
    if (change_rollout_blend_stage_enabled_) {
      return blendStageRollout(t, tau, pre_cap_mps, spec);
    }
    if (change_rollout_blend_speed_interp_enabled_) {
      return blendInterpRollout(t, tau, pre_cap_mps, spec);
    }
    // 블렌드 중 기준속도를 "합류 후 따라갈 속도"로 상한만 건다 (2026-09-21).
    // 올리지 않는다: ACC 가 자차를 목표차선 흐름보다 느리게 붙들고 있으면 그대로
    // 둔다.  release_fraction 을 낮춰 완료 후 속도를 앞당겨 적용했을 때는 그 값이
    // 오히려 예측 자차를 빠르게 만들어(ACC 10 m/s vs 목표차선 16.2 m/s) 요구가
    // 1.2 x 예측속도로 커지며 차선변경 3건이 사라졌다.  상한 형태는 그 경로가 없다.
    const double blend_ref = change_rollout_blend_speed_cap_enabled_
        ? std::min(pre_cap_mps, changePostCompletionSpeedMps(t, pre_cap_mps))
        : pre_cap_mps;
    if (tau <= release_sec) {
      return change_rollout_blend_speed_cap_enabled_
          ? blendCapRollout(ego_speed_mps_, tau, blend_ref,
                            change_rollout_blend_cap_decel_mps2_)
          : actionLongitudinalRollout(tau, pre_cap_mps, spec);
    }
    const LongitudinalRollout pre = change_rollout_blend_speed_cap_enabled_
        ? blendCapRollout(ego_speed_mps_, release_sec, blend_ref,
                          change_rollout_blend_cap_decel_mps2_)
        : actionLongitudinalRollout(release_sec, pre_cap_mps, spec);
    const LongitudinalRollout post = speedTargetRollout(
        pre.v, tau - release_sec, changePostCompletionSpeedMps(t, pre_cap_mps));
    LongitudinalRollout out;
    out.v = post.v;
    out.ds = pre.ds + post.ds;
    return out;
  }

  LongitudinalRollout actionLongitudinalRollout(double tau,
                                                double safe_speed_cap_mps,
                                                const ActionSpec& spec) const {
    if (spec.action == BehaviorAction::kKeep ||
        spec.action == BehaviorAction::kWaitForGap ||
        spec.action == BehaviorAction::kChangeNow) {
      return egoLongitudinalRollout(tau, safe_speed_cap_mps);
    }

    // A preparation request is re-evaluated at 20 Hz.  If it remains
    // selected, ACC receives a renewed cap and therefore continues the
    // requested yield/hold action instead of silently recovering at t_LC.
    // Model that persistent worst case across the horizon; it avoids
    // underestimating a fast rear vehicle after a deferred lane change.
    return boundedPreparationRollout(std::max(0.0, tau),
                                     spec.prep_accel_mps2,
                                     safe_speed_cap_mps,
                                     behaviorYieldSpeedFloorMps(spec));
  }

  double stoppedLaunchConnectorCapMps(int target_lane) const {
    const auto& by_lane = stopped_launch_connector_speed_cap_by_target_lane_mps_;
    if (target_lane >= 0 && target_lane < static_cast<int>(by_lane.size()) &&
        by_lane[target_lane] > 0.0) {
      return by_lane[target_lane];
    }
    return std::max(0.0, stopped_launch_connector_speed_cap_mps_);
  }

  LongitudinalRollout connectorLaunchRollout(double tau,
                                             double safe_speed_cap_mps,
                                             const ActionSpec& spec,
                                             double lane_change_start_sec,
                                             int target_lane = -1) const {
    const double start = std::max(0.0, lane_change_start_sec);
    if (tau <= start + 1e-6) {
      return actionLongitudinalRollout(tau, safe_speed_cap_mps, spec);
    }

    // Once waypoint_system has switched to a valid endpoint connector, ACC
    // applies its dedicated connector cap.  Model the same 0 -> cap launch
    // instead of retaining the pre-switch lane-end target of 0 m/s.
    const LongitudinalRollout pre_switch =
        actionLongitudinalRollout(start, safe_speed_cap_mps, spec);
    const double elapsed = tau - start;
    const double accel = std::max(0.0, stopped_launch_connector_max_accel_mps2_);
    const double cap = stoppedLaunchConnectorCapMps(target_lane);
    const double v0 = std::max(0.0, pre_switch.v);

    LongitudinalRollout out;
    if (accel <= 1e-6 || v0 >= cap - 1e-6) {
      out.v = v0;
      out.ds = pre_switch.ds + v0 * elapsed;
      return out;
    }

    const double accel_time = std::min(elapsed, (cap - v0) / accel);
    out.v = std::min(cap, v0 + accel * accel_time);
    out.ds = pre_switch.ds + v0 * accel_time +
        0.5 * accel * accel_time * accel_time;
    if (elapsed > accel_time) out.ds += out.v * (elapsed - accel_time);
    return out;
  }

  double connectorTravelTimeSec(double initial_speed_mps,
                                double distance_m,
                                int target_lane = -1) const {
    const double distance = std::max(0.0, distance_m);
    if (distance <= 1e-6) return 0.0;
    const double v0 = std::max(0.0, initial_speed_mps);
    const double accel = std::max(0.0, stopped_launch_connector_max_accel_mps2_);
    const double cap = stoppedLaunchConnectorCapMps(target_lane);
    if (v0 >= cap - 1e-6) {
      return v0 > 1e-6 ? distance / v0 : std::numeric_limits<double>::infinity();
    }
    if (accel <= 1e-6) return std::numeric_limits<double>::infinity();
    const double t_to_cap = (cap - v0) / accel;
    const double d_to_cap = v0 * t_to_cap + 0.5 * accel * t_to_cap * t_to_cap;
    if (distance <= d_to_cap) {
      return (-v0 + std::sqrt(std::max(0.0, v0 * v0 + 2.0 * accel * distance))) /
          accel;
    }
    return t_to_cap + (distance - d_to_cap) / std::max(1e-6, cap);
  }

  double actionTargetSpeedMps(double safe_speed_cap_mps,
                              const ActionSpec& spec) const {
    if (spec.action == BehaviorAction::kKeep ||
        spec.action == BehaviorAction::kWaitForGap ||
        spec.action == BehaviorAction::kChangeNow) {
      return std::max(0.0, safe_speed_cap_mps);
    }
    return actionLongitudinalRollout(effectiveLaneChangeStartSec(spec),
                                     safe_speed_cap_mps, spec).v;
  }

  double vehicleHalfS(const smpc_lane_change::TargetVehicle& v) const {
    return v.half_s > 1e-3 ? v.half_s : 0.5 * std::max(0.0, v.length);
  }

  double frontEdgeDeltaS(const smpc_lane_change::TargetVehicle& v) const {
    return v.front_delta_s != 0.0 ? v.front_delta_s : v.delta_s + vehicleHalfS(v);
  }

  double rearEdgeDeltaS(const smpc_lane_change::TargetVehicle& v) const {
    return v.rear_delta_s != 0.0 ? v.rear_delta_s : v.delta_s - vehicleHalfS(v);
  }

  double frontBumperGap(const smpc_lane_change::TargetVehicle& front) const {
    return rearEdgeDeltaS(front) - ego_front_extent_m_;
  }

  double rearBumperGap(const smpc_lane_change::TargetVehicle& rear) const {
    return -frontEdgeDeltaS(rear) - ego_rear_extent_m_;
  }

  double observedLongitudinalAccel(const smpc_lane_change::TargetVehicle& v) const {
    if (v.unique_id < 0) return 0.0;
    const auto it = vehicle_history_.find(v.unique_id);
    if (it == vehicle_history_.end() || !it->second.initialized) return 0.0;
    return it->second.filtered_accel;
  }

  // 속도 상한이 있는 등가속 (2026-09-22).  상한 도달 후에는 등속으로 간다.
  // rear_accel 모드가 지평선 끝까지 무제한 가속해 11 m/s 뒤차가 6.4 s 뒤
  // 23.8 m/s 가 되던 문제를 막는다.  상한이 무한대면 기존 식과 동일하다.
  static double cappedAccelSpeed(double v0, double accel, double tau,
                                 double v_max) {
    v0 = std::max(0.0, v0);
    tau = std::max(0.0, tau);
    return std::clamp(v0 + accel * tau, 0.0, std::max(v0, v_max));
  }

  static double cappedAccelDistance(double v0, double accel, double tau,
                                    double v_max) {
    v0 = std::max(0.0, v0);
    tau = std::max(0.0, tau);
    if (!std::isfinite(v_max) || v_max <= 0.0 || accel <= 1e-6) {
      return constantAccelDistance(v0, accel, tau);
    }
    if (v0 >= v_max) return std::max(0.0, v0 * tau);
    const double t_cap = (v_max - v0) / accel;
    if (tau <= t_cap) return constantAccelDistance(v0, accel, tau);
    const double d_cap = v0 * t_cap + 0.5 * accel * t_cap * t_cap;
    return std::max(0.0, d_cap + v_max * (tau - t_cap));
  }

  // onset 이 있는 mode: onset 전 등속, 이후 cappedAccel*. onset 0 이면 기존 식과 같다.
  static double modeSpeed(double v0, const TvMode& mode, double tau) {
    if (mode.onset_sec <= 0.0) return cappedAccelSpeed(v0, mode.accel_mps2, tau, mode.max_speed_mps);
    return cappedAccelSpeed(v0, mode.accel_mps2, std::max(0.0, tau - mode.onset_sec), mode.max_speed_mps);
  }

  static double modeDistance(double v0, const TvMode& mode, double tau) {
    if (mode.onset_sec <= 0.0) return cappedAccelDistance(v0, mode.accel_mps2, tau, mode.max_speed_mps);
    const double before = std::min(std::max(0.0, tau), mode.onset_sec);
    return std::max(0.0, v0) * before +
        cappedAccelDistance(v0, mode.accel_mps2, std::max(0.0, tau - mode.onset_sec), mode.max_speed_mps);
  }

  static double pressure01(double value, double low, double high) {
    if (high <= low) return value >= high ? 1.0 : 0.0;
    return std::clamp((value - low) / (high - low), 0.0, 1.0);
  }

  double adaptiveEventProbability(const smpc_lane_change::TargetVehicle& v) const {
    const bool rear = v.role.find("rear") != std::string::npos;
    const double accel_obs = observedLongitudinalAccel(v);

    if (rear && rear_accel_probability_by_accel_enabled_ &&
        rear_accel_probability_by_accel_.size() ==
            rear_accel_probability_accel_breaks_mps2_.size() + 1) {
      // Observed-accel table fitted to target-lane rear vehicles: the event is
      // "5 s later at least half the rear_accel displacement beyond constant
      // speed", keyed on the same filtered accel used here.  Closing speed,
      // TTC and gap were not predictive of that event, so the table replaces
      // the base + pressure sum (which saturated at the max for most samples).
      std::size_t bin = 0;
      while (bin < rear_accel_probability_accel_breaks_mps2_.size() &&
             accel_obs >= rear_accel_probability_accel_breaks_mps2_[bin]) {
        ++bin;
      }
      return std::clamp(rear_accel_probability_by_accel_[bin],
                        min_rear_accel_probability_, max_rear_accel_probability_);
    }

    if (rear) {
      const double gap = std::max(0.0, rearBumperGap(v));
      const double closing = std::max(0.0, v.v_long - ego_speed_mps_);
      const double ttc = closing > 0.1 ? gap / closing : std::numeric_limits<double>::infinity();
      const double ttc_pressure = std::isfinite(ttc)
          ? pressure01(probability_ttc_safe_sec_ - ttc,
                       0.0,
                       probability_ttc_safe_sec_ - probability_ttc_critical_sec_)
          : 0.0;
      const double speed_pressure =
          pressure01(closing, 0.0, probability_rel_speed_scale_mps_);
      const double accel_pressure =
          pressure01(std::max(0.0, accel_obs), 0.0, probability_accel_scale_mps2_);
      const double near_pressure =
          pressure01(target_rear_safe_gap_m_ - gap, 0.0, target_rear_safe_gap_m_);

      const double p = rear_accel_probability_ +
          0.25 * ttc_pressure +
          0.18 * speed_pressure +
          0.12 * accel_pressure +
          0.10 * near_pressure;
      return std::clamp(p, min_rear_accel_probability_, max_rear_accel_probability_);
    }

    const double gap = std::max(0.0, frontBumperGap(v));
    const double desired_gap = std::max(
        current_front_safe_gap_m_, standstill_gap_m_ + time_headway_sec_ * ego_speed_mps_);
    const double closing = std::max(0.0, ego_speed_mps_ - v.v_long);
    const double ttc = closing > 0.1 ? gap / closing : std::numeric_limits<double>::infinity();
    const double ttc_pressure = std::isfinite(ttc)
        ? pressure01(probability_ttc_safe_sec_ - ttc,
                     0.0,
                     probability_ttc_safe_sec_ - probability_ttc_critical_sec_)
        : 0.0;
    const double speed_pressure =
        pressure01(closing, 0.0, probability_rel_speed_scale_mps_);
    const double decel_pressure =
        pressure01(std::max(0.0, -accel_obs), 0.0, probability_accel_scale_mps2_);
    const double near_pressure =
        pressure01(desired_gap - gap, 0.0, desired_gap);

    const double p = front_brake_probability_ +
        0.25 * ttc_pressure +
        0.18 * speed_pressure +
        0.12 * decel_pressure +
        0.10 * near_pressure;
    return std::clamp(p, min_front_brake_probability_, max_front_brake_probability_);
  }

  double gapPenalty(double gap, double desired_gap) const {
    if (!std::isfinite(gap)) return 0.0;
    const double shortage = std::max(0.0, desired_gap - gap);
    return gap_cost_weight_ * shortage * shortage / std::max(1.0, desired_gap);
  }

  double laneEndPressureCost(const smpc_lane_change::TargetVehicleSet& t) const {
    if (!t.lane_change_required) return 0.0;
    if (t.emergency_stop_required) return lane_end_emergency_cost_;
    if (t.lane_change_urgent) return lane_end_urgent_cost_;
    if (t.lane_change_prepare) return lane_end_prepare_cost_;
    return 0.0;
  }

  double activeTargetFrontSafeGap(const smpc_lane_change::TargetVehicleSet& t) const {
    // Reaching a CSV endpoint changes the urgency to look for a gap, never
    // the physical distance required to enter one.  Relaxing this gate was a
    // direct way to turn a late highway merge into a rear/front conflict.
    (void)t;
    return target_front_safe_gap_m_;
  }

  double activeTargetRearSafeGap(const smpc_lane_change::TargetVehicleSet& t) const {
    // Same invariant for a fast target-lane rear vehicle.  The low-speed
    // guard below may only increase this requirement.
    (void)t;
    return target_rear_safe_gap_m_;
  }

  double activeRiskEpsilon(const smpc_lane_change::TargetVehicleSet& t) const {
    // Lane-end urgency is a scheduling signal, not permission to accept a
    // higher collision probability.  Keep the chance constraint invariant;
    // if no safe merge remains, the endpoint holding policy handles it.
    (void)t;
    return risk_epsilon_;
  }

  static bool roleContains(const smpc_lane_change::TargetVehicle& v,
                           const std::string& token) {
    return v.role.find(token) != std::string::npos;
  }

  static bool laneEndPressureActive(const smpc_lane_change::TargetVehicleSet& t) {
    return t.lane_change_prepare || t.lane_change_urgent || t.emergency_stop_required;
  }

  bool isRearLikeVehicle(const smpc_lane_change::TargetVehicle& v) const {
    return roleContains(v, "rear");
  }

  bool lowSpeedRearSafetyActive() const {
    return low_speed_rear_safety_enabled_ &&
        ego_speed_mps_ <= std::max(0.0, low_speed_rear_ego_speed_mps_);
  }

  bool targetFrontTtcSafe(const smpc_lane_change::TargetVehicle& front) const {
    if (!front.valid) return true;
    const double closing = std::max(0.0, ego_speed_mps_ - front.v_long);
    if (closing <= 0.1) return true;
    const double gap = std::max(0.0, frontBumperGap(front));
    return gap / closing >= std::max(0.0, target_front_min_ttc_sec_);
  }

  bool targetFrontHeadwaySafe(const smpc_lane_change::TargetVehicle& front) const {
    if (!front.valid) return true;
    const double gap = std::max(0.0, frontBumperGap(front));
    // projectedTargetGapTtcSafe 의 전방 헤드웨이와 같은 기준을 쓴다.
    const double reference_speed = target_front_headway_use_closing_speed_
        ? std::max(0.0, ego_speed_mps_ - std::max(0.0, front.v_long))
        : std::max(std::max(0.0, ego_speed_mps_), std::max(0.0, front.v_long));
    return gap >= std::max(0.0, target_front_min_headway_sec_) * reference_speed;
  }

  bool targetRearTtcSafe(const smpc_lane_change::TargetVehicle& rear) const {
    if (!rear.valid) return true;
    const double closing = conservativeRearClosingSpeed(
        rear, ego_speed_mps_, ros::Time::now());
    if (closing <= 0.1) return true;
    const double required_ttc = lowSpeedRearSafetyActive()
        ? std::max(target_rear_min_ttc_sec_, low_speed_rear_min_ttc_sec_)
        : target_rear_min_ttc_sec_;
    const double gap = std::max(0.0, rearBumperGap(rear));
    return gap / closing >= std::max(0.0, required_ttc);
  }

  bool targetRearHeadwaySafe(const smpc_lane_change::TargetVehicle& rear) const {
    if (!rear.valid) return true;
    const double gap = std::max(0.0, rearBumperGap(rear));
    // A rear time margin protects against a vehicle *closing* on ego.  Using
    // max(ego_speed, rear_speed) here required 25--30 m even when ego was
    // pulling away from a slower rear vehicle, which turned safe HOLD/CHANGE
    // opportunities into unnecessary YIELD commands.  The independent
    // bumper-gap, 3 s TTC, and every-step predicted-gap gates remain active.
    const double closing = conservativeRearClosingSpeed(
        rear, ego_speed_mps_, ros::Time::now());
    const double required_gap =
        std::max(0.0, target_rear_min_headway_sec_) * closing;
    return gap >= required_gap;
  }

  double conservativeRearClosingSpeed(
      const smpc_lane_change::TargetVehicle& rear,
      double ego_speed,
      const ros::Time& now) const {
    double closing = std::max(0.0, rear.v_long - std::max(0.0, ego_speed));
    if (!rear.valid || rear.unique_id < 0) return closing;
    const auto it = vehicle_history_.find(rear.unique_id);
    if (it == vehicle_history_.end() || it->second.first_seen_stamp.isZero()) {
      return closing;
    }
    const double age = std::max(0.0, (now - it->second.first_seen_stamp).toSec());
    const double gap = std::max(0.0, rearBumperGap(rear));
    if (age < std::max(0.0, new_rear_track_guard_sec_) &&
        gap <= std::max(0.0, new_rear_track_max_gap_m_)) {
      closing = std::max(closing, std::max(0.0, new_rear_track_closing_upper_mps_));
    }
    return closing;
  }

  double targetRearRequiredGap(
      const smpc_lane_change::TargetVehicleSet& t,
      const smpc_lane_change::TargetVehicle& rear,
      double remaining_lane_change_sec,
      double ego_speed,
      const ros::Time& now) const {
    double base_gap = activeTargetRearSafeGap(t);
    if (low_speed_rear_safety_enabled_ &&
        ego_speed <= std::max(0.0, low_speed_rear_ego_speed_mps_)) {
      base_gap = std::max(base_gap, std::max(0.0, low_speed_rear_safe_gap_m_));
    }
    if (!rear.valid) return base_gap;
    const double closing = conservativeRearClosingSpeed(rear, ego_speed, now);
    const double exposure = std::max(0.0, remaining_lane_change_sec) +
        std::max(0.0, rear_gap_perception_control_delay_sec_);
    return base_gap + closing * exposure;
  }

  bool accHardBrakeOrLaneEndActive() const {
    if (!have_behavior_status_) return true;
    std::string reason = behavior_status_.reason;
    std::transform(reason.begin(), reason.end(), reason.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    // "lane_end_wait_*" is the ACC telling us it is holding for a gap so the
    // ego can change lanes before the lane ends.  Treating it as a hard-brake
    // state deadlocks the maneuver: the wait blocks the commit, the ego never
    // changes lanes, so the wait persists.  Measured on lc_gt 2026-09-07:
    // lane_end_wait_no_safe_gap 61.5~66.6% + lane_end_wait_front_gap
    // 20.6~21.6% of behavior_status frames, blocking acc_commit_ok 83~90% of
    // the time, and acc_commit_ok was the sole remaining blocker in 20~30% of
    // frames.  Every other lane_end_* reason (stop, creep, emergency, hard
    // stop) still blocks; note that "stop" is not matched on its own, so the
    // lane_end prefix is what keeps those out.  A wait reason carrying a
    // hard/rapid/emergency suffix is still caught by the checks below.
    const bool lane_end_blocking =
        reason.find("lane_end") != std::string::npos &&
        reason.find("lane_end_wait") == std::string::npos;
    return reason.find("hard") != std::string::npos ||
        reason.find("rapid") != std::string::npos ||
        lane_end_blocking ||
        reason.find("emergency") != std::string::npos;
  }

  // The lane_end_wait_* profile (lane_end_comfort_decel ~2.5 m/s^2) is ACC
  // holding for this very gap, not a hazard brake, and ACC releases it on
  // CHANGE_LEFT.  Exempting it from the decel gate lets a merge that becomes
  // feasible while slowing commit before the ego stops (lc_gt 2026-09-15
  // 16-00/16-08/16-14: target lane clear for ~1 s at 4-5 m/s before the stop,
  // blocked only by this gate).  Emergency stop/creep variants stay blocked.
  bool accPlannedLaneEndWaitDecel() const {
    if (!change_commit_allow_lane_end_wait_decel_ || !have_behavior_status_) return false;
    std::string reason = behavior_status_.reason;
    std::transform(reason.begin(), reason.end(), reason.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return reason.rfind("lane_end_wait", 0) == 0 &&
        reason.find(":stop") == std::string::npos &&
        reason.find(":creep") == std::string::npos &&
        reason.find("emergency") == std::string::npos &&
        reason.find("hard") == std::string::npos &&
        reason.find("rapid") == std::string::npos;
  }

  bool accStableForChangeCommit() {
    const ros::Time now = ros::Time::now();
    bool stable_now = have_behavior_status_ &&
        (behavior_status_timeout_sec_ <= 0.0 ||
         (now - behavior_status_stamp_).toSec() <= behavior_status_timeout_sec_) &&
        std::isfinite(behavior_status_.final_target_speed_mps) &&
        !accHardBrakeOrLaneEndActive() &&
        // The speed-error term was removed.  The ACC target itself steps
        // between extremes (measured on lc_gt 2026-09-07-14-18: reason=none
        // commands 25.0 m/s, lane_end_wait_front_gap commands 0.0, and
        // final_target jumped by more than 3 m/s within a second 387 times in
        // 972 frames), so the ego is almost always mid-convergence.  51.1% of
        // frames exceeded the 1.5 m/s window and 42.5% of those were the ego
        // still accelerating toward an unreachable target -- only 8.6% were
        // the ego running fast.  Being behind a stepped-up target is not a
        // reason to refuse a lane change; committing while braking is, and
        // the decel gate below plus the hard/rapid/emergency check above
        // still cover that.  Restore this term only together with a rate
        // limit on the ACC target.
        (accPlannedLaneEndWaitDecel() ||
         ego_longitudinal_accel_mps2_ >=
             -std::max(0.0, change_commit_acc_max_decel_mps2_));
    if (!stable_now) {
      acc_commit_stable_since_ = ros::Time(0.0);
      return false;
    }
    if (acc_commit_stable_since_.isZero()) acc_commit_stable_since_ = now;
    return (now - acc_commit_stable_since_).toSec() >=
        std::max(0.0, change_commit_acc_stable_sec_);
  }

  // 센서가 신뢰 구간 밖의 뒤차를 간헐적으로만 보여줄 때, 그 깜빡임을
  // "장면이 바뀌었다"로 읽으면 scene 안정 시계가 계속 리셋된다.
  // lc_gt 14-18 측정: 목표 차선 뒤차의 트래커 검출률이 0~40 m 94~98 %,
  // 40~60 m 47 %, 60 m 이상 0 %.  VLP-16 링 간격이 d*tan(2deg) 이므로
  // 60 m 에서 2.1 m > 차량 높이 1.5 m 라 링이 차량을 아예 지나친다.
  // 관측 불가 거리의 부재는 "관측 실패"가 아니라 "그 거리에 대표차 없음"으로
  // 다룬다.  실제 gap/TTC 게이트는 target_rear 가 유효하면 그대로 사용하므로
  // 안전 판정 자체는 약해지지 않는다.  0 이면 기존 동작.
  bool withinSceneObservationRange(
      const smpc_lane_change::TargetVehicle& v) const {
    if (change_commit_scene_rear_max_range_m_ <= 0.0) return true;
    if (!v.valid) return true;
    if (!std::isfinite(v.delta_s)) return true;
    return std::abs(v.delta_s) <= change_commit_scene_rear_max_range_m_;
  }

  bool vehicleObservationStable(const smpc_lane_change::TargetVehicle& v,
                                const ros::Time& now) const {
    if (!v.valid || v.unique_id < 0) return true;
    const auto it = vehicle_history_.find(v.unique_id);
    if (it == vehicle_history_.end() || !it->second.initialized ||
        it->second.first_seen_stamp.isZero() || it->second.last_seen_stamp.isZero()) {
      return false;
    }
    const double timeout = std::max(0.0, change_commit_observation_gap_timeout_sec_);
    if (timeout > 0.0 && (now - it->second.last_seen_stamp).toSec() > timeout) return false;
    return (now - it->second.first_seen_stamp).toSec() >=
        std::max(0.0, change_commit_vehicle_observation_sec_);
  }

  bool targetFrontIsPassingVehicle(
      const smpc_lane_change::TargetVehicleSet& t) const {
    if (!pass_gap_enabled_ || !t.target_front.valid ||
        t.target_front.unique_id < 0 ||
        t.target_front.lane_id != t.target_lane_id) {
      return false;
    }
    const double relative_speed = t.target_front.v_long - ego_speed_mps_;
    // target_front is produced as soon as the vehicle's front edge reaches
    // ego.  Restrict this exception to a faster, already-tracked vehicle near
    // ego; an ordinary slower lead must retain the full front-gap gate.
    return relative_speed >= std::max(0.0, pass_gap_min_relative_speed_mps_) &&
        frontBumperGap(t.target_front) <=
            std::max(0.0, pass_gap_front_max_distance_m_);
  }

  const smpc_lane_change::TargetVehicle* targetRearForGap(
      const smpc_lane_change::TargetVehicleSet& t,
      int excluded_id) const {
    const smpc_lane_change::TargetVehicle* best = nullptr;
    double best_gap = std::numeric_limits<double>::infinity();
    const auto consider = [&](const smpc_lane_change::TargetVehicle& v) {
      if (!v.valid || v.unique_id < 0 || v.unique_id == excluded_id ||
          v.lane_id != t.target_lane_id || rearEdgeDeltaS(v) > 0.0) {
        return;
      }
      const double gap = rearBumperGap(v);
      if (gap < best_gap) {
        best_gap = gap;
        best = &v;
      }
    };
    consider(t.target_rear);
    // nearby_vehicles remains excluded from general SMPC risk.  Here it is
    // used only to find the next target-lane follower hidden while the first
    // vehicle occupies both target_front and target_rear slots.
    for (const auto& v : t.nearby_vehicles) consider(v);
    return best;
  }

  bool targetLanePairObservedForPassGap(
      const smpc_lane_change::TargetVehicleSet& t,
      const smpc_lane_change::TargetVehicle* rear) const {
    if (!targetFrontIsPassingVehicle(t) ||
        !target_lane_scene_history_.initialized ||
        target_lane_scene_history_.current_lane_id != t.current_lane_id ||
        target_lane_scene_history_.target_lane_id != t.target_lane_id ||
        target_lane_scene_history_.lane_pair_stable_since_stamp.isZero()) {
      return false;
    }
    const ros::Time now = ros::Time::now();
    const double timeout = std::max(0.0, change_commit_observation_gap_timeout_sec_);
    if (timeout > 0.0 &&
        (now - target_lane_scene_history_.last_seen_stamp).toSec() > timeout) {
      return false;
    }
    if ((now - target_lane_scene_history_.lane_pair_stable_since_stamp).toSec() <
        std::max(0.0, pass_gap_lane_pair_observation_sec_)) {
      return false;
    }
    return vehicleObservationStable(t.target_front, now) &&
        (!rear || vehicleObservationStable(*rear, now));
  }

  bool targetLaneSceneObservedForCommit(
      const smpc_lane_change::TargetVehicleSet& t) const {
    if (!target_lane_scene_history_.initialized ||
        target_lane_scene_history_.current_lane_id != t.current_lane_id ||
        target_lane_scene_history_.target_lane_id != t.target_lane_id) {
      return false;
    }
    const ros::Time now = ros::Time::now();
    const double timeout = std::max(0.0, change_commit_observation_gap_timeout_sec_);
    if (!target_lane_scene_history_.last_seen_stamp.isZero() && timeout > 0.0 &&
        (now - target_lane_scene_history_.last_seen_stamp).toSec() > timeout) {
      return false;
    }
    if ((now - target_lane_scene_history_.stable_since_stamp).toSec() <
        std::max(0.0, change_commit_scene_observation_sec_)) {
      return false;
    }
    // An empty target-lane scene must be stable for the same short window;
    // a present representative vehicle additionally needs a stable velocity
    // history before its nominal/accelerating modes can approve CHANGE_NOW.
    return vehicleObservationStable(t.target_front, now) &&
        (!withinSceneObservationRange(t.target_rear) ||
         vehicleObservationStable(t.target_rear, now));
  }

  bool observedTargetLaneIntrusionSafe(
      const smpc_lane_change::TargetVehicleSet& t,
      int passing_vehicle_id = -1) const {
    // This is intentionally not another TV lane-change mode.  The primary TV
    // model remains longitudinal-only for this development stage.  It is a
    // deterministic perception interlock: if a measured nearby rear is
    // already moving into the target corridor during ego's own blend, do not
    // certify an otherwise empty target lane.
    if (!observed_cutin_safety_enabled_) return true;
    const auto* current_lane = laneFor(t.current_lane_id);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!current_lane || !target_lane) return false;
    const auto ego_current = current_lane->project(ego_x_, ego_y_);
    const auto ego_target = target_lane->project(ego_x_, ego_y_);
    if (!ego_current.valid || !ego_target.valid) return false;
    const auto current_center = current_lane->sample(ego_current.s, 0.0);
    const auto target_center = target_lane->sample(ego_target.s, 0.0);
    if (!current_center.valid || !target_center.valid) return false;
    const double target_side_in_current_frame =
        -std::sin(current_center.yaw) * (target_center.x - current_center.x) +
         std::cos(current_center.yaw) * (target_center.y - current_center.y);

    const double max_entry_time = std::max(
        0.0, std::min(observed_cutin_max_entry_time_sec_,
                      egoChangeBlendDuration() + behavior_execution_delay_sec_));
    const double target_half_width = 0.5 * std::max(0.1, lane_width_m_);
    const double target_speed = egoPredictionTargetSpeedMps();

    for (const auto& v : t.nearby_vehicles) {
      if (!v.valid || !roleContains(v, "nearby")) continue;
      if (v.unique_id == passing_vehicle_id &&
          v.lane_id == t.target_lane_id && targetFrontIsPassingVehicle(t) &&
          vehicleObservationStable(v, ros::Time::now())) {
        // This is the already-modelled overtaking vehicle, not the next rear
        // threat.  Its full oriented-box trajectory is evaluated as
        // target_front.  Do not also veto it as nearby_rear while its box
        // straddles ego longitudinally.
        continue;
      }
      // Only a vehicle behind or longitudinally overlapping ego can turn a
      // lane-change start into the fast-rear collision observed in the bag.
      // `front_delta_s > 0` is not sufficient here: an overlap box has its
      // front edge ahead of ego while its rear edge is still behind ego.  It
      // must remain in the rear-intrusion gate until its *rear* edge has
      // cleared ego.
      if (v.rear_delta_s > 0.0) continue;

      const double coarse_rear_gap = std::max(
          0.0, -v.front_delta_s - std::max(0.0, ego_rear_extent_m_));
      // Lane association can switch while a box crosses a lane boundary, so
      // the sign of its target-frame lateral speed is not reliable during
      // that transition.  A close rear box with a measured lateral motion is
      // therefore an ambiguous corridor intrusion, regardless of direction.
      // Wait for it to settle instead of certifying a merge into a gap that
      // can disappear one LiDAR update later.
      // 시나리오상 타 차량이 차선을 바꾸지 않으면 이 분기는 순수한 오검출원이다
      // (GT 대조: 발동 차량의 약 74% 가 자기 차선 직진 중, 2 초간 횡방향 변화 중앙 0.03 m).
      if (!observed_cutin_assume_no_lane_change_ &&
          std::abs(v.v_lat) >=
              std::max(0.0, observed_cutin_min_lateral_speed_mps_) &&
          coarse_rear_gap <= std::max(0.0, observed_cutin_max_rear_distance_m_)) {
        return false;
      }

      // Before lane association flips, the vehicle is still expressed in the
      // current-lane Frenet frame.  Use its footprint, not only its centre:
      // if it already crosses the boundary on the target side, this is a
      // target-corridor occupant even when the fitted lateral velocity is
      // briefly near zero.  Otherwise a wide vehicle can disappear from the
      // target-rear list exactly during the most dangerous frames.
      const bool current_lane_target_side_overlap =
          v.lane_id == t.current_lane_id &&
          std::abs(target_side_in_current_frame) > 1e-3 &&
          (std::copysign(1.0, target_side_in_current_frame) * v.d +
               std::max(0.0, v.half_d) >= target_half_width);
      if (current_lane_target_side_overlap &&
          coarse_rear_gap <= std::max(0.0, observed_cutin_max_rear_distance_m_)) {
        const double closing = std::max(0.0, v.v_long - ego_speed_mps_);
        const double required_gap = std::max(
            activeTargetRearSafeGap(t),
            std::max(0.0, target_rear_min_headway_sec_) *
                closing);
        const double ttc = closing > 0.1
            ? coarse_rear_gap / closing
            : std::numeric_limits<double>::infinity();
        if (coarse_rear_gap <= required_gap ||
            (std::isfinite(ttc) && ttc < std::max(0.0, target_rear_min_ttc_sec_))) {
          return false;
        }
      }

      const auto vehicle_target = target_lane->project(v.x, v.y);
      if (!vehicle_target.valid) continue;
      const auto target_sample = target_lane->sample(vehicle_target.s, 0.0);
      if (!target_sample.valid) continue;

      const double signed_d = vehicle_target.d;
      const double corridor_half_width = target_half_width +
          std::max(0.0, v.half_d);
      const bool occupies_target_corridor =
          std::abs(signed_d) <= corridor_half_width;
      // At a lane boundary the selector can still call this object `nearby`
      // for one or more LiDAR updates.  Once its footprint physically
      // occupies the target corridor, apply the normal rear gap/headway/TTC
      // test immediately, even if its measured lateral speed has just fallen
      // below the cut-in threshold.  This avoids treating the association
      // transition as an empty target-lane gap.
      if (occupies_target_corridor) {
        const double closing = std::max(0.0, v.v_long - ego_speed_mps_);
        const double required_gap = std::max(
            activeTargetRearSafeGap(t),
            std::max(0.0, target_rear_min_headway_sec_) *
                closing);
        const double ttc = closing > 0.1
            ? coarse_rear_gap / closing
            : std::numeric_limits<double>::infinity();
        if (coarse_rear_gap <= required_gap ||
            (std::isfinite(ttc) && ttc < std::max(0.0, target_rear_min_ttc_sec_))) {
          return false;
        }
      }

      // 아래는 "이 차가 목표 차선으로 진입할 것"을 예측하는 분기다.
      // 차선 변경이 없는 시나리오에서는 건너뛴다.  위의 현재 점유 검사(발자국이
      // 이미 목표 코리도/경계를 물고 있는 경우)는 그대로 유지된다.
      if (observed_cutin_assume_no_lane_change_) continue;
      const double source_yaw = laneYawAtVehicle(v);
      const double vx = v.v_long * std::cos(source_yaw) -
          v.v_lat * std::sin(source_yaw);
      const double vy = v.v_long * std::sin(source_yaw) +
          v.v_lat * std::cos(source_yaw);
      const double tv_long = vx * std::cos(target_sample.yaw) +
          vy * std::sin(target_sample.yaw);
      const double tv_lat = -vx * std::sin(target_sample.yaw) +
          vy * std::cos(target_sample.yaw);
      const double lateral_distance = std::max(0.0,
          std::abs(signed_d) - corridor_half_width);
      const double toward_target = -std::copysign(1.0,
          std::abs(signed_d) > 1e-3 ? signed_d : tv_lat) * tv_lat;
      if (toward_target < std::max(0.0, observed_cutin_min_lateral_speed_mps_)) {
        continue;
      }
      const double entry_time = lateral_distance / std::max(1e-3, toward_target);
      if (entry_time > max_entry_time) continue;

      const LongitudinalRollout ego = egoLongitudinalRollout(entry_time, target_speed);
      const double half_s = vehicleHalfS(v);
      const double delta_s = vehicle_target.s - ego_target.s +
          tv_long * entry_time - ego.ds;
      if (delta_s >= 0.0) {
        const double front_gap = delta_s - half_s - ego_front_extent_m_;
        const double required_gap = std::max(
            activeTargetFrontSafeGap(t),
            std::max(0.0, target_front_min_headway_sec_) *
                std::max(ego.v, std::max(0.0, tv_long)));
        if (front_gap <= required_gap) return false;
      } else {
        const double rear_gap = -delta_s - half_s - ego_rear_extent_m_;
        if (rear_gap > std::max(0.0, observed_cutin_max_rear_distance_m_)) continue;
        const double closing = std::max(0.0, tv_long - ego.v);
        const double required_gap = std::max(
            activeTargetRearSafeGap(t),
            std::max(0.0, target_rear_min_headway_sec_) *
                closing);
        const double ttc = closing > 0.1
            ? rear_gap / closing
            : std::numeric_limits<double>::infinity();
        if (rear_gap <= required_gap ||
            (std::isfinite(ttc) && ttc < std::max(0.0, target_rear_min_ttc_sec_))) {
          return false;
        }
      }
    }
    return true;
  }

  bool mergePriorityRearEligible(const smpc_lane_change::TargetVehicleSet& t,
                                 const smpc_lane_change::TargetVehicle& v) const {
    if (!merge_priority_enabled_ || !laneEndPressureActive(t) || !v.valid) return false;
    if (!isRearLikeVehicle(v)) return false;
    // A stopped/slow ego cannot safely claim priority over a fast rear vehicle:
    // it needs time to launch and finish the lateral blend.
    if (lowSpeedRearSafetyActive()) return false;

    const double gap = rearBumperGap(v);
    if (!std::isfinite(gap) || gap < merge_priority_rear_min_gap_m_) return false;

    const double closing = std::max(0.0, v.v_long - ego_speed_mps_);
    const double ttc = closing > 0.1
        ? gap / closing
        : std::numeric_limits<double>::infinity();
    if (closing > merge_priority_rear_max_closing_mps_ &&
        ttc < merge_priority_rear_min_ttc_sec_) {
      return false;
    }

    return true;
  }

  double mergePriorityRiskScale(const smpc_lane_change::TargetVehicleSet* t,
                                const smpc_lane_change::TargetVehicle& v,
                                bool change_left) const {
    if (!t) return 1.0;

    double scale = 1.0;
    if (egoLanePriorityEligible(*t, v, change_left)) {
      scale = std::min(
          scale, std::clamp(ego_lane_priority_side_risk_scale_, 0.0, 1.0));
    }

    if (change_left && mergePriorityRearEligible(*t, v)) {
      if (roleContains(v, "nearby")) {
        scale = std::min(
            scale, std::clamp(merge_priority_nearby_rear_risk_scale_, 0.0, 1.0));
      } else {
        scale = std::min(
            scale, std::clamp(merge_priority_rear_risk_scale_, 0.0, 1.0));
      }
    }
    return scale;
  }

  double effectiveTargetRearSafeGap(const smpc_lane_change::TargetVehicleSet& t,
                                    const smpc_lane_change::TargetVehicle& rear,
                                    double base_gap) const {
    double required_gap = base_gap;
    if (lowSpeedRearSafetyActive()) {
      required_gap = std::max(required_gap, std::max(0.0, low_speed_rear_safe_gap_m_));
    }
    if (!mergePriorityRearEligible(t, rear)) return required_gap;
    return std::min(required_gap, std::max(0.0, merge_priority_rear_safe_gap_m_));
  }

  bool egoLanePriorityEligible(const smpc_lane_change::TargetVehicleSet& t,
                               const smpc_lane_change::TargetVehicle& v,
                               bool change_left) const {
    if (!ego_lane_priority_enabled_ || !v.valid) return false;

    const bool current_front = roleContains(v, "current_front");
    const bool nearby = roleContains(v, "nearby");
    if (!current_front && !nearby) return false;

    // If the vehicle is already very close in front of ego, it is no longer a
    // right-of-way negotiation problem; it is a hard safety constraint.
    const double front_gap = frontBumperGap(v);
    if (std::isfinite(front_gap) &&
        front_gap <= ego_lane_priority_front_hard_gap_m_) {
      return false;
    }

    // A vehicle selected as current_front only because its box touches the
    // current-lane boundary should not make ego yield until its center is
    // actually established near the lane center.
    if (current_front) {
      return std::abs(v.d) > ego_lane_priority_current_max_abs_d_m_;
    }

    // For nearby vehicles, preserve full weight for ego's current lane and for
    // the target lane while ego is actively evaluating a lane change.  Other
    // lanes are treated as yielding side traffic during lane-end pressure.
    if (v.lane_id == t.current_lane_id &&
        std::abs(v.d) <= ego_lane_priority_current_max_abs_d_m_) {
      return false;
    }
    if (v.lane_id == t.current_lane_id &&
        std::abs(v.d) > ego_lane_priority_current_max_abs_d_m_) {
      return true;
    }
    if (change_left && v.lane_id == t.target_lane_id) return false;

    return laneEndPressureActive(t);
  }

  std::vector<smpc_lane_change::TargetVehicle> uniqueVehicles(
      const smpc_lane_change::TargetVehicleSet& t) const {
    std::vector<smpc_lane_change::TargetVehicle> out;
    auto add_unique = [&out](const smpc_lane_change::TargetVehicle& v) {
      if (!v.valid) return;
      bool duplicate = false;
      for (const auto& prev : out) {
        if (v.unique_id >= 0 && prev.unique_id == v.unique_id) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) out.push_back(v);
    };

    add_unique(t.current_front);
    add_unique(t.target_front);
    add_unique(t.target_rear);
    if (risk_include_nearby_vehicles_) {
      for (const auto& v : t.nearby_vehicles) add_unique(v);
    }
    return out;
  }

  double laneYawAtVehicle(const smpc_lane_change::TargetVehicle& v) const {
    const auto* lane = laneFor(v.lane_id);
    if (!lane) return v.yaw;
    const auto sample = lane->sample(v.s, v.d);
    return sample.valid ? sample.yaw : v.yaw;
  }

  bool lateralModeAllowedForVehicle(const smpc_lane_change::TargetVehicle& v) const {
    if (!lateral_mode_enabled_) return false;
    const bool nearby = v.role.find("nearby") != std::string::npos;
    const bool rear = v.role.find("rear") != std::string::npos;
    if (nearby && !lateral_mode_allow_nearby_) return false;
    if (rear && !lateral_mode_allow_rear_) return false;
    if (std::abs(v.d) < lateral_mode_min_abs_d_m_) return false;
    return true;
  }

  // rear_accel 모드의 속도 상한 (2026-09-22).  확률표가 적합된 사건은 "5 s 뒤
  // 등속 예측보다 12.5 m 앞섬"인데, 모드 자체는 상한 없이 가속해 지평선 끝에서
  // 도로 흐름을 크게 넘는 속도가 됐다.  도로 흐름(미션 속도)과 현재 속도 중 큰
  // 값에 여유를 더해 묶는다.  이미 그보다 빠른 TV 는 자기 속도가 하한이 된다.
  double rearAccelMaxSpeedMps(const smpc_lane_change::TargetVehicle& v) const {
    double base = rear_accel_max_speed_mps_ > 0.0 ? rear_accel_max_speed_mps_
                                                  : currentTargetSpeedMps();
    base = std::max(base, std::max(0.0, v.v_long));
    return base + std::max(0.0, rear_accel_max_speed_margin_mps_);
  }

  std::vector<TvMode> modesForVehicle(const smpc_lane_change::TargetVehicle& v) const {
    const bool rear = v.role.find("rear") != std::string::npos;
    const double event_probability = adaptiveEventProbability(v);
    const double event_accel = rear ? rear_accel_mps2_ : front_brake_accel_mps2_;

    std::vector<TvMode> modes;
    modes.push_back({"nominal", 1.0 - event_probability, nominal_target_accel_mps2_, 0.0});
    TvMode event_mode{rear ? "rear_accel" : "front_brake", event_probability,
                      event_accel, 0.0};
    if (rear && rear_accel_max_speed_enabled_) {
      event_mode.max_speed_mps = rearAccelMaxSpeedMps(v);
    }
    modes.push_back(event_mode);

    const double lane_yaw = laneYawAtVehicle(v);
    const double yaw_error = smpc_lane_change::wrapToPi(v.yaw - lane_yaw);
    double inferred_lat_rate = v.v_lat;
    if (lateral_mode_allow_yaw_only_ &&
        std::abs(inferred_lat_rate) < lateral_mode_v_threshold_mps_ &&
        std::abs(yaw_error) > yaw_lateral_threshold_rad_) {
      inferred_lat_rate = v.v_long * std::sin(yaw_error);
    }
    const double yaw_implied_lat_rate = v.v_long * std::sin(yaw_error);

    bool lateral_observed =
        lateralModeAllowedForVehicle(v) &&
        std::abs(inferred_lat_rate) >= lateral_mode_v_threshold_mps_ &&
        std::abs(yaw_error) >= yaw_lateral_threshold_rad_;
    if (lateral_observed && lateral_mode_require_yaw_agreement_ &&
        inferred_lat_rate * yaw_implied_lat_rate <= 0.0) {
      lateral_observed = false;
    }
    if (lateral_observed && lateral_mode_require_away_from_center_ &&
        v.d * inferred_lat_rate <= 0.0) {
      lateral_observed = false;
    }
    if (lateral_observed) {
      const double over_v = std::max(0.0, std::abs(inferred_lat_rate) - lateral_mode_v_threshold_mps_);
      const double over_yaw = std::max(0.0, std::abs(yaw_error) - yaw_lateral_threshold_rad_);
      const double max_p_lat_by_straight_floor =
          std::max(0.0, 1.0 - std::clamp(tv_no_lateral_probability_floor_, 0.0, 1.0));
      const double max_allowed_lateral_probability =
          std::min(max_lateral_mode_probability_, max_p_lat_by_straight_floor);
      const double p_lat = clampProbability(std::min(
          max_allowed_lateral_probability,
          lateral_mode_probability_ + 0.08 * over_v + yaw_lateral_probability_gain_ * over_yaw));
      if (p_lat > 1e-6) {
        for (auto& mode : modes) mode.probability *= (1.0 - p_lat);
        modes.push_back({"observed_lateral", p_lat, nominal_target_accel_mps2_,
                         std::clamp(inferred_lat_rate,
                                    -max_lateral_prediction_mps_, max_lateral_prediction_mps_)});
      }
    }

    const double total = std::max(
        1e-6, std::accumulate(modes.begin(), modes.end(), 0.0,
                              [](double acc, const TvMode& mode) {
                                return acc + std::max(0.0, mode.probability);
                              }));
    for (auto& mode : modes) mode.probability = clampProbability(mode.probability / total);
    return modes;
  }

  std::vector<PredPose> buildEgoTrajectoryFromReferencePreview(
      const smpc_lane_change::TargetVehicleSet& t,
      const ActionSpec& spec) const {
    std::vector<PredPose> traj;
    ego_traj_build_detail_ = "none";
    if (!referencePreviewFreshFor(t)) {
      ego_traj_build_detail_ = "preview_stale";
      return traj;
    }

    const ReferencePreview preview = reference_preview_;
    const auto* current_lane = laneFor(t.current_lane_id);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!current_lane || !target_lane || !spec.change_left) {
      ego_traj_build_detail_ = "no_lane";
      return traj;
    }

    const double source_origin_arc =
        projectPolylineArc(preview.source, ego_x_, ego_y_);
    const double target_origin_arc =
        projectPolylineArc(preview.target, ego_x_, ego_y_);
    const double connector_origin_arc = preview.connector.valid()
        ? projectPolylineArc(preview.connector, ego_x_, ego_y_)
        : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(source_origin_arc) ||
        (preview.mode != waypoint_system::PathSwitchPreview::CONNECTOR &&
         !std::isfinite(target_origin_arc)) ||
        (preview.mode == waypoint_system::PathSwitchPreview::CONNECTOR &&
         !std::isfinite(connector_origin_arc))) {
      ego_traj_build_detail_ = "origin_arc_invalid";
      return traj;
    }

    const double target_speed_mps = egoPredictionTargetSpeedMps();
    const double lane_change_start_sec = effectiveLaneChangeStartSec(spec);
    const LongitudinalRollout pre_switch_rollout = actionLongitudinalRollout(
        lane_change_start_sec, target_speed_mps, spec);
    if (preview.mode == waypoint_system::PathSwitchPreview::CONNECTOR &&
        pre_switch_rollout.ds >
            std::max(0.0, stopped_launch_connector_max_preswitch_distance_m_)) {
      // A connector preview is anchored at the current stopped ego pose.  Do
      // not certify a delayed candidate after it has already travelled away
      // from that anchor; the next 20 Hz preview will provide the new pose.
      return traj;
    }
    if (preview.mode == waypoint_system::PathSwitchPreview::CONNECTOR) {
      const double join_distance = std::max(
          0.0, preview.connector_join_arc_m - connector_origin_arc);
      const double completion_time = connectorTravelTimeSec(
          pre_switch_rollout.v, join_distance, preview.target_lane_id);
      const double horizon_sec = std::max(1, horizon_steps_) * prediction_dt_sec_;
      if (!std::isfinite(completion_time) ||
          lane_change_start_sec + completion_time + behavior_post_merge_buffer_sec_ >
              horizon_sec + 1e-6) {
        // A short target tail is not permission to clamp at its endpoint.
        // Require enough path/time to observe the completed connector and its
        // post-merge buffer before calling the stopped launch safe.
        ego_traj_build_detail_ = "connector_horizon_short";
        return traj;
      }
    }

    double motion_started_at_sec = std::numeric_limits<double>::infinity();
    bool ego_traj_truncated = false;
    traj.reserve(std::max(1, horizon_steps_));
    for (int k = 1; k <= std::max(1, horizon_steps_); ++k) {
      const double tau = prediction_dt_sec_ * static_cast<double>(k);
      const bool after_switch = tau >= lane_change_start_sec - 1e-6;
      const LongitudinalRollout rollout =
          preview.mode == waypoint_system::PathSwitchPreview::CONNECTOR
          ? connectorLaunchRollout(tau, target_speed_mps, spec,
                                   lane_change_start_sec,
                                   preview.target_lane_id)
          : changeAwareRollout(t, tau, target_speed_mps, spec);

      PredPose p;
      if (!after_switch) {
        p = samplePolyline(preview.source, source_origin_arc + rollout.ds);
      } else if (preview.mode == waypoint_system::PathSwitchPreview::CONNECTOR) {
        const double after_switch_distance = std::max(
            0.0, rollout.ds - pre_switch_rollout.ds);
        p = samplePolyline(preview.connector,
                           connector_origin_arc + after_switch_distance);
      } else if (preview.mode == waypoint_system::PathSwitchPreview::DIRECT) {
        p = samplePolyline(preview.target, target_origin_arc + rollout.ds);
      } else if (preview.mode == waypoint_system::PathSwitchPreview::BLEND) {
        const double after_switch_distance = std::max(
            0.0, rollout.ds - pre_switch_rollout.ds);
        if (!std::isfinite(motion_started_at_sec) &&
            (rollout.v >= ego_reference_preview_motion_start_speed_mps_ ||
             after_switch_distance >=
                 ego_reference_preview_motion_start_distance_m_)) {
          motion_started_at_sec = tau;
        }
        const double alpha = std::isfinite(motion_started_at_sec)
            ? std::clamp((tau - motion_started_at_sec) /
                             std::max(0.1, preview.blend_duration_sec *
                                 std::max(0.1,
                                     ego_reference_preview_duration_scale_)),
                         0.0, 1.0)
            : 0.0;
        const ReferencePolyline blended = blendReferencePreview(preview, alpha);
        p = samplePolyline(blended, source_origin_arc + rollout.ds);
      } else {
        ego_traj_build_detail_ = "unknown_preview_mode";
        return {};
      }

      if (!p.valid) {
        // preview 폴리라인 끝을 넘어선 스텝이다.  타깃 차량 예측이
        // target_prediction_truncated_at_lane_end 로 처리하는 것과 동일하게,
        // 여기까지 쌓인 유효 스텝을 살리고 멈춘다.  전체를 버리면 "차선 끝이
        // 가까울수록 차선변경이 필요한데 바로 그 이유로 증명이 불가능해지는"
        // 교착이 생긴다.  합류 완료 + post-merge 버퍼까지 덮지 못한 경우에만
        // 아래에서 후보를 기각한다.
        if (!ego_reference_preview_allow_truncation_) {
          ego_traj_build_detail_ = "off_preview_end";
          return {};
        }
        ego_traj_truncated = true;
        break;
      }
      p.t = tau;
      p.v = rollout.v;
      const auto projection = (after_switch ? target_lane : current_lane)->project(p.x, p.y);
      if (projection.valid) {
        p.s = projection.s;
        p.d = projection.d;
      }
      traj.push_back(p);
    }

    if (traj.empty()) {
      ego_traj_build_detail_ = "no_valid_step";
      return {};
    }

    // 잘린 궤적은 합류가 끝나고 post-merge 버퍼까지 관측된 경우에만 인정한다.
    // 이 구간만 덮으면 그 뒤 스텝이 preview 밖으로 나가도 차선변경의 안전성
    // 판정에는 영향이 없다 (합류 이후는 이미 목표 차선 주행이다).
    if (ego_traj_truncated) {
      const double required_coverage_sec =
          effectiveLaneChangeStartSec(spec) +
          std::max(0.1, spec.lane_change_duration_sec) +
          std::max(0.0, behavior_post_merge_buffer_sec_);
      if (traj.back().t + 1e-6 < required_coverage_sec) {
        ego_traj_build_detail_ = "truncated_before_merge_complete";
        return {};
      }
    }

    if (ego_reference_preview_require_target_completion_) {
      const PredPose& terminal = traj.back();
      const auto terminal_target_projection =
          target_lane->project(terminal.x, terminal.y);
      if (!terminal_target_projection.valid ||
          std::abs(terminal_target_projection.d) >
              std::max(0.0,
                       ego_reference_preview_target_completion_max_abs_d_m_)) {
        // In particular, a stopped BLEND stays on the source path because its
        // motion clock never starts. Returning an empty trajectory makes the
        // CHANGE_LEFT candidate infeasible rather than certifying that state.
        ego_traj_build_detail_ = "terminal_not_on_target_lane";
        return {};
      }
    }
    ego_traj_build_detail_ = ego_traj_truncated ? "truncated_after_merge" : "full";
    return traj;
  }

  std::vector<PredPose> buildEgoTrajectory(const smpc_lane_change::TargetVehicleSet& t,
                                           const ActionSpec& spec) const {
    std::vector<PredPose> traj;
    if (spec.change_left && ego_reference_preview_enabled_) {
      if (referencePreviewFreshFor(t)) {
        return buildEgoTrajectoryFromReferencePreview(t, spec);
      }
      if (ego_reference_preview_required_for_change_) {
        // A brief preview mismatch is expected while waypoint/target epochs
        // resynchronise after completion or recovery.  The empty trajectory
        // still rejects CHANGE_LEFT; keep the expected safety rejection out
        // of the operator WARN stream and expose it through ref_preview=0 in
        // /smpc/smpc_decision_debug.
        ROS_DEBUG_THROTTLE(
            1.0,
            "[smpc_decision] refusing CHANGE_LEFT without a fresh matching "
            "path-switch preview (expected %d -> %d)",
            t.current_lane_id, t.target_lane_id);
        return traj;
      }
    }
    const auto* current_lane = laneFor(t.current_lane_id);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!current_lane || !target_lane) return traj;

    const auto ego_current = current_lane->project(ego_x_, ego_y_);
    const auto ego_target = target_lane->project(ego_x_, ego_y_);
    if (!ego_current.valid || !ego_target.valid) return traj;
    if (spec.change_left && t.target_lane_id == t.current_lane_id) return traj;

    const double ego_rollout_target_speed_mps = egoPredictionTargetSpeedMps();
    const double lane_change_start_sec = effectiveLaneChangeStartSec(spec);
    const double lane_change_duration_sec = spec.lane_change_duration_sec > 1e-3
        ? spec.lane_change_duration_sec
        : egoChangeBlendDuration();
    traj.reserve(std::max(1, horizon_steps_));
    for (int k = 1; k <= std::max(1, horizon_steps_); ++k) {
      const double tau = prediction_dt_sec_ * static_cast<double>(k);
      const LongitudinalRollout rollout =
          changeAwareRollout(t, tau, ego_rollout_target_speed_mps, spec);
      const double predicted_v = rollout.v;
      const double ds = rollout.ds;

      PredPose p;
      p.t = tau;
      p.v = predicted_v;

      if (!spec.change_left) {
        const double d_decay = std::exp(-tau / std::max(0.1, lane_center_decay_tau_sec_));
        const double sample_s = ego_current.s + ds;
        // LanePath::sample intentionally clamps to the CSV endpoint.  A
        // clamped stationary pose with nonzero velocity is not a valid future
        // state, particularly for a deferred merge near the lane end.
        if (sample_s > current_lane->length() + 1e-6) return {};
        const auto sample = current_lane->sample(sample_s, ego_current.d * d_decay);
        if (!sample.valid) return {};
        p.valid = true;
        p.x = sample.x;
        p.y = sample.y;
        p.s = sample.s;
        p.d = sample.d;
        p.yaw = sample.yaw;
      } else {
        const double alpha = egoChangeAlpha(tau, lane_change_start_sec,
                                            lane_change_duration_sec);
        const double d_decay = std::exp(-tau / std::max(0.1, lane_center_decay_tau_sec_));
        // Before t_LC the candidate must remain on the current physical lane,
        // rather than jumping to a centered source-lane sample.
        smpc_lane_change::LaneSample cur;
        smpc_lane_change::LaneSample target;
        if (alpha < 1.0 - 1e-6) {
          const double cur_s = ego_current.s + ds;
          if (cur_s > current_lane->length() + 1e-6) return {};
          const double cur_d = ego_current.d * (1.0 - alpha) * d_decay;
          cur = current_lane->sample(cur_s, cur_d);
          if (!cur.valid) return {};
        }
        if (alpha > 1e-6) {
          const double target_s = ego_target.s + ds;
          if (target_s > target_lane->length() + 1e-6) return {};
          target = target_lane->sample(target_s, 0.0);
          if (!target.valid) return {};
        }
        p.valid = true;
        if (alpha <= 1e-6) {
          p.x = cur.x;
          p.y = cur.y;
          p.s = cur.s;
          p.d = cur.d;
          p.yaw = cur.yaw;
        } else if (alpha >= 1.0 - 1e-6) {
          p.x = target.x;
          p.y = target.y;
          p.s = target.s;
          p.d = target.d;
          p.yaw = target.yaw;
        } else {
          p.x = (1.0 - alpha) * cur.x + alpha * target.x;
          p.y = (1.0 - alpha) * cur.y + alpha * target.y;
          p.s = (1.0 - alpha) * cur.s + alpha * target.s;
          p.d = (1.0 - alpha) * cur.d + alpha * target.d;
          p.yaw = angleLerp(cur.yaw, target.yaw, alpha);
        }
      }
      traj.push_back(p);
    }
    return traj;
  }

  // extra_ds: 자차가 이 모드에 반응해 감속한 만큼을 TV 를 전방으로 밀어 표현한다.
  // 평행 차선에서는 자차를 뒤로 물리는 것과 상대 기하가 같고, TV 를 자기 차선
  // 위에 그대로 두므로 횡방향 판정이 왜곡되지 않는다.  TV 속도는 바뀌지 않는다.
  PredPose predictTargetPose(const smpc_lane_change::TargetVehicle& v,
                             const TvMode& mode,
                             double tau,
                             bool* exceeded_lane_end = nullptr,
                             double extra_ds = 0.0) const {
    PredPose p;
    if (exceeded_lane_end != nullptr) *exceeded_lane_end = false;
    const auto* lane = laneFor(v.lane_id);
    if (!lane) return p;

    const double predicted_v_long = modeSpeed(v.v_long, mode, tau);
    const double ds = modeDistance(v.v_long, mode, tau) + std::max(0.0, extra_ds);
    double d = v.d;
    if (std::abs(mode.lateral_rate_mps) > 1e-3) {
      d += mode.lateral_rate_mps * tau;
    } else {
      d *= std::exp(-tau / std::max(0.1, lane_center_decay_tau_sec_));
    }
    d = std::clamp(d, -1.25 * lane_width_m_, 1.25 * lane_width_m_);

    const double sample_s = v.s + ds;
    if (sample_s > lane->length() + 1e-6) {
      // The prediction is still well-defined in time, but this TV has left
      // the finite CSV coverage.  Callers may finish this TV mode's risk
      // trace at its last valid, time-matched step without invalidating the
      // ego trajectory or the other vehicles' traces.
      if (exceeded_lane_end != nullptr) *exceeded_lane_end = true;
      return p;
    }
    const auto sample = lane->sampleExtended(
        sample_s, d, tv_lane_projection_extension_before_m_);
    if (!sample.valid) return p;

    const double yaw_from_frenet_vel =
        smpc_lane_change::wrapToPi(sample.yaw +
                                  std::atan2(mode.lateral_rate_mps,
                                             std::max(0.1, predicted_v_long)));
    const double yaw_rate_limited = std::clamp(
        v.yaw_rate, -max_abs_tv_yaw_rate_, max_abs_tv_yaw_rate_);
    const double yaw_from_rate = smpc_lane_change::wrapToPi(
        v.yaw + yaw_rate_limited * std::min(tau, tv_yaw_rate_horizon_sec_));

    p.valid = true;
    p.t = tau;
    p.x = sample.x;
    p.y = sample.y;
    p.s = sample.s;
    p.d = sample.d;
    p.v = predicted_v_long;
    const double dynamic_yaw = angleLerp(yaw_from_frenet_vel, yaw_from_rate,
                                        std::clamp(tv_yaw_rate_weight_, 0.0, 1.0));
    p.yaw = angleLerp(dynamic_yaw, sample.yaw,
                      std::clamp(tv_csv_yaw_weight_, 0.0, 1.0));
    return p;
  }

  static double axisExtent(const OrientedBox& box, double ax, double ay) {
    const double ux = std::cos(box.yaw);
    const double uy = std::sin(box.yaw);
    const double vx = -std::sin(box.yaw);
    const double vy = std::cos(box.yaw);
    return box.half_l * std::abs(ux * ax + uy * ay) +
           box.half_w * std::abs(vx * ax + vy * ay);
  }

  OrientedBox egoOrientedBox(const PredPose& ego) const {
    const double centre_offset =
        0.5 * (ego_front_extent_m_ - ego_rear_extent_m_);
    return OrientedBox{
        ego.x + centre_offset * std::cos(ego.yaw),
        ego.y + centre_offset * std::sin(ego.yaw), ego.yaw,
        0.5 * (ego_front_extent_m_ + ego_rear_extent_m_),
        0.5 * ego_width_m_};
  }

  CollisionEval collisionProbability(const PredPose& ego,
                                     const PredPose& tv,
                                     const smpc_lane_change::TargetVehicle& vehicle,
                                     const TvMode& mode) const {
    CollisionEval out;
    const OrientedBox ego_box = egoOrientedBox(ego);
    const OrientedBox tv_box{
        tv.x, tv.y, tv.yaw, 0.5 * std::max(0.1, vehicle.length),
        0.5 * std::max(0.1, vehicle.width)};

    const std::array<std::array<double, 2>, 4> axes{{
        {std::cos(ego_box.yaw), std::sin(ego_box.yaw)},
        {-std::sin(ego_box.yaw), std::cos(ego_box.yaw)},
        {std::cos(tv_box.yaw), std::sin(tv_box.yaw)},
        {-std::sin(tv_box.yaw), std::cos(tv_box.yaw)},
    }};

    const double dx = tv_box.x - ego_box.x;
    const double dy = tv_box.y - ego_box.y;
    double sigma_base = prediction_sigma_base_m_;
    double sigma_growth = prediction_sigma_growth_mps_;
    if (mode.name == "nominal") {
      sigma_base = prediction_nominal_sigma_base_m_;
      sigma_growth = prediction_nominal_sigma_growth_mps_;
    } else if (mode.name == "front_brake") {
      sigma_base = prediction_front_brake_sigma_base_m_;
      sigma_growth = prediction_front_brake_sigma_growth_mps_;
    } else if (mode.name == "rear_accel") {
      sigma_base = prediction_rear_accel_sigma_base_m_;
      sigma_growth = prediction_rear_accel_sigma_growth_mps_;
    }
    const double sigma_long = std::max(
        1e-3, sigma_base + sigma_growth * ego.t);
    const double sigma_lat = std::max(
        1e-3, prediction_lateral_sigma_base_m_ +
                   prediction_lateral_sigma_growth_mps_ * ego.t);

    double collision_prob = 1.0;
    double best_separating_gap = -std::numeric_limits<double>::infinity();
    for (const auto& axis : axes) {
      const double ax = axis[0];
      const double ay = axis[1];
      const double center_distance = std::abs(dx * ax + dy * ay);
      const double gap = center_distance -
          axisExtent(ego_box, ax, ay) - axisExtent(tv_box, ax, ay);
      best_separating_gap = std::max(best_separating_gap, gap);

      const double ego_long_axis_alignment =
          std::abs(std::cos(ego_box.yaw) * ax + std::sin(ego_box.yaw) * ay);
      const double tv_long_axis_alignment =
          std::abs(std::cos(tv_box.yaw) * ax + std::sin(tv_box.yaw) * ay);
      const double longness = std::max(ego_long_axis_alignment, tv_long_axis_alignment);
      const double sigma = longness * sigma_long + (1.0 - longness) * sigma_lat;
      const double p_axis_overlap = normalCdf((collision_margin_m_ - gap) / sigma);
      collision_prob = std::min(collision_prob, p_axis_overlap);
    }

    out.probability = clampProbability(collision_prob);
    out.signed_clearance = best_separating_gap - collision_margin_m_;
    return out;
  }

  TrajectoryEval evaluateTrajectory(
      const std::vector<PredPose>& ego_traj,
      const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
      const smpc_lane_change::TargetVehicleSet* merge_context = nullptr,
      bool change_left = false) const {
    TrajectoryEval out;
    if (ego_traj.empty()) {
      out.detail = "empty_ego_traj";
      return out;
    }

    out.valid = true;
    const std::size_t steps = ego_traj.size();
    std::vector<double> no_collision_probability_by_step(steps, 1.0);
    out.risk_by_step.assign(steps, 0.0);
    out.min_signed_clearance_by_step.assign(
        steps, std::numeric_limits<double>::infinity());

    // Retain the legacy result in out.risk while the time-indexed gate is
    // calibrated.  The new result is out.risk_by_step / peak_step_risk.
    double legacy_no_collision_probability = 1.0;

    for (const auto& vehicle : vehicles) {
      const auto modes = modesForVehicle(vehicle);
      VehicleRiskTrace vehicle_trace;
      vehicle_trace.unique_id = vehicle.unique_id;
      vehicle_trace.role = vehicle.role;
      vehicle_trace.priority_scale =
          mergePriorityRiskScale(merge_context, vehicle, change_left);
      vehicle_trace.mixed_risk_by_step.assign(steps, 0.0);

      double legacy_expected_vehicle_risk = 0.0;

      // 자차 종방향 반응.  자차 롤아웃은 ACC 의 "현재" 명령을 개루프로 8 초
      // 연장한 것이라, 앞차가 제동하는 모드에서도 자차가 끝까지 가속한다.
      // lc_gt 14-18 t=23.8s: 앞차 27.32 m/s 가 -2.5 m/s^2 로 감속하면 0.93 초
      // 뒤부터 자차(25.0)보다 느려지는데도 자차는 8 초 내내 25.0 을 유지해
      // t=2.7s 에 겹치고, 제동 모드 확률 0.406 이 위험도로 거의 그대로 넘어왔다
      // (peak_step_risk 0.426, risk_epsilon 0.04).  전방 차량이 자차보다 느려지면
      // ACC 가 실제로 감속한다.  이것은 상대 차량의 의도(뒤차 양보 등)가 아니라
      // 자차 제어기의 확정된 동작이므로 모델에 넣는다.  단, 반응 지연과 감속
      // 한계를 ACC 실제값(braking_decel 3.8)보다 보수적으로 잡는다.
      const bool react_to_this_vehicle =
          ego_brake_reaction_enabled_ &&
          vehicle.role.find("front") != std::string::npos;

      for (const auto& mode : modes) {
        ModeRiskTrace mode_trace;
        mode_trace.name = mode.name;
        mode_trace.probability = clampProbability(mode.probability);
        mode_trace.conditional_risk_by_step.assign(steps, 0.0);
        double legacy_mode_risk = 0.0;

        double ego_react_v = std::max(0.0, ego_speed_mps_);
        double ego_react_lag_ds = 0.0;
        double reaction_trigger_t = std::numeric_limits<double>::infinity();

        for (std::size_t k = 0; k < steps; ++k) {
          const auto& ego = ego_traj[k];
          if (react_to_this_vehicle) {
            const double tv_v = modeSpeed(vehicle.v_long, mode, ego.t);
            if (!std::isfinite(reaction_trigger_t) && tv_v < ego.v) {
              reaction_trigger_t = ego.t;
            }
            const bool reacting = std::isfinite(reaction_trigger_t) &&
                ego.t >= reaction_trigger_t +
                    std::max(0.0, ego_brake_reaction_delay_sec_);
            if (reacting) {
              ego_react_v = std::max(
                  tv_v,
                  std::min(ego.v,
                           ego_react_v -
                               std::max(0.0, ego_brake_reaction_max_decel_mps2_) *
                                   prediction_dt_sec_));
            } else {
              ego_react_v = ego.v;
            }
            ego_react_lag_ds +=
                std::max(0.0, ego.v - ego_react_v) * prediction_dt_sec_;
          }
          bool exceeded_lane_end = false;
          const auto tv = predictTargetPose(
              vehicle, mode, ego.t, &exceeded_lane_end, ego_react_lag_ds);
          if (exceeded_lane_end) {
            // Compare EV[k] only with TV[k].  Once this mode has progressed
            // beyond the downstream CSV endpoint, retain all risk already
            // accumulated through k-1 and stop evaluating this mode.  Other
            // modes and vehicles continue over their own valid intervals.
            if (out.detail.empty()) {
              out.detail = "target_prediction_truncated_at_lane_end";
            }
            break;
          }
          if (!tv.valid) {
            out.valid = false;
            out.detail = "target_prediction_invalid_before_lane_end";
            return out;
          }
          const auto collision = collisionProbability(ego, tv, vehicle, mode);
          const double conditional_risk = clampProbability(collision.probability);
          mode_trace.conditional_risk_by_step[k] = conditional_risk;
          vehicle_trace.mixed_risk_by_step[k] +=
              mode_trace.probability * conditional_risk;
          legacy_mode_risk = std::max(legacy_mode_risk, conditional_risk);
          out.min_signed_clearance = std::min(out.min_signed_clearance,
                                              collision.signed_clearance);
          out.min_signed_clearance_by_step[k] = std::min(
              out.min_signed_clearance_by_step[k], collision.signed_clearance);
        }
        legacy_expected_vehicle_risk += mode_trace.probability * legacy_mode_risk;
        vehicle_trace.modes.push_back(std::move(mode_trace));
      }

      legacy_expected_vehicle_risk = clampProbability(
          legacy_expected_vehicle_risk * vehicle_trace.priority_scale);
      legacy_no_collision_probability *= (1.0 - legacy_expected_vehicle_risk);
      for (std::size_t k = 0; k < steps; ++k) {
        vehicle_trace.mixed_risk_by_step[k] = clampProbability(
            vehicle_trace.mixed_risk_by_step[k] * vehicle_trace.priority_scale);
        no_collision_probability_by_step[k] *=
            (1.0 - vehicle_trace.mixed_risk_by_step[k]);
      }
      out.vehicle_risk_traces.push_back(std::move(vehicle_trace));
    }

    out.risk = clampProbability(1.0 - legacy_no_collision_probability);
    out.peak_step_risk = 0.0;
    out.peak_step = 0;
    for (std::size_t k = 0; k < steps; ++k) {
      out.risk_by_step[k] = clampProbability(1.0 - no_collision_probability_by_step[k]);
      if (out.risk_by_step[k] > out.peak_step_risk) {
        out.peak_step_risk = out.risk_by_step[k];
        out.peak_step = k;
      }
    }
    if (!std::isfinite(out.min_signed_clearance)) {
      out.min_signed_clearance = std::numeric_limits<double>::infinity();
    }
    return out;
  }

  void publishPredictionPath(const std::vector<PredPose>& traj,
                             const ros::Publisher& pub,
                             const std_msgs::Header& source_header) const {
    nav_msgs::Path path;
    path.header = source_header;
    path.header.frame_id = "map";
    path.header.stamp = ros::Time::now();
    path.poses.reserve(traj.size());
    for (const auto& p : traj) {
      if (!p.valid) continue;
      geometry_msgs::PoseStamped pose;
      pose.header = path.header;
      pose.pose.position.x = p.x;
      pose.pose.position.y = p.y;
      pose.pose.position.z = 0.0;
      pose.pose.orientation.z = std::sin(0.5 * p.yaw);
      pose.pose.orientation.w = std::cos(0.5 * p.yaw);
      path.poses.push_back(pose);
    }
    pub.publish(path);
  }

  static geometry_msgs::Point markerPoint(double x, double y, double z) {
    geometry_msgs::Point p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
  }

  static void setMarkerColor(visualization_msgs::Marker& marker,
                             double r, double g, double b, double a) {
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = a;
  }

  ros::Duration predictionMarkerLifetime() const {
    return ros::Duration(std::max(0.0, debug_prediction_marker_lifetime_sec_));
  }

  std_msgs::Header predictionMarkerHeader(const std_msgs::Header& source_header) const {
    std_msgs::Header header = source_header;
    header.frame_id = "map";
    header.stamp = ros::Time::now();
    return header;
  }

  void appendTrajectoryLine(visualization_msgs::MarkerArray& array,
                            const std_msgs::Header& header,
                            const std::string& ns,
                            int id,
                            const std::vector<PredPose>& traj,
                            double r,
                            double g,
                            double b,
                            double a,
                            double line_width_scale = 1.0) const {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::LINE_STRIP;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = std::max(0.02, debug_prediction_line_width_m_ * line_width_scale);
    marker.lifetime = predictionMarkerLifetime();
    setMarkerColor(marker, r, g, b, a);
    for (const auto& p : traj) {
      if (!p.valid) continue;
      marker.points.push_back(markerPoint(p.x, p.y, debug_prediction_z_m_));
    }
    if (marker.points.size() >= 2) array.markers.push_back(marker);
  }

  void appendFootprints(visualization_msgs::MarkerArray& array,
                        const std_msgs::Header& header,
                        const std::string& ns,
                        int base_id,
                        const std::vector<PredPose>& traj,
                        double length,
                        double width,
                        int stride,
                        double r,
                        double g,
                        double b,
                        double a,
                        double longitudinal_center_offset = 0.0) const {
    const int safe_stride = std::max(1, stride);
    int marker_index = 0;
    for (std::size_t i = 0; i < traj.size(); i += static_cast<std::size_t>(safe_stride)) {
      const auto& p = traj[i];
      if (!p.valid) continue;
      visualization_msgs::Marker marker;
      marker.header = header;
      marker.ns = ns;
      marker.id = base_id + marker_index;
      marker.type = visualization_msgs::Marker::CUBE;
      marker.action = visualization_msgs::Marker::ADD;
      marker.pose.position.x =
          p.x + longitudinal_center_offset * std::cos(p.yaw);
      marker.pose.position.y =
          p.y + longitudinal_center_offset * std::sin(p.yaw);
      marker.pose.position.z = debug_prediction_z_m_ + 0.5 * debug_prediction_footprint_height_m_;
      marker.pose.orientation.z = std::sin(0.5 * p.yaw);
      marker.pose.orientation.w = std::cos(0.5 * p.yaw);
      marker.scale.x = std::max(0.1, length);
      marker.scale.y = std::max(0.1, width);
      marker.scale.z = std::max(0.02, debug_prediction_footprint_height_m_);
      marker.lifetime = predictionMarkerLifetime();
      setMarkerColor(marker, r, g, b, a);
      array.markers.push_back(marker);
      ++marker_index;
    }
  }

  void appendPredictionLabel(visualization_msgs::MarkerArray& array,
                             const std_msgs::Header& header,
                             const std::string& ns,
                             int id,
                             const PredPose& pose,
                             const std::string& text,
                             double r,
                             double g,
                             double b,
                             double a) const {
    if (!debug_prediction_show_labels_ || !pose.valid) return;
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = pose.x;
    marker.pose.position.y = pose.y;
    marker.pose.position.z = debug_prediction_z_m_ + 1.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.75;
    marker.text = text;
    marker.lifetime = predictionMarkerLifetime();
    setMarkerColor(marker, r, g, b, a);
    array.markers.push_back(marker);
  }

  static bool lastValidPose(const std::vector<PredPose>& traj, PredPose& out) {
    for (auto it = traj.rbegin(); it != traj.rend(); ++it) {
      if (it->valid) {
        out = *it;
        return true;
      }
    }
    return false;
  }

  void targetModeColor(const TvMode& mode,
                       double& r,
                       double& g,
                       double& b,
                       double& a) const {
    a = 0.85;
    if (mode.name == "nominal") {
      r = 0.20;
      g = 1.00;
      b = 0.75;
    } else if (mode.name.find("brake") != std::string::npos) {
      r = 1.00;
      g = 0.35;
      b = 0.20;
    } else if (mode.name.find("accel") != std::string::npos) {
      r = 0.25;
      g = 0.55;
      b = 1.00;
    } else {
      r = 0.95;
      g = 0.45;
      b = 1.00;
    }
  }

  std::vector<PredPose> buildTargetTrajectory(const smpc_lane_change::TargetVehicle& vehicle,
                                              const TvMode& mode) const {
    std::vector<PredPose> traj;
    traj.reserve(std::max(1, horizon_steps_));
    for (int k = 1; k <= std::max(1, horizon_steps_); ++k) {
      const double tau = prediction_dt_sec_ * static_cast<double>(k);
      const PredPose pose = predictTargetPose(vehicle, mode, tau);
      if (pose.valid) traj.push_back(pose);
    }
    return traj;
  }

  void appendTargetPredictionMarkers(
      visualization_msgs::MarkerArray& array,
      const std_msgs::Header& header,
      const std::vector<smpc_lane_change::TargetVehicle>& vehicles) const {
    const int max_vehicles = std::max(0, debug_prediction_target_max_vehicles_);
    const int vehicle_count = std::min(static_cast<int>(vehicles.size()), max_vehicles);
    for (int vehicle_index = 0; vehicle_index < vehicle_count; ++vehicle_index) {
      const auto& vehicle = vehicles[vehicle_index];
      const auto modes = modesForVehicle(vehicle);
      for (std::size_t mode_index = 0; mode_index < modes.size(); ++mode_index) {
        if (!debug_prediction_show_target_modes_ && mode_index > 0) continue;
        const auto& mode = modes[mode_index];
        const auto traj = buildTargetTrajectory(vehicle, mode);
        double r = 0.0, g = 1.0, b = 0.0, a = 0.85;
        targetModeColor(mode, r, g, b, a);
        if (mode_index > 0) a *= 0.75;

        const int marker_base = vehicle_index * 20 + static_cast<int>(mode_index);
        appendTrajectoryLine(array, header, "smpc_tv_prediction_lines", marker_base,
                             traj, r, g, b, a, mode_index == 0 ? 1.0 : 0.65);

        if (mode_index == 0) {
          appendFootprints(array, header, "smpc_tv_prediction_footprints",
                           vehicle_index * 100, traj, vehicle.length, vehicle.width,
                           debug_prediction_target_footprint_stride_, r, g, b, 0.22);
        }

        PredPose label_pose;
        if (lastValidPose(traj, label_pose)) {
          std::ostringstream ss;
          ss.setf(std::ios::fixed);
          ss.precision(2);
          ss << "TV " << vehicle.unique_id << " " << mode.name
             << "\np=" << mode.probability;
          appendPredictionLabel(array, header, "smpc_tv_prediction_labels", marker_base,
                                label_pose, ss.str(), r, g, b, 0.95);
        }
      }
    }
  }

  void publishPredictionMarkers(const smpc_lane_change::TargetVehicleSet& t,
                                const std::vector<PredPose>& keep_traj,
                                const std::vector<PredPose>& change_traj,
                                const std::vector<smpc_lane_change::TargetVehicle>& vehicles) const {
    if (!debug_prediction_publish_markers_) return;

    visualization_msgs::MarkerArray array;
    const auto header = predictionMarkerHeader(t.header);

    visualization_msgs::Marker clear;
    clear.header = header;
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);

    appendTrajectoryLine(array, header, "smpc_ego_keep_prediction_line", 0,
                         keep_traj, 0.20, 0.75, 1.00, 0.95, 1.1);
    appendTrajectoryLine(array, header, "smpc_ego_change_prediction_line", 0,
                         change_traj, 1.00, 0.75, 0.10, 0.95, 1.4);
    appendFootprints(array, header, "smpc_ego_keep_prediction_footprints", 0,
                     keep_traj, ego_length_m_, ego_width_m_,
                     debug_prediction_ego_footprint_stride_, 0.20, 0.75, 1.00, 0.16,
                     0.5 * (ego_front_extent_m_ - ego_rear_extent_m_));
    appendFootprints(array, header, "smpc_ego_change_prediction_footprints", 0,
                     change_traj, ego_length_m_, ego_width_m_,
                     debug_prediction_ego_footprint_stride_, 1.00, 0.75, 0.10, 0.20,
                     0.5 * (ego_front_extent_m_ - ego_rear_extent_m_));

    PredPose keep_end;
    if (lastValidPose(keep_traj, keep_end)) {
      appendPredictionLabel(array, header, "smpc_ego_prediction_labels", 0,
                            keep_end, "ego keep", 0.20, 0.75, 1.00, 0.95);
    }
    PredPose change_end;
    if (lastValidPose(change_traj, change_end)) {
      appendPredictionLabel(array, header, "smpc_ego_prediction_labels", 1,
                            change_end, "ego change", 1.00, 0.75, 0.10, 0.95);
    }

    appendTargetPredictionMarkers(array, header, vehicles);
    prediction_marker_pub_.publish(array);
  }

  static const char* behaviorActionName(BehaviorAction action) {
    switch (action) {
      case BehaviorAction::kKeep: return "KEEP";
      case BehaviorAction::kWaitForGap: return "WAIT_FOR_GAP";
      case BehaviorAction::kChangeNow: return "CHANGE_NOW";
      case BehaviorAction::kYieldDecelThenChange: return "YIELD_DECEL_THEN_CHANGE";
      case BehaviorAction::kHoldThenChange: return "HOLD_THEN_CHANGE";
      case BehaviorAction::kAccelThenChange: return "ACCEL_THEN_CHANGE";
    }
    return "KEEP";
  }

  static uint8_t behaviorPlanActionMessageValue(BehaviorAction action) {
    switch (action) {
      case BehaviorAction::kWaitForGap:
        return smpc_lane_change::BehaviorPlan::WAIT_FOR_GAP;
      case BehaviorAction::kChangeNow:
        return smpc_lane_change::BehaviorPlan::CHANGE_NOW;
      case BehaviorAction::kYieldDecelThenChange:
        return smpc_lane_change::BehaviorPlan::YIELD_DECEL_THEN_CHANGE;
      case BehaviorAction::kHoldThenChange:
        return smpc_lane_change::BehaviorPlan::HOLD_THEN_CHANGE;
      case BehaviorAction::kAccelThenChange:
        return smpc_lane_change::BehaviorPlan::ACCEL_THEN_CHANGE;
      case BehaviorAction::kKeep:
      default:
        return smpc_lane_change::BehaviorPlan::KEEP;
    }
  }

  static uint8_t behaviorRequestActionMessageValue(BehaviorAction action) {
    switch (action) {
      case BehaviorAction::kWaitForGap:
        return smpc_lane_change::BehaviorLongitudinalRequest::WAIT_FOR_GAP;
      case BehaviorAction::kYieldDecelThenChange:
        return smpc_lane_change::BehaviorLongitudinalRequest::YIELD;
      case BehaviorAction::kHoldThenChange:
        return smpc_lane_change::BehaviorLongitudinalRequest::HOLD;
      case BehaviorAction::kAccelThenChange:
        return smpc_lane_change::BehaviorLongitudinalRequest::ACCELERATE;
      case BehaviorAction::kKeep:
      case BehaviorAction::kChangeNow:
      default:
        return smpc_lane_change::BehaviorLongitudinalRequest::NONE;
    }
  }

  bool candidateFitsHorizon(const ActionSpec& spec) const {
    if (!spec.change_left) return true;
    const double required = effectiveLaneChangeStartSec(spec) +
        std::max(0.1, spec.lane_change_duration_sec) +
        std::max(0.0, behavior_post_merge_buffer_sec_);
    const double available = std::max(1, horizon_steps_) * prediction_dt_sec_;
    return required <= available + 1e-6;
  }

  bool rearAccelerationOpportunity(
      const smpc_lane_change::TargetVehicleSet& t) const {
    if (!behavior_execute_accel_ || !t.target_rear.valid) return false;
    const double gap = std::max(0.0, rearBumperGap(t.target_rear));
    const double closing = conservativeRearClosingSpeed(
        t.target_rear, ego_speed_mps_, ros::Time::now());
    if (closing < std::max(0.0, behavior_rear_accel_min_closing_mps_)) {
      return false;
    }
    const double ttc = gap / std::max(0.1, closing);
    // Acceleration is an escape/preparation option only while there is still
    // time to create a gap.  A close rear is WAIT/abort territory, never a
    // reason to force the vehicle into the target lane.
    return gap > activeTargetRearSafeGap(t) &&
        ttc > std::max(0.0, behavior_rear_accel_emergency_ttc_sec_);
  }

  double rearAwarePreparationAccel(
      const smpc_lane_change::TargetVehicleSet& t) const {
    if (!t.target_rear.valid) return behavior_accel_accel_mps2_;
    const double closing = conservativeRearClosingSpeed(
        t.target_rear, ego_speed_mps_, ros::Time::now());
    return std::clamp(
        std::max(0.0, behavior_rear_accel_gain_) * closing,
        std::max(0.0, behavior_rear_accel_min_mps2_),
        std::max(behavior_rear_accel_min_mps2_, behavior_accel_accel_mps2_));
  }

  std::vector<ActionSpec> buildBehaviorCandidates(
      const smpc_lane_change::TargetVehicleSet& t) const {
    const double duration = egoChangeBlendDuration();
    std::vector<ActionSpec> out;
    out.push_back({BehaviorAction::kChangeNow, 0.0, 0.0, duration, true, "change_now"});
    if (!behavior_planner_enabled_) return out;

    for (const double start_sec : behavior_lc_start_times_sec_) {
      out.push_back({BehaviorAction::kHoldThenChange,
                     behavior_hold_accel_mps2_, start_sec, duration, true,
                     "hold_then_change"});
      // Evaluate the lightest comfortable yield first.  The lexicographic
      // policy still gives every valid HOLD candidate priority over YIELD;
      // within YIELD, cost selects the smallest deceleration that makes a
      // future lane change safe.
      for (const double decel : behavior_yield_decel_candidates_mps2_) {
        out.push_back({BehaviorAction::kYieldDecelThenChange,
                       decel, start_sec, duration, true,
                       "yield_decel_then_change"});
      }
      // Positive-acceleration candidates must not win the action selection
      // while their ACC execution path is intentionally disabled.  Otherwise
      // a non-executed shadow action can hide a real YIELD/HOLD candidate.
      if (rearAccelerationOpportunity(t)) {
        out.push_back({BehaviorAction::kAccelThenChange,
                       rearAwarePreparationAccel(t), start_sec, duration, true,
                       "accel_then_change"});
      }
    }
    return out;
  }

  bool timeIndexedRiskSafe(const TrajectoryEval& eval, double allowed_risk) const {
    if (!eval.valid || eval.risk_by_step.empty()) return false;
    return std::all_of(eval.risk_by_step.begin(), eval.risk_by_step.end(),
                       [allowed_risk](double risk) {
                         return std::isfinite(risk) && risk <= allowed_risk;
                       });
  }

  bool clearanceSafe(const TrajectoryEval& eval) const {
    if (!eval.valid) return false;
    return std::all_of(eval.min_signed_clearance_by_step.begin(),
                       eval.min_signed_clearance_by_step.end(),
                       [this](double clearance) {
                         // +inf means no relevant vehicle contributed a
                         // clearance at that step.  NaN/-inf are invalid
                         // numerical states and must fail closed.
                         return (std::isinf(clearance) && clearance > 0.0) ||
                             (std::isfinite(clearance) &&
                              clearance >= behavior_min_signed_clearance_m_);
                       });
  }

  double firstRiskViolationTimeSec(const TrajectoryEval& eval,
                                   double allowed_risk) const {
    for (std::size_t k = 0; k < eval.risk_by_step.size(); ++k) {
      if (eval.risk_by_step[k] > allowed_risk) {
        return prediction_dt_sec_ * static_cast<double>(k + 1);
      }
    }
    return -1.0;
  }

  bool behaviorStatusConfirmsLastIssuedPreparation(
      const smpc_lane_change::TargetVehicleSet& t) const {
    // This acknowledgement identifies the already-applied command that
    // explains a lower current ACC baseline.  It deliberately does not try
    // to pre-approve the candidate currently being evaluated: that candidate
    // has not been published yet and may receive a new command revision.
    if (!have_behavior_status_ || last_behavior_plan_id_ == 0 ||
        last_behavior_command_revision_ == 0 ||
        !last_behavior_command_active_) {
      return false;
    }
    const ros::Time now = ros::Time::now();
    if (behavior_status_timeout_sec_ > 0.0 &&
        (now - behavior_status_stamp_).toSec() > behavior_status_timeout_sec_) {
      return false;
    }
    if (behavior_status_.header.stamp.isZero() ||
        behavior_status_.header.stamp > now + ros::Duration(0.05) ||
        (behavior_status_timeout_sec_ > 0.0 &&
         (now - behavior_status_.header.stamp).toSec() > behavior_status_timeout_sec_)) {
      return false;
    }
    constexpr double kSpeedCapMatchToleranceMps = 0.10;
    const auto matches_issued_command =
        [&](uint32_t plan_id, uint32_t command_revision, bool command_active,
            double command_speed_cap_mps) {
          return command_active && plan_id == last_behavior_plan_id_ &&
              plan_id != 0 && command_revision != 0 &&
              behavior_status_.plan_id == plan_id &&
              behavior_status_.command_revision == command_revision &&
              std::isfinite(command_speed_cap_mps) &&
              std::abs(behavior_status_.applied_speed_cap_mps -
                       command_speed_cap_mps) <= kSpeedCapMatchToleranceMps;
        };
    return behavior_status_.behavior_cap_applied &&
        !behavior_status_.external_speed_limit_active &&
        behavior_status_.current_lane_id == t.current_lane_id &&
        behavior_status_.target_lane_id == t.target_lane_id &&
        std::isfinite(behavior_status_.applied_speed_cap_mps) &&
        (matches_issued_command(last_behavior_command_plan_id_,
                                last_behavior_command_revision_,
                                last_behavior_command_active_,
                                last_behavior_command_speed_cap_mps_) ||
         // Decision and ACC timers are asynchronous.  A fresh status can
         // legitimately describe the immediately preceding material command
         // for one cycle while ACC is consuming the latest request.
         matches_issued_command(previous_behavior_command_plan_id_,
                                previous_behavior_command_revision_,
                                previous_behavior_command_active_,
                                previous_behavior_command_speed_cap_mps_));
  }

  bool preparationLongitudinalModelCompatible(
      const smpc_lane_change::TargetVehicleSet& t,
      const ActionSpec& spec,
      double base_speed_cap_mps) const {
    if (spec.action == BehaviorAction::kKeep ||
        spec.action == BehaviorAction::kWaitForGap ||
        spec.action == BehaviorAction::kChangeNow) {
      return true;
    }

    // A preparation action owns only the extra longitudinal cap.  If normal
    // ACC is already asking ego to decelerate, its lead/lane-end logic may
    // legitimately use a stronger brake rate than the behavior request.
    // Do not execute a future candidate whose model cannot represent that
    // safety override; it remains visible as ordinary KEEP/replanning.
    constexpr double kCompatibilityToleranceMps = 0.10;
    if (!std::isfinite(base_speed_cap_mps)) return false;
    if (base_speed_cap_mps + kCompatibilityToleranceMps >= ego_speed_mps_) {
      return true;
    }

    // A lower ACC target is normally an external lead/lane-end brake and
    // makes this simple behavior rollout incompatible.  The only exception
    // is an explicit fresh acknowledgement that the lower target is this
    // same plan's rate-limited behavior cap; without this handshake, a
    // renewed YIELD would oscillate between selected and cancelled.
    return behaviorStatusConfirmsLastIssuedPreparation(t);
  }

  bool projectedTargetGapTtcSafe(
      const smpc_lane_change::TargetVehicleSet& t,
      const std::vector<PredPose>& ego_traj,
      const ActionSpec& spec,
      const smpc_lane_change::TargetVehicle* rear_override = nullptr,
      bool passing_front = false) const {
    proj_fail_ = ProjGateFail{};
    if (!spec.change_left || ego_traj.empty()) return false;

    const double lane_change_start = effectiveLaneChangeStartSec(spec);
    const double front_required_gap = activeTargetFrontSafeGap(t);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!target_lane) return false;
    // Do not apply a merge-priority relaxation in a future hard gate.  It is
    // safer to wait for a fresh CHANGE_NOW decision than to pre-authorize a
    // reduced rear gap several seconds ahead.
    const auto check_vehicle = [&](const smpc_lane_change::TargetVehicle& vehicle,
                                   bool front) {
      if (!vehicle.valid) return true;
      const auto modes = modesForVehicle(vehicle);
      for (const auto& ego : ego_traj) {
        if (!ego.valid || ego.t + 1e-6 < lane_change_start) continue;
        const auto target_projection = target_lane->project(ego.x, ego.y);
        if (!target_projection.valid) return false;
        const auto target_sample = target_lane->sample(target_projection.s, 0.0);
        if (!target_sample.valid) return false;
        const double nx = -std::sin(target_sample.yaw);
        const double ny = std::cos(target_sample.yaw);
        const OrientedBox ego_for_lateral_extent = egoOrientedBox(ego);
        const double ego_lateral_extent = axisExtent(ego_for_lateral_extent, nx, ny);
        const double target_entry_distance =
            0.5 * std::max(0.1, lane_width_m_) + ego_lateral_extent +
            std::max(0.0, collision_margin_m_);
        const double target_abs_d = std::abs(target_projection.d);
        // Before the ego footprint reaches the target-lane corridor, a full
        // front/rear headway is not physically required yet.  The oriented
        // collision-risk calculation remains active for every horizon step.
        if (target_abs_d > target_entry_distance) continue;
        const double full_occupancy_d = std::max(
            0.0, ego_reference_preview_target_completion_max_abs_d_m_);
        const double occupancy_progress = std::clamp(
            (target_entry_distance - target_abs_d) /
                std::max(0.1, target_entry_distance - full_occupancy_d),
            0.0, 1.0);
        for (const auto& mode : modes) {
          // A faster vehicle that has just passed ego is allowed to create a
          // merge opening.  Its low-probability brake mode still contributes
          // to risk_by_step, but it must not impose a worst-case full-headway
          // veto in addition to the chance constraint.
          if (front && passing_front && mode.name != "nominal") continue;
          // Likewise, the deterministic rear gap/TTC gate represents the
          // measured rear motion (plus the explicit new-track closing upper
          // bound).  A hypothetical accelerate mode remains in risk_by_step
          // with its mode probability; applying it here as a second
          // probability-one veto made every periodically arriving vehicle
          // block the opening even when its measured closing speed was zero.
          if (!front && mode.name != "nominal") continue;
          bool exceeded_lane_end = false;
          const PredPose tv = predictTargetPose(
              vehicle, mode, ego.t, &exceeded_lane_end);
          // Apply the deterministic gap/TTC gate over exactly the same
          // time-matched interval used by the stochastic risk trace.  A TV
          // mode beyond the downstream CSV endpoint contributes no later
          // samples, but it does not invalidate the complete maneuver.
          if (exceeded_lane_end) continue;
          if (!tv.valid) return false;

          const double ex = std::cos(ego.yaw);
          const double ey = std::sin(ego.yaw);
          const double longitudinal = (tv.x - ego.x) * ex + (tv.y - ego.y) * ey;
          const OrientedBox ego_box = egoOrientedBox(ego);
          const OrientedBox tv_box{
              tv.x, tv.y, tv.yaw, 0.5 * std::max(0.1, vehicle.length),
              0.5 * std::max(0.1, vehicle.width)};
          const double ego_extent = axisExtent(ego_box, ex, ey);
          const double tv_extent = axisExtent(tv_box, ex, ey);
          const double tv_forward_speed =
              std::max(0.0, tv.v * std::cos(smpc_lane_change::wrapToPi(tv.yaw - ego.yaw)));

          if (front) {
            const double gap = longitudinal - ego_extent - tv_extent;
            // 뒤따라갈 앞차에게 절대속도 기반 헤드웨이를 요구하면, 자차가
            // 가속할수록 요구 간격이 함께 커져 합류가 오히려 멀어진다.
            // 접근속도(자차 - 앞차)가 0 이하이면 간격이 줄지 않으므로 절대
            // 하한(target_front_safe_gap_m)만 요구한다.  TTC 3 초 검사와
            // 진입 클리어런스, front_brake 확률 항은 그대로 유지된다.
            // 추월 분기의 기준 속도 (2026-09-17).  앞차가 자차보다 빠르면
            // 간격은 벌어지는데, 절대속도 기준은 자차가 빨라질수록 요구를 키운다.
            // lc_gt 09-16/09-17 측정: ACC 가 자차를 10.7 m/s 로 붙든 채 목표차선
            // 흐름이 16.2 m/s 인 구간에서 완료 후 속도를 앞당겨 적용하자 요구가
            // 1.2 x 예측속도로 18.85 m 까지 올라, 통과하던 279 프레임이 실패했다.
            const double front_headway_reference = passing_front
                ? (pass_gap_headway_use_closing_speed_
                       ? std::max(0.0, ego.v - tv_forward_speed)
                       : ego.v)
                : (target_front_headway_use_closing_speed_
                       ? std::max(0.0, ego.v - tv_forward_speed)
                       : std::max(ego.v, tv_forward_speed));
            const double full_required_gap = std::max(
                front_required_gap,
                std::max(0.0, target_front_min_headway_sec_) *
                    front_headway_reference);
            // 추월 차량이 아직 멀어지는 중이면 occupancy 램프를 적용하지 않고
            // 진입 클리어런스만 요구한다.  full_required_gap 은 "앞차가 급제동해도
            // 받지 않을 거리"인데, passing_front 의 front_brake 모드는 바로 위에서
            // 이 결정론적 게이트에서 제외하고 risk_by_step 의 확률 항으로 넘겼다.
            // 정속으로 멀어지는 nominal 모드에까지 그 거리를 요구하면 같은 위험을
            // 두 번 세게 된다.  TTC 검사와 2.0 m 진입 클리어런스, front_brake 의
            // 확률적 위험도는 그대로 유지된다.
            //
            // lc_gt 14-18 t=23.2s 측정: 자차 20.8 m/s, 추월차 27.3 m/s, 램프가
            // 2.6 초 만에 2.4 -> 30.0 m 로 오르는데(10.6 m/s) 실제 갭은 2.7 m/s 로만
            // 벌어져 8 초 지평 끝에서도 17.6 대 30.0 으로 미달했다.  통과에 필요한
            // 추가 갭이 23.1 m(=3.54 초)라 pass_gap 창 1.6 초를 넘어섰다.
            // 램프를 빼면 0.35 초 뒤 통과한다.
            //
            // 앞차가 다시 자차보다 느려지면 receding 이 거짓이 되어 스텝 단위로
            // 즉시 원래 램프로 복귀한다.
            const bool front_receding = pass_gap_skip_ramp_while_receding_ &&
                tv_forward_speed > ego.v;
            // The tracker places a passing vehicle's rear too far forward
            // (short bbox, position lag): on lc_gt 2026-09-15 0->1 merges SMPC
            // saw a 2 m opening 0.35-0.45 s before ground truth did, while the
            // passer (~9 m/s faster) still overlapped ego by 1-2 m.  Scale the
            // entry clearance with the passer's relative speed; an optional
            // fixed length margin covers the bbox error.
            const double pass_entry_clearance =
                std::max(0.0, pass_gap_entry_front_clearance_m_) +
                std::max(0.0, pass_gap_entry_length_margin_m_) +
                std::max(0.0, pass_gap_entry_relative_speed_time_sec_) *
                    std::max(0.0, tv_forward_speed - ego.v);
            const double required_gap = passing_front
                ? (front_receding
                       ? pass_entry_clearance
                       : pass_entry_clearance +
                             occupancy_progress *
                                 std::max(0.0, full_required_gap - pass_entry_clearance))
                : full_required_gap;
            if (!(gap > required_gap)) {
              proj_fail_ = ProjGateFail{true, true, false, ego.t, gap,
                                        required_gap, vehicle.unique_id,
                                        mode.name, ego.v, tv_forward_speed,
                                        ego.yaw, tv.yaw};
              return false;
            }
            const double closing = std::max(0.0, ego.v - tv_forward_speed);
            if (closing > 0.1 &&
                gap / closing < std::max(0.0, target_front_min_ttc_sec_)) {
              proj_fail_ = ProjGateFail{
                  true, true, true, ego.t, gap,
                  closing * std::max(0.0, target_front_min_ttc_sec_),
                  vehicle.unique_id, mode.name, ego.v, tv_forward_speed,
                  ego.yaw, tv.yaw};
              return false;
            }
          } else {
            const double gap = -longitudinal - ego_extent - tv_extent;
            const double measured_or_new_track_closing =
                conservativeRearClosingSpeed(vehicle, ego.v, ros::Time::now());
            const double closing = std::max(
                measured_or_new_track_closing,
                std::max(0.0, tv_forward_speed - ego.v));
            double base_gap = activeTargetRearSafeGap(t);
            if (low_speed_rear_safety_enabled_ &&
                ego.v <= std::max(0.0, low_speed_rear_ego_speed_mps_)) {
              base_gap = std::max(base_gap,
                                  std::max(0.0, low_speed_rear_safe_gap_m_));
            }
            const double remaining_sec = std::max(
                0.0, lane_change_start +
                    std::max(0.1, spec.lane_change_duration_sec) - ego.t);
            const double required_gap = base_gap + closing *
                (remaining_sec +
                 std::max(0.0, rear_gap_perception_control_delay_sec_));
            if (!(gap > required_gap)) {
              proj_fail_ = ProjGateFail{true, false, false, ego.t, gap,
                                        required_gap, vehicle.unique_id,
                                        mode.name, ego.v, tv_forward_speed,
                                        ego.yaw, tv.yaw};
              return false;
            }
            const double required_ttc = (low_speed_rear_safety_enabled_ &&
                                         ego.v <= std::max(0.0, low_speed_rear_ego_speed_mps_))
                ? std::max(target_rear_min_ttc_sec_, low_speed_rear_min_ttc_sec_)
                : target_rear_min_ttc_sec_;
            if (closing > 0.1 && gap / closing < std::max(0.0, required_ttc)) {
              proj_fail_ = ProjGateFail{true, false, true, ego.t, gap,
                                        closing * std::max(0.0, required_ttc),
                                        vehicle.unique_id, mode.name,
                                        ego.v, tv_forward_speed,
                                        ego.yaw, tv.yaw};
              return false;
            }
          }
        }
      }
      return true;
    };

    const auto* rear = rear_override ? rear_override : &t.target_rear;
    return check_vehicle(t.target_front, true) &&
           (!rear || check_vehicle(*rear, false));
  }

  CandidateEval evaluateCandidate(
      const smpc_lane_change::TargetVehicleSet& t,
      const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
      const ActionSpec& spec,
      double allowed_risk,
      double target_gap_cost) const {
    CandidateEval out;
    out.spec = spec;
    if (!candidateFitsHorizon(spec)) return out;

    const double base_speed_cap_mps = egoPredictionTargetSpeedMps();
    out.longitudinal_model_compatible =
        preparationLongitudinalModelCompatible(t, spec, base_speed_cap_mps);
    if (!out.longitudinal_model_compatible) return out;

    const auto trajectory = buildEgoTrajectory(t, spec);
    out.trajectory = evaluateTrajectory(trajectory, vehicles, &t, spec.change_left);
    out.target_speed_mps = actionTargetSpeedMps(base_speed_cap_mps, spec);
    out.time_risk_safe = timeIndexedRiskSafe(out.trajectory, allowed_risk);
    out.legacy_risk_safe = !behavior_require_legacy_risk_gate_ ||
        (out.trajectory.valid && out.trajectory.risk <= allowed_risk);
    out.clearance_safe = clearanceSafe(out.trajectory);
    const bool passing_front = targetFrontIsPassingVehicle(t);
    const int passing_id = passing_front ? t.target_front.unique_id : -1;
    const auto* effective_rear = targetRearForGap(t, passing_id);
    out.projected_gap_ttc_safe =
        out.trajectory.valid && projectedTargetGapTtcSafe(
            t, trajectory, spec, effective_rear, passing_front);
    const bool drop_gates = !tree_planner_enabled_ && !baseline_keep_gates_;
    out.feasible = out.time_risk_safe && out.legacy_risk_safe &&
        ((out.clearance_safe && out.projected_gap_ttc_safe) || drop_gates);
    if (!out.trajectory.valid) return out;

    const double preparation_time = std::max(0.0, spec.lane_change_start_sec);
    out.cost = lane_change_base_cost_ + target_gap_cost +
        risk_cost_weight_ * out.trajectory.peak_step_risk +
        behavior_delay_cost_weight_ * preparation_time +
        behavior_accel_cost_weight_ * spec.prep_accel_mps2 * spec.prep_accel_mps2;
    return out;
  }

  bool behaviorCommandEnabled(const ActionSpec& spec) const {
    switch (spec.action) {
      case BehaviorAction::kWaitForGap:
      case BehaviorAction::kYieldDecelThenChange:
      case BehaviorAction::kHoldThenChange:
        return behavior_execute_decel_hold_;
      case BehaviorAction::kAccelThenChange:
        return behavior_execute_accel_;
      case BehaviorAction::kKeep:
      case BehaviorAction::kChangeNow:
      default:
        return false;
    }
  }

  void publishBehaviorOutputs(const smpc_lane_change::TargetVehicleSet& t,
                              const CandidateEval& selected,
                              bool now_feasible,
                              bool future_feasible,
                              double allowed_risk,
                              const std::string& reason) {
    const double reported_lane_change_start_sec = selected.spec.change_left
        ? effectiveLaneChangeStartSec(selected.spec)
        : -1.0;
    const bool longitudinal_command_enabled = behaviorCommandEnabled(selected.spec);

    // Keep one ID for a continuous preparation episode.  ACC and decision
    // timers are asynchronous, so incrementing this value every 50 ms would
    // turn a legitimate one-cycle status delay into a false cancellation.
    const bool same_behavior_contract =
        last_behavior_plan_id_ != 0 &&
        last_behavior_action_ == selected.spec.action &&
        last_behavior_current_lane_id_ == t.current_lane_id &&
        last_behavior_target_lane_id_ == t.target_lane_id;
    if (selected.spec.action == BehaviorAction::kYieldDecelThenChange) {
      if (!same_behavior_contract ||
          !std::isfinite(behavior_yield_episode_start_speed_mps_)) {
        behavior_yield_episode_start_speed_mps_ = std::max(0.0, ego_speed_mps_);
      }
    } else {
      behavior_yield_episode_start_speed_mps_ =
          std::numeric_limits<double>::quiet_NaN();
    }
    if (!same_behavior_contract) {
      ++behavior_plan_id_;
      if (behavior_plan_id_ == 0) ++behavior_plan_id_;
      last_behavior_plan_id_ = behavior_plan_id_;
      last_behavior_action_ = selected.spec.action;
      last_behavior_current_lane_id_ = t.current_lane_id;
      last_behavior_target_lane_id_ = t.target_lane_id;
    }

    // plan_id identifies the continuous preparation episode, while this
    // revision identifies the exact cap/timing command within that episode.
    // In particular, a new t_LC or speed cap must not be mistaken for the
    // older request when ACC reports its applied cap back to SMPC.
    constexpr double kCommandStartToleranceSec = 0.05;
    constexpr double kCommandSpeedCapToleranceMps = 0.10;
    const bool same_command_contract =
        last_behavior_command_revision_ != 0 &&
        last_behavior_command_plan_id_ == last_behavior_plan_id_ &&
        last_behavior_command_action_ == selected.spec.action &&
        last_behavior_command_active_ == longitudinal_command_enabled &&
        last_behavior_command_current_lane_id_ == t.current_lane_id &&
        last_behavior_command_target_lane_id_ == t.target_lane_id &&
        std::isfinite(last_behavior_command_lane_change_start_sec_) &&
        std::isfinite(last_behavior_command_speed_cap_mps_) &&
        std::abs(last_behavior_command_lane_change_start_sec_ -
                 reported_lane_change_start_sec) <= kCommandStartToleranceSec &&
        std::abs(last_behavior_command_speed_cap_mps_ - selected.target_speed_mps) <=
            kCommandSpeedCapToleranceMps;
    if (!same_command_contract) {
      previous_behavior_command_plan_id_ = last_behavior_command_plan_id_;
      previous_behavior_command_revision_ = last_behavior_command_revision_;
      previous_behavior_command_active_ = last_behavior_command_active_;
      previous_behavior_command_speed_cap_mps_ = last_behavior_command_speed_cap_mps_;
      ++behavior_command_revision_;
      if (behavior_command_revision_ == 0) ++behavior_command_revision_;
      last_behavior_command_plan_id_ = last_behavior_plan_id_;
      last_behavior_command_revision_ = behavior_command_revision_;
      last_behavior_command_action_ = selected.spec.action;
      last_behavior_command_active_ = longitudinal_command_enabled;
      last_behavior_command_current_lane_id_ = t.current_lane_id;
      last_behavior_command_target_lane_id_ = t.target_lane_id;
      last_behavior_command_lane_change_start_sec_ = reported_lane_change_start_sec;
      last_behavior_command_speed_cap_mps_ = selected.target_speed_mps;
    }

    smpc_lane_change::BehaviorPlan plan;
    plan.header = t.header;
    plan.header.stamp = ros::Time::now();
    plan.action = behaviorPlanActionMessageValue(selected.spec.action);
    plan.plan_id = last_behavior_plan_id_;
    plan.command_revision = last_behavior_command_revision_;
    plan.now_feasible = now_feasible;
    plan.future_feasible = future_feasible;
    plan.longitudinal_command_enabled = longitudinal_command_enabled;
    plan.current_lane_id = t.current_lane_id;
    plan.target_lane_id = t.target_lane_id;
    plan.preparation_accel_mps2 = selected.spec.prep_accel_mps2;
    plan.target_speed_mps = selected.target_speed_mps;
    // Expose the same physical transition start that the trajectory uses,
    // including supervisor/path-switch latency.  The raw candidate delay is
    // deliberately internal so consumers cannot treat t_LC as a reservation.
    plan.lane_change_start_sec = reported_lane_change_start_sec;
    plan.lane_change_duration_sec = selected.spec.lane_change_duration_sec;
    plan.horizon_sec = std::max(1, horizon_steps_) * prediction_dt_sec_;
    plan.selected_cost = selected.cost;
    plan.peak_risk = selected.trajectory.peak_step_risk;
    plan.legacy_risk = selected.trajectory.risk;
    plan.first_risk_violation_time_sec =
        firstRiskViolationTimeSec(selected.trajectory, allowed_risk);
    plan.min_signed_clearance_m = selected.trajectory.min_signed_clearance;
    plan.risk_by_step = selected.trajectory.risk_by_step;
    plan.reason = reason;
    behavior_plan_pub_.publish(plan);

    smpc_lane_change::BehaviorLongitudinalRequest request;
    request.header = plan.header;
    request.action = behaviorRequestActionMessageValue(selected.spec.action);
    request.plan_id = last_behavior_plan_id_;
    request.command_revision = last_behavior_command_revision_;
    request.active = plan.longitudinal_command_enabled;
    request.current_lane_id = t.current_lane_id;
    request.target_lane_id = t.target_lane_id;
    request.speed_cap_mps = selected.target_speed_mps;
    request.requested_accel_mps2 = selected.spec.prep_accel_mps2;
    request.planned_lc_start_sec = reported_lane_change_start_sec;
    request.candidate_cost = selected.cost;
    request.peak_risk = selected.trajectory.peak_step_risk;
    request.reason = reason;
    behavior_request_pub_.publish(request);
  }

  void publishKeep(const smpc_lane_change::TargetVehicleSet& t,
                   const std::string& reason,
                   double keep_cost,
                   double change_cost,
                   double collision_risk) {
    smpc_lane_change::LaneChangeDecision d;
    d.header = t.header;
    d.mode = smpc_lane_change::LaneChangeDecision::KEEP;
    d.request = false;
    d.current_lane_id = t.current_lane_id;
    d.target_lane_id = t.target_lane_id;
    d.target_speed_mps = currentTargetSpeedMps();
    d.keep_cost = keep_cost;
    d.change_cost = change_cost;
    d.collision_risk = collision_risk;
    d.reason = reason;
    decision_pub_.publish(d);
  }

  void publishChange(const smpc_lane_change::TargetVehicleSet& t,
                     const std::string& reason,
                     double keep_cost,
                     double change_cost,
                     double collision_risk) {
    smpc_lane_change::LaneChangeDecision d;
    d.header = t.header;
    d.mode = smpc_lane_change::LaneChangeDecision::CHANGE_LEFT;
    d.request = true;
    d.current_lane_id = t.current_lane_id;
    d.target_lane_id = t.target_lane_id;
    d.target_speed_mps = currentTargetSpeedMps();
    d.keep_cost = keep_cost;
    d.change_cost = change_cost;
    d.collision_risk = collision_risk;
    d.reason = reason;
    decision_pub_.publish(d);
  }

  void publishDebug(const std::string& text) {
    std_msgs::String msg;
    msg.data = text;
    debug_pub_.publish(msg);
    if (debug_publish_marker_) {
      publishDebugMarker(text);
    }
  }

  std::string wrapDebugText(const std::string& text) const {
    constexpr std::size_t kMaxLine = 72;
    std::istringstream input(text);
    std::ostringstream wrapped;
    std::string word;
    std::size_t line_len = 0;
    while (input >> word) {
      if (line_len > 0 && line_len + 1 + word.size() > kMaxLine) {
        wrapped << '\n';
        line_len = 0;
      } else if (line_len > 0) {
        wrapped << ' ';
        ++line_len;
      }
      wrapped << word;
      line_len += word.size();
    }
    return wrapped.str();
  }

  void publishDebugMarker(const std::string& text) const {
    visualization_msgs::Marker marker;
    marker.header.frame_id = "map";
    marker.header.stamp = ros::Time::now();
    marker.ns = "smpc_decision_debug";
    marker.id = 0;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = ego_x_;
    marker.pose.position.y = ego_y_;
    marker.pose.position.z = debug_marker_height_m_;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 0.9;
    marker.color.r = 1.0;
    marker.color.g = 1.0;
    marker.color.b = 1.0;
    marker.color.a = 0.95;
    marker.lifetime = ros::Duration(std::max(0.0, debug_marker_lifetime_sec_));
    marker.text = wrapDebugText(text);
    debug_marker_pub_.publish(marker);
  }

  // ======================================================================
  // 시나리오 트리 플래너 (research/tree-planner). 파이썬 기준 구현:
  // research/sim/smpc_sim/tree.py. tree_planner_enabled=false 이면 아무것도 바뀌지 않는다.
  //
  // 루트 a0 하나마다: 가지 j(정상 / 뒤차 가속 onset 0·1·2·3 s / 앞차 급제동, 확률 p_j,
  // 분기 시점 t_b = onset + 감지 지연), 가지별 대응(계속/계속+가속/복귀/복귀+감속/제동).
  // min_c Σ_j p_j J_j(c_j)  s.t.  모든 스텝 k 에서 Σ_j p_j r_j(c_j, k) ≤ ε  를 조합 열거로 푼다.
  // ======================================================================
  enum TreeResponse { kTreeContinue = 0, kTreeContAccel, kTreeReturn, kTreeReturnDecel, kTreeBrake };

  struct TreeBranch {
    std::string name;
    double probability{0.0};
    double t_b{std::numeric_limits<double>::infinity()};
    std::map<int, TvMode> modes;   // 이 가지에서 사건을 일으키는 차량. 나머지는 nominal.
  };

  struct TreeResult {
    bool feasible{false};
    double cost{std::numeric_limits<double>::infinity()};
    double peak_risk{1.0};
    std::vector<int> combo;
  };

  std::vector<TreeBranch> treeBranches(const smpc_lane_change::TargetVehicleSet& t,
                                       const smpc_lane_change::TargetVehicle* rear_override) const {
    std::vector<TreeBranch> out;
    const smpc_lane_change::TargetVehicle* rear =
        rear_override ? rear_override : (t.target_rear.valid ? &t.target_rear : nullptr);
    double p_rear = rear ? adaptiveEventProbability(*rear) : 0.0;
    std::vector<const smpc_lane_change::TargetVehicle*> fronts;
    if (t.target_front.valid) fronts.push_back(&t.target_front);
    if (t.current_front.valid) fronts.push_back(&t.current_front);
    double p_front = 0.0;
    for (const auto* f : fronts) p_front = std::max(p_front, adaptiveEventProbability(*f));
    p_rear = std::min(p_rear, 0.95);
    p_front = std::min(p_front, 0.95 - p_rear);
    TreeBranch nominal;
    nominal.name = "nominal";
    nominal.probability = 1.0 - p_rear - p_front;
    out.push_back(nominal);
    if (rear && p_rear > 1e-6) {
      const bool already = observedLongitudinalAccel(*rear) >= 0.5;
      const std::vector<double> onsets = already ? std::vector<double>{0.0} : tree_onsets_sec_;
      for (const double o : onsets) {
        TreeBranch b;
        std::ostringstream name;
        // 이미 드러난 가속은 미래 정보가 아니다: 'rear_now' 가지는 복귀를 대응으로 쓰지 않는다.
        if (already) name << "rear_now";
        else name << "rear@" << static_cast<int>(std::round(o));
        b.name = name.str();
        b.probability = p_rear / static_cast<double>(onsets.size());
        b.t_b = already ? 0.0 : o + tree_rear_detect_delay_sec_;
        TvMode m{"rear_accel", 1.0, rear_accel_mps2_, 0.0};
        m.onset_sec = o;
        b.modes[rear->unique_id] = m;
        out.push_back(b);
      }
    }
    if (!fronts.empty() && p_front > 1e-6) {
      TreeBranch b;
      b.name = "front_brake";
      b.probability = p_front;
      b.t_b = tree_front_detect_delay_sec_;
      for (const auto* f : fronts) b.modes[f->unique_id] = TvMode{"front_brake", 1.0, front_brake_accel_mps2_, 0.0};
      out.push_back(b);
    }
    return out;
  }

  // 루트 궤적 → 대응 r 의 궤적. 목표 차선 Frenet (s, d) 에서 바꾼 뒤 지도 좌표로 되돌린다.
  // allowed 가 false 면 그 대응은 쓸 수 없다(복귀는 목표 차선 경계 1.75 m 전까지만).
  // 루트 궤적의 목표 차선 Frenet (s, d). projectExtended 는 폴리라인 전체를 훑으므로 루트마다 한 번만 한다.
  void treeRootFrenet(const std::vector<PredPose>& root, const smpc_lane_change::TargetVehicleSet& t,
                      std::vector<double>* s, std::vector<double>* d) const {
    s->assign(root.size(), 0.0);
    d->assign(root.size(), 0.0);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!target_lane) return;
    for (std::size_t k = 0; k < root.size(); ++k) {
      const auto pr = target_lane->projectExtended(root[k].x, root[k].y, 400.0, 400.0);
      (*s)[k] = pr.s;
      (*d)[k] = pr.d;
    }
  }

  // 복귀 진행률 p(τ/scale), 0.1 s 표(파이썬 tree.RETURN_PROGRESS 와 같다). scale = max(0.35, √(거리/차선 폭)).
  double treeReturnProgress(double tau, double dist) const {
    static const double kProg[57] = {
        0.0000, 0.0005, 0.0019, 0.0041, 0.0068, 0.0100, 0.0151, 0.0233, 0.0339, 0.0464, 0.0600, 0.0770, 0.0992,
        0.1255, 0.1551, 0.1867, 0.2194, 0.2521, 0.2838, 0.3135, 0.3400, 0.3641, 0.3871, 0.4093, 0.4309, 0.4521,
        0.4732, 0.4943, 0.5156, 0.5375, 0.5600, 0.5833, 0.6073, 0.6317, 0.6563, 0.6811, 0.7058, 0.7302, 0.7541,
        0.7775, 0.8000, 0.8219, 0.8432, 0.8634, 0.8825, 0.9026, 0.9228, 0.9419, 0.9586, 0.9717, 0.9800, 0.9853,
        0.9901, 0.9941, 0.9973, 0.9993, 1.0000};
    if (!tree_plant_return_) {
      const double dur = std::max(0.8, tree_return_time_full_sec_ * std::abs(dist) / lane_width_m_);
      return std::clamp(tau / dur, 0.0, 1.0);
    }
    const double scale = std::max(0.35, std::sqrt(std::min(1.0, std::abs(dist) / lane_width_m_)));
    const double x = std::max(0.0, tau / scale) / 0.1;
    const std::size_t i = static_cast<std::size_t>(std::floor(x));
    if (i >= 56) return 1.0;
    const double a = x - static_cast<double>(i);
    return kProg[i] + (kProg[i + 1] - kProg[i]) * a;
  }

  std::vector<PredPose> treeResponse(const std::vector<PredPose>& root, bool change_left, double t_b,
                                     int response, const smpc_lane_change::TargetVehicleSet& t,
                                     bool* allowed, const std::vector<double>* root_s = nullptr,
                                     const std::vector<double>* root_d = nullptr) const {
    *allowed = true;
    if (response == kTreeContinue) return root;
    const std::size_t K = root.size();
    if (K == 0 || !std::isfinite(t_b) || t_b >= root.back().t) {
      *allowed = false;
      return {};
    }
    if (!change_left && (response == kTreeReturn || response == kTreeReturnDecel)) {
      *allowed = false;
      return {};
    }
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!target_lane) {
      *allowed = false;
      return {};
    }
    const double ext = 400.0;
    std::vector<double> s(K), d(K), v(K), tk(K);
    if (root_s && root_d && root_s->size() == K && root_d->size() == K) {
      s = *root_s;
      d = *root_d;
    } else {
      treeRootFrenet(root, t, &s, &d);
    }
    for (std::size_t k = 0; k < K; ++k) {
      v[k] = root[k].v;
      tk[k] = root[k].t;
    }
    const double d0 = t.ego_d_target;
    auto interp = [&](const std::vector<double>& y, double y_at0, double tq) {
      if (tq <= tk[0]) {
        const double a = tk[0] > 1e-9 ? tq / tk[0] : 1.0;
        return y_at0 + (y[0] - y_at0) * a;
      }
      for (std::size_t k = 1; k < K; ++k) {
        if (tq <= tk[k]) {
          const double a = (tq - tk[k - 1]) / std::max(1e-9, tk[k] - tk[k - 1]);
          return y[k - 1] + (y[k] - y[k - 1]) * a;
        }
      }
      return y.back();
    };
    const double d_b = interp(d, d0, t_b);
    const double v_b = interp(v, v[0], t_b);
    const double d_src = -lane_width_m_;
    if ((response == kTreeReturn || response == kTreeReturnDecel) &&
        std::abs(d_b) <= tree_abort_before_abs_d_m_) {
      *allowed = false;
      return {};
    }
    std::vector<double> v2 = v, d2 = d;
    for (std::size_t k = 0; k < K; ++k) {
      if (!(tk[k] > t_b + 1e-9)) continue;
      const double tau = tk[k] - t_b;
      switch (response) {
        case kTreeContAccel:
          v2[k] = std::max(v[k], std::min(v[k] + tree_escape_accel_mps2_ * tau,
                                          std::max(0.0, currentTargetSpeedMps())));
          break;
        case kTreeBrake:
          v2[k] = std::max(0.0, v_b + tree_brake_decel_mps2_ * tau);
          break;
        case kTreeReturn:
        case kTreeReturnDecel: {
          const double frac = treeReturnProgress(tau, d_b - d_src);
          d2[k] = d_b + (d_src - d_b) * frac;
          if (response == kTreeReturnDecel) {
            v2[k] = std::max(0.5 * v_b, v_b + tree_return_decel_mps2_ * tau);
          }
          break;
        }
        default:
          break;
      }
    }
    // 속도가 바뀐 대응은 s 를 다시 적분한다(파이썬과 같은 사다리꼴, 출발점 s[0]).
    std::vector<double> s2 = s;
    if (response == kTreeContAccel || response == kTreeBrake || response == kTreeReturnDecel) {
      double acc = s[0];
      for (std::size_t k = 1; k < K; ++k) {
        acc += 0.5 * (v2[k] + v2[k - 1]) * prediction_dt_sec_;
        if (tk[k] > t_b + 1e-9) s2[k] = acc;
      }
    }
    std::vector<PredPose> out(K);
    for (std::size_t k = 0; k < K; ++k) {
      const auto smp = target_lane->sampleExtended(s2[k], d2[k], ext, ext);
      PredPose p = root[k];
      if (smp.valid) {
        p.x = smp.x;
        p.y = smp.y;
        p.yaw = smp.yaw;
      }
      p.s = s2[k];
      p.d = d2[k];
      p.v = v2[k];
      out[k] = p;
    }
    return out;
  }

  // 가지 하나(차량별 mode 고정)에서 스텝별 충돌 확률 1 - Π_v (1 - P_v,k). evaluateTrajectory 와 같은 식.
  static std::uint64_t treeCacheKey(int uid, const TvMode& mode, std::size_t k) {
    // (차량 id, mode 종류, onset 0.1 s 단위, 스텝) → 정수 키. 가속도는 mode 종류마다 고정이다.
    const std::uint64_t kind = mode.name == "nominal" ? 0u : (mode.name == "rear_accel" ? 1u : 2u);
    const std::uint64_t onset = static_cast<std::uint64_t>(std::lround(std::max(0.0, mode.onset_sec) * 10.0)) & 0xFFu;
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(uid)) << 24) | (kind << 16) | (onset << 8) |
        (static_cast<std::uint64_t>(k) & 0xFFu);
  }

  std::vector<double> treeStepRisk(const std::vector<PredPose>& ego,
                                   const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
                                   const TreeBranch& branch, std::vector<double>* min_clear,
                                   bool* invalid = nullptr) const {
    const std::size_t K = ego.size();
    if (invalid) *invalid = false;
    std::vector<double> no_coll(K, 1.0);
    if (min_clear) min_clear->assign(K, std::numeric_limits<double>::infinity());
    for (const auto& vehicle : vehicles) {
      const auto it = branch.modes.find(vehicle.unique_id);
      const TvMode mode = it != branch.modes.end()
          ? it->second : TvMode{"nominal", 1.0, nominal_target_accel_mps2_, 0.0};
      const bool react = ego_brake_reaction_enabled_ && vehicle.role.find("front") != std::string::npos;
      double react_v = std::max(0.0, ego_speed_mps_);
      double lag = 0.0;
      double trigger = std::numeric_limits<double>::infinity();
      for (std::size_t k = 0; k < K; ++k) {
        const auto& e = ego[k];
        if (react) {
          const double tv_v = modeSpeed(vehicle.v_long, mode, e.t);
          if (!std::isfinite(trigger) && tv_v < e.v) trigger = e.t;
          const bool reacting = std::isfinite(trigger) &&
              e.t >= trigger + std::max(0.0, ego_brake_reaction_delay_sec_);
          react_v = reacting
              ? std::max(tv_v, std::min(e.v, react_v - std::max(0.0, ego_brake_reaction_max_decel_mps2_) * prediction_dt_sec_))
              : e.v;
          lag += std::max(0.0, e.v - react_v) * prediction_dt_sec_;
        }
        bool exceeded = false;
        PredPose tv;
        if (!react) {
          // F7 지연이 없는 차량의 예측은 루트·대응과 무관하다: (차량, mode, 스텝) 단위로 한 번만 계산.
          auto& slot = tree_pred_cache_[treeCacheKey(vehicle.unique_id, mode, k)];
          if (!slot.first) {
            bool ex = false;
            slot.second = predictTargetPose(vehicle, mode, e.t, &ex, 0.0);
            slot.first = ex ? 2 : 1;
          }
          exceeded = slot.first == 2;
          tv = slot.second;
        } else {
          tv = predictTargetPose(vehicle, mode, e.t, &exceeded, lag);
        }
        if (exceeded) break;
        if (!tv.valid) {
          // 차선 끝 전에 TV 예측이 실패: evaluateTrajectory 처럼 fail-closed (그 대응은 쓰지 않는다)
          if (invalid) *invalid = true;
          continue;
        }
        const auto c = collisionProbability(e, tv, vehicle, mode);
        no_coll[k] *= 1.0 - clampProbability(c.probability);
        if (min_clear) (*min_clear)[k] = std::min((*min_clear)[k], c.signed_clearance);
      }
    }
    std::vector<double> risk(K);
    for (std::size_t k = 0; k < K; ++k) risk[k] = clampProbability(1.0 - no_coll[k]);
    return risk;
  }

  TreeResult treeEvaluate(const std::vector<PredPose>& root, bool change_left,
                          const smpc_lane_change::TargetVehicleSet& t,
                          const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
                          const std::vector<TreeBranch>& branches, double root_cost,
                          double allowed_risk) const {
    TreeResult out;
    const std::size_t B = branches.size();
    const std::size_t K = root.size();
    if (K == 0) return out;
    // risk[b][r][k], ok[b][r]
    std::vector<std::array<std::vector<double>, 5>> risk(B);
    std::vector<std::array<bool, 5>> ok(B);
    bool nominal_clear = true;
    std::vector<double> root_s, root_d;
    treeRootFrenet(root, t, &root_s, &root_d);
    for (std::size_t b = 0; b < B; ++b) {
      for (int r = 0; r < 5; ++r) {
        bool allowed = false;
        const auto traj = treeResponse(root, change_left, branches[b].t_b, r, t, &allowed, &root_s, &root_d);
        ok[b][r] = allowed;
        if (!allowed) continue;
        std::vector<double> clear;
        bool invalid = false;
        risk[b][r] = treeStepRisk(traj, vehicles, branches[b],
                                  (branches[b].name == "nominal" && r == kTreeContinue) ? &clear : nullptr,
                                  &invalid);
        if (invalid) ok[b][r] = false;
        if (branches[b].name == "nominal" && r == kTreeContinue) {
          for (const double c : clear) {
            // NaN·-inf 는 통과시키지 않는다(clearanceSafe 처럼 fail-closed). +inf 는 상호작용 없음.
            if (std::isnan(c) || c < behavior_min_signed_clearance_m_) nominal_clear = false;
          }
        }
      }
    }
    // 가지별 의미 있는 대응만 (파이썬 options() 와 같다)
    std::vector<std::vector<int>> options(B);
    for (std::size_t b = 0; b < B; ++b) {
      const auto& n = branches[b].name;
      const bool ex = tree_executable_responses_only_;
      if (n == "nominal") options[b] = {kTreeContinue};
      else if (n == "rear_now") options[b] = ex ? std::vector<int>{kTreeContinue}
                                                : std::vector<int>{kTreeContinue, kTreeContAccel};
      else if (n.rfind("rear", 0) == 0) {
        options[b] = ex ? std::vector<int>{kTreeContinue, kTreeReturn}
                        : std::vector<int>{kTreeContinue, kTreeContAccel, kTreeReturn, kTreeReturnDecel};
      } else options[b] = {kTreeContinue, kTreeBrake, kTreeReturn};
    }
    std::vector<std::size_t> idx(B, 0);
    double best_any_peak = std::numeric_limits<double>::infinity();
    std::vector<int> best_any_combo;
    while (true) {
      bool allowed = true;
      double j = 0.0;
      std::vector<double> r_sum(K, 0.0);
      std::vector<int> combo(B);
      for (std::size_t b = 0; b < B && allowed; ++b) {
        const int r = options[b][idx[b]];
        combo[b] = r;
        if (!ok[b][r]) {
          allowed = false;
          break;
        }
        const double p = branches[b].probability;
        j += p * tree_response_cost_[static_cast<std::size_t>(r)];
        for (std::size_t k = 0; k < K; ++k) r_sum[k] += p * risk[b][r][k];
      }
      if (allowed) {
        const double peak = *std::max_element(r_sum.begin(), r_sum.end());
        const double total = root_cost + j + tree_risk_weight_ * peak;
        if (peak <= allowed_risk && nominal_clear && total < out.cost) {
          out.feasible = true;
          out.cost = total;
          out.peak_risk = peak;
          out.combo = combo;
        }
        if (peak < best_any_peak) {
          best_any_peak = peak;
          best_any_combo = combo;
        }
      }
      // 다음 조합 (가지별 자리 올림)
      std::size_t b = 0;
      while (b < B && ++idx[b] >= options[b].size()) {
        idx[b] = 0;
        ++b;
      }
      if (b >= B) break;
    }
    if (!out.feasible) {
      out.peak_risk = best_any_peak;
      out.combo = best_any_combo;
    }
    return out;
  }

  // 차선 변경 중 재계획: '지금 계속(이후 분기 허용)' vs '지금 복귀'. true 면 복귀한다.
  bool treeMonitorReturn(const smpc_lane_change::TargetVehicleSet& t,
                         const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
                         const smpc_lane_change::TargetVehicle* rear, double allowed_risk,
                         std::string* detail) const {
    const double d_t = std::abs(t.ego_d_target);
    if (d_t <= tree_abort_before_abs_d_m_ || vehicles.empty()) return false;
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!target_lane) return false;
    const double v_e = std::max(0.0, ego_speed_mps_);
    const double T_blend = (have_reference_preview_ && reference_preview_.valid &&
                            reference_preview_.mode == waypoint_system::PathSwitchPreview::BLEND)
        ? reference_preview_.blend_duration_sec : waypoint_blend_duration_sec_;
    const double T_rem = std::max(0.5, T_blend * d_t / lane_width_m_);
    const double pre_cap = egoPredictionTargetSpeedMps();
    const double mission = currentTargetSpeedMps();
    double v_post = mission;
    if (t.target_front.valid) v_post = std::min(v_post, std::max(0.0, t.target_front.v_long));
    const double d0 = std::abs(t.ego_d_target);
    const double sw = d0 > change_rollout_soft_acc_start_abs_d_m_
        ? T_rem * (1.0 - change_rollout_soft_acc_start_abs_d_m_ / d0) : 0.0;
    const auto ego_proj = target_lane->projectExtended(ego_x_, ego_y_, 400.0, 400.0);
    std::vector<PredPose> go, back;
    double v_go = v_e, s_go = ego_proj.s, v_bk = v_e, s_bk = ego_proj.s;
    const double dt = prediction_dt_sec_;
    // 정지·저속이면 계획과 같은 connector 출발 모델: 가속 후 cap, 횡이동은 주행 거리(남은 연결 길이)에 비례.
    // 시간에 비례해 옆으로 미끄러지는 모델은 출발 전 차를 옆 차선에 밀어 넣어 위험을 크게 본다(MORAI S4).
    const bool launch = v_e < change_rollout_launch_speed_threshold_mps_;
    const int tgt = std::clamp(t.target_lane_id, 0, 3);
    const double cap_conn = tgt < static_cast<int>(stopped_launch_connector_speed_cap_by_target_lane_mps_.size())
        ? stopped_launch_connector_speed_cap_by_target_lane_mps_[tgt] : stopped_launch_connector_speed_cap_mps_;
    const double join = (have_reference_preview_ && reference_preview_.mode == waypoint_system::PathSwitchPreview::CONNECTOR &&
                         std::isfinite(reference_preview_.connector_join_arc_m))
        ? reference_preview_.connector_join_arc_m : 20.0;
    const double L_rem = std::max(2.0, join * d_t / lane_width_m_);
    const double s_start = s_go;
    for (int k = 1; k <= std::max(1, horizon_steps_); ++k) {
      const double tau = dt * k;
      // 계속: blendStage 와 같은 법칙 (t_sw = 0), 0.2 s 단위. 저속이면 connector 출발.
      const double ref = launch ? std::max(cap_conn, 0.0) : ((tau - dt) < sw ? pre_cap : v_post);
      const double up = launch ? stopped_launch_connector_max_accel_mps2_ : 2.4;
      double vn = v_go;
      if (ref > v_go + 0.05) vn = std::min(ref, v_go + up * dt);
      else if (ref < v_go - 0.05) vn = std::max(ref, v_go - 4.0 * dt);
      s_go += 0.5 * dt * (v_go + vn);
      v_go = std::max(0.0, vn);
      // 복귀: KEEP 법칙(speedTargetRollout)
      double vb = v_bk;
      if (pre_cap > v_bk + 0.05) vb = std::min(pre_cap, v_bk + 2.4 * dt);
      else if (pre_cap < v_bk - 0.05) vb = std::max(pre_cap, v_bk - 9.0 * dt);
      s_bk += 0.5 * dt * (v_bk + vb);
      v_bk = std::max(0.0, vb);
      double d_go = t.ego_d_target * (1.0 - std::clamp(tau / T_rem, 0.0, 1.0));
      if (launch) {
        const double u = std::clamp((s_go - s_start) / L_rem, 0.0, 1.0);
        d_go = t.ego_d_target * (1.0 - u * u * u * (10.0 - 15.0 * u + 6.0 * u * u));
      }
      const double d_bk = t.ego_d_target + (-lane_width_m_ - t.ego_d_target) *
          treeReturnProgress(tau, t.ego_d_target + lane_width_m_);
      for (int which = 0; which < 2; ++which) {
        const double s_q = which == 0 ? s_go : s_bk;
        const double d_q = which == 0 ? d_go : d_bk;
        const auto smp = target_lane->sampleExtended(s_q, d_q, 400.0, 400.0);
        PredPose p;
        p.valid = smp.valid;
        p.t = tau;
        p.x = smp.x;
        p.y = smp.y;
        p.yaw = smp.yaw;
        p.s = s_q;
        p.d = d_q;
        p.v = which == 0 ? v_go : v_bk;
        (which == 0 ? go : back).push_back(p);
      }
    }
    tree_pred_cache_.clear();
    const auto branches = treeBranches(t, rear);
    const double eps = tree_abort_risk_factor_ * allowed_risk;
    const TreeResult r_go = treeEvaluate(go, true, t, vehicles, branches, 0.0, eps);
    const TreeResult r_bk = treeEvaluate(back, false, t, vehicles, branches,
                                         tree_response_cost_[kTreeReturn], eps);
    std::ostringstream os;
    os << "go_feas=" << r_go.feasible << " go_peak=" << r_go.peak_risk
       << " back_feas=" << r_bk.feasible << " back_peak=" << r_bk.peak_risk;
    if (detail) *detail = os.str();
    const bool revealed = rear && observedLongitudinalAccel(*rear) >= 0.5;
    if (r_go.feasible) {
      if (revealed) {
        for (std::size_t b = 0; b < branches.size() && b < r_go.combo.size(); ++b) {
          if (branches[b].name.rfind("rear", 0) == 0 &&
              (r_go.combo[b] == kTreeReturn || r_go.combo[b] == kTreeReturnDecel)) {
            return true;
          }
        }
      }
      return false;
    }
    return r_bk.feasible || r_bk.peak_risk < r_go.peak_risk;
  }

  // timerCallback 에서 호출: 트리로 CHANGE_NOW / 미래 후보 중 기대 비용 최소를 고른다.
  void treeSelect(const smpc_lane_change::TargetVehicleSet& t,
                  const std::vector<smpc_lane_change::TargetVehicle>& vehicles,
                  const smpc_lane_change::TargetVehicle* rear, bool passing_front,
                  const ActionSpec& change_now_spec, const std::vector<PredPose>& change_traj,
                  bool change_now_gates_ok, bool common_ok, double target_gap_cost,
                  double allowed_risk, bool* request_change, CandidateEval* best_future,
                  bool* have_future, std::string* detail) {
    *request_change = false;
    *have_future = false;
    if (!common_ok) {
      if (detail) *detail = "tree common_gate_fail";
      return;
    }
    tree_pred_cache_.clear();
    const auto branches = treeBranches(t, rear);
    const ros::WallTime wall0 = ros::WallTime::now();
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(3);
    os << "tree br=";
    for (const auto& b : branches) os << b.name << ":" << b.probability << ",";
    double best_cost = std::numeric_limits<double>::infinity();
    int best_index = -1;
    std::vector<ActionSpec> specs;
    specs.push_back(change_now_spec);
    for (const auto& c : buildBehaviorCandidates(t)) {
      if (c.action != BehaviorAction::kChangeNow && candidateFitsHorizon(c)) specs.push_back(c);
    }
    std::vector<int> best_combo;
    // 분기 한정: 대응 비용·위험 비용은 0 이상이라 루트 총비용 ≥ 루트 비용. 루트 비용 순으로 평가하고
    // 루트 비용이 지금까지의 최소 기대 비용 이상이면 그 뒤 루트는 볼 필요가 없다(결과는 같다).
    std::vector<std::size_t> order(specs.size());
    std::iota(order.begin(), order.end(), 0);
    auto root_cost_of = [&](const ActionSpec& sp) {
      return lane_change_base_cost_ + target_gap_cost +
          behavior_delay_cost_weight_ * std::max(0.0, sp.lane_change_start_sec) +
          behavior_accel_cost_weight_ * sp.prep_accel_mps2 * sp.prep_accel_mps2;
    };
    std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      return root_cost_of(specs[a]) < root_cost_of(specs[b]);
    });
    int evaluated = 0;
    for (const std::size_t i : order) {
      const auto& spec = specs[i];
      if (root_cost_of(spec) >= best_cost) break;
      std::vector<PredPose> traj;
      bool gate_ok = true;
      if (i == 0) {
        traj = change_traj;
        gate_ok = !change_traj.empty() && (!tree_keep_gates_ || change_now_gates_ok);
        if (gate_ok && tree_monitor_enabled_) {
          // 같은 상태에서 실행 중 재계획이 곧바로 복귀를 고를 바꾸기는 승인하지 않는다(리뷰 #6).
          // treeMonitorReturn 이 예측 캐시를 비우므로 이 루트 평가 전에 부른다.
          if (treeMonitorReturn(t, vehicles, rear, allowed_risk, nullptr)) {
            os << " change_now_monitor_return";
            gate_ok = false;
          }
          tree_pred_cache_.clear();
        }
      } else {
        const double base_cap = egoPredictionTargetSpeedMps();
        if (!preparationLongitudinalModelCompatible(t, spec, base_cap)) continue;
        traj = buildEgoTrajectory(t, spec);
        if (traj.empty()) continue;
        if (tree_keep_gates_ && !projectedTargetGapTtcSafe(t, traj, spec, rear, passing_front)) continue;
      }
      if (!gate_ok || traj.empty()) continue;
      const double root_cost = root_cost_of(spec);
      ++evaluated;
      const TreeResult r = treeEvaluate(traj, true, t, vehicles, branches, root_cost, allowed_risk);
      if (r.feasible && r.cost < best_cost) {
        best_cost = r.cost;
        best_index = static_cast<int>(i);
        best_combo = r.combo;
      }
    }
    const double ms = (ros::WallTime::now() - wall0).toSec() * 1000.0;
    os << " roots=" << specs.size() << " evaluated=" << evaluated << " ms=" << ms;
    if (best_index < 0) {
      os << " none_feasible";
      if (detail) *detail = os.str();
      return;
    }
    os << " pick=" << specs[best_index].name << " t_lc=" << specs[best_index].lane_change_start_sec
       << " a=" << specs[best_index].prep_accel_mps2 << " cost=" << best_cost << " resp=";
    for (const int r : best_combo) os << r;
    if (detail) *detail = os.str();
    if (best_index == 0) {
      *request_change = true;
      return;
    }
    CandidateEval ev = evaluateCandidate(t, vehicles, specs[best_index], allowed_risk, target_gap_cost);
    ev.feasible = true;
    ev.cost = best_cost;
    *best_future = ev;
    *have_future = true;
  }

  // ===== 반응형 예측 + 트리 단독 판단 (research/docs/plan_v2.md 단계 6) ==========================
  // 파이썬 기준 구현 research/sim/smpc_sim/reactive.py (joint_rollout), tree_reactive.py 와 같은 식이다.
  // 종방향은 자차 출발점 기준 상대 좌표(자차 s = 0, 타겟 차량 s = delta_s), 횡방향은 목표 차선 중심 기준 d 로
  // 굴리고, 위험은 차선 polyline 으로 지도 좌표에 옮겨 collisionProbability 로 구한다.
  struct RxBranch {
    std::string rear;    // "yield" | "nonyield"
    std::string front;   // "base" | "brake" | "accel"
    std::string tb;      // "" | "rear" | "front" | "now"  (복귀 대응의 분기 시점 규칙)
    double p{1.0};
  };
  struct RxSpec {
    bool change{false};
    double t_sw{0.0};
    double dur{4.0};
    BehaviorAction action{BehaviorAction::kKeep};
    double prep_a{0.0};
  };
  struct RxTv {
    smpc_lane_change::TargetVehicle v;
    std::string role;    // "cf" | "tf" | "rear" | "ctx"
    int lead{-1};
  };
  struct RxResult {
    int B{0}, N{0}, K{0};
    std::vector<double> risk;       // ((b*2 + q)*N + n)*K + k
    std::vector<uint8_t> allowed;   // (b*2 + q)*N + n
    std::vector<uint8_t> valid;     // n
    std::vector<double> ego_s, ego_d, ego_v;               // 행 × K (단위 시험용)
    std::vector<std::vector<double>> tv_s;                 // 차량별, 행 × K
    double& at(int b, int q, int n, int k) { return risk[((b * 2 + q) * N + n) * K + k]; }
  };

  static double rxIdm(double v, double gap, double v_lead, double v0, double a, double b, double s0, double T) {
    gap = std::max(gap, 0.1);
    const double s_star = s0 + std::max(0.0, v * T + v * (v - v_lead) / (2.0 * std::sqrt(a * b)));
    return a * (1.0 - std::pow(v / std::max(v0, 0.5), 4.0) - (s_star / gap) * (s_star / gap));
  }

  // acc.py speed_for_lead / acc_speed_planner speedForLead 와 같은 앞차 추종 목표 속도. hard(또는 0) 여부도 낸다.
  void rxSpeedForLead(double gap, double v_lead, double v_e, double cruise, double* cmd_out, bool* hard_out) const {
    const double vL = std::min(std::max(v_lead, 0.0), cruise);
    const double c = std::max(0.0, v_e - vL);
    const double g_star = rx_acc_.standstill_gap_m + rx_acc_.time_headway_sec * v_e +
        rx_acc_.closing_speed_gap_time_sec * c + c * c / (2.0 * 3.8);
    double cmd = std::max(vL + rx_acc_.gap_gain * (gap - g_star), vL - rx_acc_.max_follow_speed_drop_mps);
    cmd = std::min(cmd, std::sqrt(vL * vL + 2.0 * 3.8 * std::max(0.0, gap - rx_acc_.emergency_gap_m)));
    const double closing = v_e - std::max(0.0, v_lead);
    const double ttc = closing > 0.1 ? gap / std::max(closing, 1e-6) : std::numeric_limits<double>::infinity();
    if (closing > 0.1) cmd = std::min(cmd, vL + std::max(0.0, gap - rx_acc_.emergency_gap_m) / rx_acc_.soft_ttc_sec);
    bool hard = closing > 0.1 && ttc <= rx_acc_.hard_ttc_sec;
    if (hard) cmd = std::min(cmd, vL);
    if (gap <= rx_acc_.critical_gap_m || (closing > 0.1 && ttc <= rx_acc_.critical_ttc_sec)) {
      cmd = 0.0;
      hard = true;
    }
    *cmd_out = std::clamp(cmd, 0.0, cruise);
    *hard_out = hard;
  }

  // (현재 차선 앞차, 목표 차선 앞차, 목표 차선 뒤·옆 차량들, 앞차의 앞차). tree_reactive.py _vehicles + target_lane_vehicles.
  std::vector<RxTv> rxVehicles(const smpc_lane_change::TargetVehicleSet& t) const {
    std::vector<smpc_lane_change::TargetVehicle> all;
    std::set<int> seen;
    auto add = [&](const smpc_lane_change::TargetVehicle& v) {
      if (v.valid && !seen.count(v.unique_id)) {
        seen.insert(v.unique_id);
        all.push_back(v);
      }
    };
    add(t.current_front);
    add(t.target_front);
    add(t.target_rear);
    for (const auto& v : t.nearby_vehicles) add(v);
    // 목표 차선 차량은 중심 위치로 나눈다: 중심이 자차 앞(delta_s > 0)인 차 중 가장 가까운 차 = 앞차, 중심이 자차 뒤인 차 =
    // 뒤·옆 차. 실제 ACC 의 앞차 조건(lead_requires_center_ahead)과 같은 기준이다. 선택기의 target_front 는 앞끝만 자차를
    // 넘은 차(옆에 겹친 차)도 고르므로, 그대로 쓰면 자차가 추월 중인 옆 차가 앞차가 되고 진짜 앞차는 그 차의 리더(ctx)로
    // 밀려 위험에서 빠진다(MORAI H_s70_d40: 선택기가 옆 차를 target_rear 로 넘기는 순간 위험 0 → 0.86).
    // 선택기가 차선을 넘나드는 차를 target_front 에서 잠깐 빼도(MORAI S1) 같은 기준으로 다시 잡힌다.
    // 앞차보다 더 앞의 차(tf2)도 넣는다: 자차가 느린 앞차를 지나치면 그 다음 차가 앞차가 되는데, 리더(ctx)로만 두면
    // 위험·자차 ACC 에서 빠져 지나치는 순간 위험이 뛴다(H_s70_d40). tf2 는 자기 앞차를 따르고 앞차 예외 가지는 받지 않는다.
    std::vector<smpc_lane_change::TargetVehicle> fronts;
    for (const auto& v : all) {
      if (v.lane_id == t.target_lane_id && v.delta_s > 0.0) fronts.push_back(v);
    }
    std::sort(fronts.begin(), fronts.end(), [](const auto& a, const auto& b) { return a.delta_s < b.delta_s; });
    std::vector<RxTv> out;
    if (t.current_front.valid) out.push_back({t.current_front, "cf", -1});
    for (std::size_t i = 0; i < fronts.size(); ++i) out.push_back({fronts[i], i == 0 ? "tf" : "tf2", -1});
    std::vector<smpc_lane_change::TargetVehicle> rears;
    for (const auto& v : all) {
      if (v.lane_id != t.target_lane_id || v.delta_s > 0.0) continue;
      rears.push_back(v);
    }
    std::sort(rears.begin(), rears.end(), [](const auto& a, const auto& b) { return a.delta_s > b.delta_s; });
    for (const auto& r : rears) out.push_back({r, "rear", -1});
    const std::size_t n_main = out.size();
    for (std::size_t i = 0; i < n_main; ++i) {
      if (out[i].role != "cf" && out[i].role != "tf" && out[i].role != "tf2") continue;
      const smpc_lane_change::TargetVehicle* best = nullptr;
      for (const auto& w : all) {
        if (w.lane_id != out[i].v.lane_id || w.unique_id == out[i].v.unique_id) continue;
        if (w.delta_s <= out[i].v.delta_s + 1e-3) continue;
        if (!best || w.delta_s < best->delta_s) best = &w;
      }
      if (!best) continue;
      int main_j = -1;            // 리더가 이미 예측하는 차(tf2)면 그 차를 따른다
      for (std::size_t j = 0; j < n_main; ++j) {
        if (out[j].v.unique_id == best->unique_id) main_j = static_cast<int>(j);
      }
      if (main_j >= 0) {
        out[i].lead = main_j;
      } else {
        out.push_back({*best, "ctx", -1});
        out[i].lead = static_cast<int>(out.size()) - 1;
      }
    }
    for (std::size_t i = 0; i < n_main; ++i) {
      if (out[i].role != "rear") continue;
      int best = -1;
      for (std::size_t j = 0; j < n_main; ++j) {
        if (j == i || out[j].v.lane_id != out[i].v.lane_id || out[j].v.delta_s <= out[i].v.delta_s) continue;
        if (best < 0 || out[j].v.delta_s < out[best].v.delta_s) best = static_cast<int>(j);
      }
      out[i].lead = best;
    }
    return out;
  }

  // reactive.py joint_rollout. mid_T_rem >= 0 이면 차선 변경 도중(모니터): 루트 = 지금부터 계속, 대응 1 = 지금 복귀.
  RxResult rxRollout(const smpc_lane_change::TargetVehicleSet& t, const std::vector<RxSpec>& specs,
                     const std::vector<RxBranch>& br, double v_e0, double mission, bool connector,
                     double join_arc, double yield_ref, double mid_T_rem) const {
    RxResult res;
    const int K = std::max(1, horizon_steps_);
    const double dt = prediction_dt_sec_;
    const double W = lane_width_m_;
    const int N = static_cast<int>(specs.size());
    const int B = static_cast<int>(br.size());
    res.B = B;
    res.N = N;
    res.K = K;
    res.risk.assign(static_cast<std::size_t>(B * 2 * N * K), 0.0);
    res.allowed.assign(static_cast<std::size_t>(B * 2 * N), 1);
    res.valid.assign(static_cast<std::size_t>(N), 1);
    const bool mid = mid_T_rem >= 0.0;
    const double d0 = std::isfinite(t.ego_d_target) ? t.ego_d_target : -W;
    const double d_src = d0 >= 0.0 ? W : -W;           // 원 차선 중심(목표 차선 기준)
    const double src_end = std::max(0.0, t.distance_to_lane_end);
    const auto* target_lane = laneFor(t.target_lane_id);
    if (!target_lane) {
      std::fill(res.valid.begin(), res.valid.end(), 0);
      return res;
    }
    // 목표 차선 위 자차 위치: 선택기(target_selector_node)와 같은 투영(시작 전 연장만, 끝 뒤 연장 없음).
    // 끝 뒤 400 m 연장을 허용하면 연장선이 옆 차선의 자차 가까이 지나는 곳(MORAI H 격자 차선 2→3)에서 자차가
    // 연장선에 붙어(s > 차선 길이) 모든 단계가 '목표 차선 끝 뒤'로 빠지고 위험이 0 이 됐다.
    const auto ego_proj = target_lane->projectExtended(ego_x_, ego_y_, tv_lane_projection_extension_before_m_, 0.0);
    const double tgt_end = target_lane->length() - ego_proj.s;
    const double half_w = 0.5 * ego_width_m_;
    const auto tvs = rxVehicles(t);
    const int V = static_cast<int>(tvs.size());
    const double L_rem = mid ? std::max(2.0, join_arc * std::abs(d0) / W) : std::max(1.0, join_arc);
    const double yield_floor = std::max(0.0, yield_ref - behavior_yield_max_speed_drop_mps_);
    const int M = B * 2 * N;
    res.ego_s.assign(static_cast<std::size_t>(M * K), 0.0);
    res.ego_d.assign(static_cast<std::size_t>(M * K), 0.0);
    res.ego_v.assign(static_cast<std::size_t>(M * K), 0.0);
    res.tv_s.assign(static_cast<std::size_t>(V), std::vector<double>(static_cast<std::size_t>(M * K), 0.0));
    std::vector<uint8_t> tf_hard(static_cast<std::size_t>(M * K), 0);
    // 자차가 지나친 목표 차선 앞차(passed_front_as_rear): 차량별 [m*K + k], 한 번 지나치면 그 행에서 계속 뒤차
    std::vector<std::vector<uint8_t>> passed_mk(static_cast<std::size_t>(V), std::vector<uint8_t>(static_cast<std::size_t>(M * K), 0));
    std::vector<double> tau_m_row(static_cast<std::size_t>(M), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> ds_sw_row(static_cast<std::size_t>(M), std::numeric_limits<double>::quiet_NaN());
    std::vector<double> ret_row(static_cast<std::size_t>(M), std::numeric_limits<double>::infinity());

    for (int b = 0; b < B; ++b) {
      for (int q = 0; q < 2; ++q) {
        for (int n = 0; n < N; ++n) {
          const int m = (b * 2 + q) * N + n;
          const RxSpec& sp = specs[static_cast<std::size_t>(n)];
          const double t_sw = mid ? 0.0 : sp.t_sw;
          double s = 0.0, v = std::max(0.0, v_e0), a = 0.0, f = v, ref_prev = f, y = d0;
          double ds_sw = (mid && sp.change) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
          double tau_m = (mid && sp.change) ? 0.0 : std::numeric_limits<double>::quiet_NaN();
          double ret_t = std::numeric_limits<double>::infinity(), y_ret0 = 0.0;
          double y_m = mid ? d0 : std::numeric_limits<double>::quiet_NaN();   // 횡이동 시작 때 자차 위치
          double t_in = std::numeric_limits<double>::infinity();
          std::vector<double> xs(static_cast<std::size_t>(V)), xv(static_cast<std::size_t>(V));
          std::vector<uint8_t> passed(static_cast<std::size_t>(V), 0);
          for (int i = 0; i < V; ++i) {
            xs[static_cast<std::size_t>(i)] = tvs[static_cast<std::size_t>(i)].v.delta_s;
            xv[static_cast<std::size_t>(i)] = std::max(0.0, tvs[static_cast<std::size_t>(i)].v.v_long);
            const auto& role = tvs[static_cast<std::size_t>(i)].role;
            passed[static_cast<std::size_t>(i)] =
                rx_.passed_front_as_rear && (role == "tf" || role == "tf2") && xs[static_cast<std::size_t>(i)] < 0.0;
          }
          for (int k = 0; k < K; ++k) {
            const double tau0 = dt * k;
            // ---- 횡 위치 ----
            const double moved = s;
            if (!mid && sp.change && std::isnan(ds_sw) && tau0 >= t_sw - 1e-6) ds_sw = moved;
            if (!mid && sp.change && !std::isnan(ds_sw) && std::isnan(tau_m) &&
                (v >= ego_reference_preview_motion_start_speed_mps_ ||
                 moved - ds_sw >= ego_reference_preview_motion_start_distance_m_)) {
              tau_m = tau0;
            }
            double alpha = 0.0;
            if (!std::isnan(tau_m)) {
              if (connector) {
                const double u = std::clamp((moved - (std::isnan(ds_sw) ? 0.0 : ds_sw)) / L_rem, 0.0, 1.0);
                alpha = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u);
              } else {
                alpha = std::clamp((tau0 - tau_m) / (mid ? mid_T_rem : sp.dur), 0.0, 1.0);
              }
            }
            // 횡이동은 그때의 실제 위치에서 시작한다(connector 는 현재 자세에서 만든다). 원 차선 중심에서 시작한다고
            // 보면, 복귀 뒤 옆으로 비켜 선 채 다시 승인할 때(MORAI H_s50_d60: 0.6–0.8 m) 승인이 침범을 그만큼 적게
            // 보고 실제 위치를 쓰는 모니터와 어긋난다.
            const double y_keep = d_src + (d0 - d_src) * std::exp(-tau0 / std::max(0.1, lane_center_decay_tau_sec_));
            if (std::isnan(y_m) && !std::isnan(tau_m)) y_m = y_keep;
            double y_go = !sp.change ? y_keep : std::isnan(y_m) ? y_keep : y_m + (0.0 - y_m) * alpha;
            if (sp.change && s > src_end) y_go = 0.0;
            double tb = std::numeric_limits<double>::infinity();
            const std::string& rule = br[static_cast<std::size_t>(b)].tb;
            if (rule == "now") tb = 0.0;      // 실행 중 모니터: 지금 복귀
            else if (rule == "front") tb = rx_.front_detect_s;
            else if (rule == "rear") tb = t_in + rx_.rear_react_delay_s + rx_.rear_detect_s;
            const double y_now = k == 0 ? d0 : y;
            if (q == 1 && sp.change && std::isinf(ret_t) && tau0 >= tb - 1e-9 && std::abs(y_now) > rx_.abort_before_abs_d) {
              ret_t = tau0;
              y_ret0 = y_now;
            }
            if (std::isfinite(ret_t)) {
              const double scale = std::max(0.35, std::sqrt(std::min(1.0, std::abs(y_ret0 - d_src) / W)));
              const double fr = interpReturnProgress(std::max(0.0, tau0 - ret_t) / scale);
              y = y_ret0 + (d_src - y_ret0) * fr;
            } else {
              y = y_go;
            }
            const double d_t = std::abs(y);
            const double intr = 0.5 * W + half_w - d_t;
            if (std::isinf(t_in) && intr >= rx_.rear_react_intrusion_m) t_in = tau0;
            const bool in_target = d_t < 0.5 * W;
            const bool returning = std::isfinite(ret_t);
            const bool lc_active = sp.change && tau0 >= t_sw && !returning;
            const std::string& rear_mode = br[static_cast<std::size_t>(b)].rear;
            const std::string& front_mode = br[static_cast<std::size_t>(b)].front;
            // ---- 타겟 차량 ----
            std::vector<double> ns(static_cast<std::size_t>(V)), nv(static_cast<std::size_t>(V));
            for (int i = 0; i < V; ++i) {
              const auto& x = tvs[static_cast<std::size_t>(i)];
              const std::size_t ui = static_cast<std::size_t>(i);
              if (x.role == "ctx") {
                ns[ui] = xs[ui];
                nv[ui] = xv[ui];
                continue;
              }
              const double v0x = std::max(0.0, x.v.v_long);
              const double len = std::max(0.1, x.v.length);
              double base, gap_l = 0.0;
              if (x.lead >= 0) {
                const auto& L = tvs[static_cast<std::size_t>(x.lead)];
                gap_l = xs[static_cast<std::size_t>(x.lead)] - 0.5 * std::max(0.1, L.v.length) - xs[ui] - 0.5 * len;
                base = x.role == "rear"
                    ? rxIdm(xv[ui], gap_l, xv[static_cast<std::size_t>(x.lead)], std::max(v0x, 0.5), rx_.rear_idm_a, rx_.rear_idm_b, rx_.rear_idm_s0, rx_.rear_idm_T)
                    : rxIdm(xv[ui], gap_l, xv[static_cast<std::size_t>(x.lead)], std::max(v0x, 0.5), rx_.front_idm_a, rx_.front_idm_b, rx_.front_idm_s0, rx_.front_idm_T);
              } else {
                base = x.v.accel * std::exp(-tau0 / rx_.accel_decay_s);
              }
              double acc;
              if (passed[ui]) {
                // 지나친 목표 차선 앞차 = 뒤차: yield 가지면 반응 지연 뒤 자차를 따르고, nonyield 면 자기 앞차만 본다.
                // 앞차 예외(급감속·가속)는 지나친 뒤엔 없다.
                const bool react = rear_mode == "yield" && tau0 >= t_in + rx_.rear_react_delay_s - 1e-9 && intr > 0.0 && s > xs[ui];
                if (react) {
                  const double gap_e = (s - ego_rear_extent_m_) - (xs[ui] + 0.5 * len);
                  base = std::min(base, rxIdm(xv[ui], gap_e, v, std::max(v0x, 0.5), rx_.rear_idm_a, rx_.rear_idm_b, rx_.rear_idm_s0, rx_.rear_idm_T));
                }
                acc = std::max(base, -rx_.rear_max_brake);
              } else if (x.role == "cf" || x.role == "tf") {
                if (front_mode == "brake") {
                  acc = rx_.front_brake;
                } else if (front_mode == "accel") {
                  acc = xv[ui] < v0x + 5.0 ? rx_.front_accel : 0.0;
                  // 가속 예외도 자기 앞차는 뚫고 지나가지 않는다(IDM 으로 따른다). 없으면 앞차가 tf2 를 통과하고 자차 ACC 는
                  // tf 만 따라 느린 tf2 에 들이받는 가짜 충돌이 지평선 끝에 생긴다. 승인(출발 지연 포함)에는 지평선 밖, 출발 직후
                  // 모니터에는 안이라 위험이 0.05 → 0.094 로 넘어가 복귀했다(MORAI H_s30_d40 v8).
                  if (x.lead >= 0) {
                    acc = std::min(acc, rxIdm(xv[ui], gap_l, xv[static_cast<std::size_t>(x.lead)], v0x + 5.0, rx_.front_idm_a,
                                              rx_.front_idm_b, rx_.front_idm_s0, rx_.front_idm_T));
                  }
                } else {
                  acc = base;
                }
              } else if (x.role == "tf2") {
                acc = base;
              } else {
                const bool react = rear_mode == "yield" && tau0 >= t_in + rx_.rear_react_delay_s - 1e-9 && intr > 0.0 && s > xs[ui];
                if (react) {
                  const double gap_e = (s - ego_rear_extent_m_) - (xs[ui] + 0.5 * len);
                  base = std::min(base, rxIdm(xv[ui], gap_e, v, std::max(v0x, 0.5), rx_.rear_idm_a, rx_.rear_idm_b, rx_.rear_idm_s0, rx_.rear_idm_T));
                }
                acc = std::max(base, -rx_.rear_max_brake);
              }
              nv[ui] = std::max(0.0, xv[ui] + acc * dt);
              ns[ui] = xs[ui] + 0.5 * (xv[ui] + nv[ui]) * dt;
            }
            // ---- 자차 ACC ----
            double cruise_k = mission;
            if (rx_.mission_ramp > 0.0 && rx_.cruise_max > 0.0) {
              cruise_k = std::min(rx_.cruise_max, mission + rx_.mission_ramp * std::max(0.0, sp.change ? tau0 - t_sw : -1.0));
            }
            double raw = cruise_k, c_max = 0.0, bind_c = std::numeric_limits<double>::quiet_NaN(), bind_g = 0.0;
            bool hard = false, soft_bind = false;
            const bool entry = d_t <= rx_acc_.target_front_acc_entry_abs_d_m;
            bool front_ahead = false;     // 더 가까운 목표 차선 앞차가 아직 자차 앞에 있나
            bool floor_block = false;     // merge_floor: 8 m 안 앞차 또는 80 m 안 느린(≤ 11.1 m/s) 앞차가 있어 하한을 걸지 않음
            bool tf_blocks = false;       // merge_floor: 목표 차선 앞차가 11 m 안(targetFrontBlocksMerge)
            for (int i = 0; i < V; ++i) {
              const auto& x = tvs[static_cast<std::size_t>(i)];
              bool use = false, allow_hard = true;
              if (x.role == "cf") {
                use = !in_target || returning;
              } else if (x.role == "tf" || x.role == "tf2") {
                use = entry || (lc_active && d_t <= rx_acc_.target_front_overlap_soft_acc_start_abs_d_m);
                allow_hard = entry;
              } else {
                continue;
              }
              const double len = std::max(0.1, x.v.length);
              const double gap = (xs[static_cast<std::size_t>(i)] - 0.5 * len) - (s + ego_front_extent_m_);
              // tf2 는 더 가까운 앞차를 모두 지나친 뒤에만 따른다(선택기·ACC 가 그때 그 차를 앞차로 잡는다)
              const bool closer_ahead = front_ahead;
              if (x.role == "tf2" && front_ahead) use = false;
              if (x.role != "cf") front_ahead = front_ahead || gap > -len;
              // 실제 ACC 는 하한 판정에 현재 차선 앞차와 목표 차선 앞차를 따르든 말든 본다(mergeRollingFrontBlocked,
              // mergeRollingLowSpeedLeadBlocked). 앞 = 중심이 자차 기준점보다 앞(leadIsInFront).
              if (rx_.merge_floor && xs[static_cast<std::size_t>(i)] > s) {
                const bool chk = x.role == "cf" ? (!in_target || returning) : (x.role == "tf" || !closer_ahead);
                if (chk && (gap <= rx_acc_.merge_keep_rolling_front_block_gap_m ||
                            (xv[static_cast<std::size_t>(i)] <= rx_acc_.merge_keep_rolling_low_speed_lead_mps &&
                             gap <= rx_acc_.merge_keep_rolling_low_speed_lead_max_gap_m))) {
                  floor_block = true;
                }
                if (x.role == "tf" && gap <= rx_acc_.min_target_front_gap_m) tf_blocks = true;
              }
              if (!use || gap <= -len) continue;
              double cmd;
              bool hd;
              const double xvv = xv[static_cast<std::size_t>(i)];
              rxSpeedForLead(gap, xvv, v, cruise_k, &cmd, &hd);
              const double c = std::max(0.0, v - std::min(std::max(xvv, 0.0), cruise_k));
              const double closing = v - std::max(0.0, xvv);
              const bool rapid = rx_acc_.rapid_lead_brake_enabled && c >= rx_acc_.rapid_lead_min_closing_speed_mps &&
                  (gap / std::max(closing, 1e-6) <= rx_acc_.rapid_lead_ttc_sec ||
                   (gap > rx_acc_.emergency_gap_m && c * c / (2.0 * std::max(gap - rx_acc_.emergency_gap_m, 1e-6)) >= 5.0));
              c_max = std::max(c_max, c);
              if (cmd < raw) {
                raw = cmd;
                soft_bind = !allow_hard;
                bind_c = c;
                bind_g = gap;
              }
              if ((hd || rapid) && allow_hard) hard = true;
            }
            const double dist = src_end - s;
            // merge_floor: 차선 끝 대기(정지) = 차선 변경 전, 급함(남은 거리 ≤ 120 m 또는 시간 ≤ 5 s), 목표 차선 앞차가
            // 11 m 안(acc mergeWaitStopRequired; WAIT_FOR_GAP 은 루트에 없다). 실제 ACC 는 이때만 차선 끝 앞에서 선다.
            bool merge_wait = false;
            if (rx_.merge_floor) {
              const bool urgent = t.lane_change_required &&
                  (dist <= rx_.floor_urgent_distance_m || (v > 0.1 && dist / v <= rx_.floor_urgent_time_s));
              merge_wait = !lc_active && urgent && tf_blocks;
            }
            if (!lc_active && !in_target && dist < 200.0 && (!rx_.merge_floor || merge_wait)) {
              const double lim = std::sqrt(2.0 * rx_.lane_end_decel * std::max(0.0, dist - rx_acc_.lane_end_merge_wait_stop_distance_m));
              if (lim < raw) {
                raw = lim;
                soft_bind = false;
                bind_c = std::numeric_limits<double>::quiet_NaN();
              }
            }
            // merge_floor: 실제 ACC 의 merge_keep_rolling 하한 — 앞차·차선 끝 제한 뒤, 차선 변경 중 차선 끝 상한·
            // HOLD/YIELD 상한·connector 상한 앞(acc_speed_planner applyMergeRollingFloor 순서).
            if (rx_.merge_floor && rx_acc_.merge_keep_rolling_enabled && !hard && !merge_wait && !floor_block) {
              const bool pressure = lc_active ||
                  (t.lane_change_required && (t.lane_change_prepare || t.lane_change_urgent || t.emergency_stop_required));
              bool stop_zone = false;     // 차선 끝에서 서는 차로(3)의 정지 준비 구간
              if (in_target && !returning &&
                  std::find(rx_acc_.lane_end_stop_lane_ids.begin(), rx_acc_.lane_end_stop_lane_ids.end(), t.target_lane_id) !=
                      rx_acc_.lane_end_stop_lane_ids.end()) {
                const double d3 = tgt_end - s;
                stop_zone = d3 <= rx_acc_.lane_end_stop_prepare_distance_m || (v > 0.1 && d3 / v <= rx_acc_.lane_end_stop_prepare_time_sec);
              }
              const double fl = std::min(std::max(0.0, rx_acc_.merge_keep_rolling_min_speed_mps), cruise_k);
              if (pressure && !stop_zone && fl > 1e-3 && raw < fl) {
                raw = fl;
                soft_bind = false;
              }
            }
            // 실제 ACC 처럼 connector 실행 중에는 이 상한을 걸지 않는다(connector 상한만, acc_speed_planner
            // !endpoint_connector_active). 걸면 차선 끝이 가까울수록 자차가 느려진다고 예측해 승인이 위험을 낮게 보고,
            // 실제로 7 m/s 로 달리는 자차를 보는 모니터와 어긋난다(MORAI H_s30_d80 v8: 승인 뒤 3.4 s 에 복귀).
            if (lc_active && !connector && d_t > rx_acc_.lane_change_lane_end_cap_release_abs_d_m) {
              const double elapsed = std::max(0.0, tau0 - t_sw);
              const double lim = std::min(std::max((dist - rx_acc_.lane_change_lane_end_margin_m) / std::max(0.5, 4.2 - elapsed),
                                                   std::min(rx_acc_.lane_change_lane_end_min_cap_mps, cruise_k)), cruise_k);
              if (lim < raw) {
                raw = lim;
                soft_bind = false;
                bind_c = std::numeric_limits<double>::quiet_NaN();
              }
            }
            if (!(sp.change && tau0 >= t_sw)) {
              double cap = std::numeric_limits<double>::infinity();
              if (sp.action == BehaviorAction::kHoldThenChange) cap = v_e0;
              else if (sp.action == BehaviorAction::kYieldDecelThenChange) cap = std::max(yield_floor, v_e0 + sp.prep_a * tau0);
              else if (sp.action == BehaviorAction::kAccelThenChange) cap = std::min(cruise_k, v_e0 + sp.prep_a * tau0);
              if (cap < raw) {
                raw = cap;
                soft_bind = false;
                bind_c = std::numeric_limits<double>::quiet_NaN();
              }
            }
            if (connector && lc_active) {
              const int tgt = std::clamp(t.target_lane_id, 0, 3);
              const double cap_conn = tgt < static_cast<int>(stopped_launch_connector_speed_cap_by_target_lane_mps_.size())
                  ? stopped_launch_connector_speed_cap_by_target_lane_mps_[static_cast<std::size_t>(tgt)]
                  : stopped_launch_connector_speed_cap_mps_;
              if (rx_.connector_cap && cap_conn < raw) raw = cap_conn;
            }
            const double boost = std::clamp((c_max - rx_acc_.closing_decel_boost_start_mps) /
                                                 (rx_acc_.closing_decel_boost_full_mps - rx_acc_.closing_decel_boost_start_mps), 0.0, 1.0);
            double dec = hard ? 12.0 : rx_.acc_dn + boost * (11.0 - rx_.acc_dn);
            if (soft_bind) dec = std::min(dec, 1.5);
            if (rx_acc_.lead_required_decel_limit_enabled && !std::isnan(bind_c) && !hard) {
              const double margin = bind_g - rx_acc_.emergency_gap_m - rx_acc_.lead_required_decel_limit_headway_sec * v;
              if (margin > 0.5) dec = std::min(dec, std::max(1.5, rx_acc_.lead_required_decel_limit_gain * bind_c * bind_c / (2.0 * margin)));
            }
            if (hard && rx_acc_.hard_brake_bypass_filter) f = raw;
            else f = std::clamp(raw, f - dec * dt, f + rx_.acc_up * dt);
            f = std::min(std::max(f, 0.0), cruise_k);
            tf_hard[static_cast<std::size_t>(m * K + k)] = entry ? 1 : 0;
            // ---- 자차 응답 ----
            const double ff = (f - ref_prev) / dt;
            ref_prev = f;
            const double a_des = std::clamp((f - v) / rx_.plant_tau + ff, f < 0.01 ? rx_.plant_a_stop : rx_.plant_a_min, rx_.plant_a_max);
            a += std::clamp(a_des - a, -rx_.plant_jerk * dt, rx_.plant_jerk * dt);
            if (v <= 0.0 && a < 0.0) a = 0.0;
            const double vn = std::max(0.0, v + a * dt);
            s += 0.5 * (v + vn) * dt;
            v = vn;
            for (int i = 0; i < V; ++i) {
              const std::size_t ui = static_cast<std::size_t>(i);
              if (tvs[ui].role == "ctx") {
                xs[ui] += xv[ui] * dt;
              } else {
                xs[ui] = ns[ui];
                xv[ui] = nv[ui];
              }
              res.tv_s[ui][static_cast<std::size_t>(m * K + k)] = xs[ui];
              if (rx_.passed_front_as_rear && (tvs[ui].role == "tf" || tvs[ui].role == "tf2") && s > xs[ui]) passed[ui] = 1;
              passed_mk[ui][static_cast<std::size_t>(m * K + k)] = passed[ui];
            }
            res.ego_s[static_cast<std::size_t>(m * K + k)] = s;
            res.ego_d[static_cast<std::size_t>(m * K + k)] = y;
            res.ego_v[static_cast<std::size_t>(m * K + k)] = v;
          }
          tau_m_row[static_cast<std::size_t>(m)] = tau_m;
          ds_sw_row[static_cast<std::size_t>(m)] = ds_sw;
          ret_row[static_cast<std::size_t>(m)] = ret_t;
          if (q == 1) res.allowed[static_cast<std::size_t>(m)] = std::isfinite(ret_t) ? 1 : 0;
        }
      }
    }

    // ---- 유효성(계속 대응·첫 가지 기준) ----
    for (int n = 0; n < N; ++n) {
      const RxSpec& sp = specs[static_cast<std::size_t>(n)];
      const std::size_t last = static_cast<std::size_t>(n * K + K - 1);
      bool ok = true;
      if (sp.change) {
        if (!mid && std::isnan(tau_m_row[static_cast<std::size_t>(n)])) ok = false;
        if (!mid && connector && !std::isnan(ds_sw_row[static_cast<std::size_t>(n)]) &&
            ds_sw_row[static_cast<std::size_t>(n)] > stopped_launch_connector_max_preswitch_distance_m_) ok = false;
        if (std::abs(res.ego_d[last]) > ego_reference_preview_target_completion_max_abs_d_m_ && res.ego_s[last] <= tgt_end) ok = false;
      } else if (res.ego_s[last] > src_end + 1e-6) {
        ok = false;
      }
      res.valid[static_cast<std::size_t>(n)] = ok ? 1 : 0;
    }

    // ---- 위험: 지도 좌표로 옮겨 collisionProbability ----
    const double sig_cap = rx_.follow_sigma_cap_s;
    for (int m = 0; m < M; ++m) {
      const int b = m / (2 * N);
      const int n = m % N;
      const bool brake_row = br[static_cast<std::size_t>(b)].front == "brake";
      for (int k = 0; k < K; ++k) {
        const std::size_t mk = static_cast<std::size_t>(m * K + k);
        const double tau = dt * (k + 1);
        if (res.ego_s[static_cast<std::size_t>(n * K + k)] > tgt_end + 1e-6) continue;   // 목표 차선 CSV 끝 뒤는 보지 않음
        const auto es = target_lane->sampleExtended(ego_proj.s + res.ego_s[mk], res.ego_d[mk], tv_lane_projection_extension_before_m_);
        if (!es.valid) continue;
        PredPose ego;
        ego.valid = true;
        ego.x = es.x;
        ego.y = es.y;
        ego.yaw = es.yaw;
        double no_coll = 1.0;
        for (int i = 0; i < V; ++i) {
          const auto& x = tvs[static_cast<std::size_t>(i)];
          if (x.role == "ctx") continue;
          const auto* lane = laneFor(x.v.lane_id);
          if (!lane) continue;
          const double sv = x.v.s + (res.tv_s[static_cast<std::size_t>(i)][mk] - x.v.delta_s);
          if (sv > lane->length() + 1e-6) continue;
          const double dv = x.v.d * std::exp(-tau / std::max(0.1, lane_center_decay_tau_sec_));
          const auto ts = lane->sampleExtended(sv, dv, tv_lane_projection_extension_before_m_);
          if (!ts.valid) continue;
          PredPose tv;
          tv.valid = true;
          tv.x = ts.x;
          tv.y = ts.y;
          tv.yaw = ts.yaw;
          const bool pf = passed_mk[static_cast<std::size_t>(i)][mk] != 0;
          const bool front = (x.role == "cf" || x.role == "tf") && !pf;
          ego.t = (x.role == "cf" || ((x.role == "tf" || x.role == "tf2") && tf_hard[mk] && !pf)) ? std::min(tau, sig_cap) : tau;
          TvMode mode{front && brake_row ? "front_brake" : "nominal", 1.0, 0.0, 0.0};
          const auto c = collisionProbability(ego, tv, x.v, mode);
          no_coll *= 1.0 - c.probability;
        }
        res.risk[mk] = 1.0 - no_coll;
      }
    }
    return res;
  }

  double interpReturnProgress(double tau_scaled) const {
    // tree.py RETURN_PROGRESS (0.1 s 표) 와 같은 표를 쓴다.
    static const std::vector<double> kP{
        0.0000, 0.0005, 0.0019, 0.0041, 0.0068, 0.0100, 0.0151, 0.0233, 0.0339, 0.0464, 0.0600, 0.0770, 0.0992,
        0.1255, 0.1551, 0.1867, 0.2194, 0.2521, 0.2838, 0.3135, 0.3400, 0.3641, 0.3871, 0.4093, 0.4309, 0.4521,
        0.4732, 0.4943, 0.5156, 0.5375, 0.5600, 0.5833, 0.6073, 0.6317, 0.6563, 0.6811, 0.7058, 0.7302, 0.7541,
        0.7775, 0.8000, 0.8219, 0.8432, 0.8634, 0.8825, 0.9026, 0.9228, 0.9419, 0.9586, 0.9717, 0.9800, 0.9853,
        0.9901, 0.9941, 0.9973, 0.9993, 1.0000};
    const double x = std::max(0.0, tau_scaled) / 0.1;
    const std::size_t i = static_cast<std::size_t>(std::floor(x));
    if (i + 1 >= kP.size()) return 1.0;
    const double w = x - static_cast<double>(i);
    return kP[i] * (1.0 - w) + kP[i + 1] * w;
  }

  std::vector<RxBranch> rxBranches(const smpc_lane_change::TargetVehicleSet& t, double p_rear) const {
    if (reactive_single_) return {{"yield", "base", "", 1.0}};
    const auto tvs = rxVehicles(t);
    bool rears = false, fronts = false;
    for (const auto& x : tvs) {
      rears |= x.role == "rear" || (rx_.passed_front_as_rear && (x.role == "tf" || x.role == "tf2"));
      fronts |= x.role == "cf" || x.role == "tf";
    }
    const double fl = rx_.p_exc_floor;
    const double pr = p_rear >= 0.0 ? p_rear : fl;
    std::vector<std::tuple<std::string, double, std::string>> rb, fb;
    if (rears) rb = {{"yield", 1.0 - pr, ""}, {"nonyield", pr, "rear"}};
    else rb = {{"yield", 1.0, ""}};
    if (fronts) fb = {{"base", 1.0 - 2.0 * fl, ""}, {"brake", fl, "front"}, {"accel", fl, ""}};
    else fb = {{"base", 1.0, ""}};
    std::vector<RxBranch> out;
    for (const auto& r : rb) {
      for (const auto& f : fb) {
        out.push_back({std::get<0>(r), std::get<0>(f), !std::get<2>(f).empty() ? std::get<2>(f) : std::get<2>(r),
                       std::get<1>(r) * std::get<1>(f)});
      }
    }
    return out;
  }

  // tree_reactive.py _solve: 루트마다 가지별 대응(계속/복귀) 조합 중 Σ p_b r_b(k) ≤ ε 를 지키는 최소 기대 비용.
  TreeResult rxSolveRoot(RxResult& r, const std::vector<RxBranch>& br, int n, double root_cost, double eps) const {
    TreeResult out;
    const int B = static_cast<int>(br.size());
    std::vector<int> opts_count(static_cast<std::size_t>(B));
    std::size_t combos = 1;
    for (int b = 0; b < B; ++b) {
      opts_count[static_cast<std::size_t>(b)] = br[static_cast<std::size_t>(b)].tb.empty() ? 1 : 2;
      combos *= static_cast<std::size_t>(opts_count[static_cast<std::size_t>(b)]);
    }
    const double cost_return = tree_response_cost_[kTreeReturn];
    double best_peak_any = std::numeric_limits<double>::infinity();
    for (std::size_t c = 0; c < combos; ++c) {
      std::size_t rem = c;
      std::vector<int> q(static_cast<std::size_t>(B));
      bool ok = true;
      double jc = 0.0;
      for (int b = B - 1; b >= 0; --b) {          // itertools.product 순서(마지막 가지가 가장 빨리 바뀜)
        const std::size_t oc = static_cast<std::size_t>(opts_count[static_cast<std::size_t>(b)]);
        q[static_cast<std::size_t>(b)] = static_cast<int>(rem % oc);
        rem /= oc;
      }
      for (int b = 0; b < B; ++b) {
        const int qq = q[static_cast<std::size_t>(b)];
        if (!r.allowed[static_cast<std::size_t>((b * 2 + qq) * r.N + n)]) ok = false;
        jc += br[static_cast<std::size_t>(b)].p * (qq == 1 ? cost_return : 0.0);
      }
      if (!ok) continue;
      double peak = 0.0;
      for (int k = 0; k < r.K; ++k) {
        double rk = 0.0;
        for (int b = 0; b < B; ++b) rk += br[static_cast<std::size_t>(b)].p * r.at(b, q[static_cast<std::size_t>(b)], n, k);
        peak = std::max(peak, rk);
      }
      best_peak_any = std::min(best_peak_any, peak);
      const double total = root_cost + jc + tree_risk_weight_ * peak;
      if (peak <= eps && total < out.cost) {
        out.feasible = true;
        out.cost = total;
        out.peak_risk = peak;
        out.combo = q;
      }
    }
    if (!out.feasible && std::isfinite(best_peak_any)) out.peak_risk = best_peak_any;
    return out;
  }

  RxSpec rxSpecOf(const ActionSpec& a) const {
    RxSpec s;
    s.change = a.change_left;
    s.t_sw = a.change_left ? effectiveLaneChangeStartSec(a) : std::numeric_limits<double>::infinity();
    s.dur = std::max(0.1, a.lane_change_duration_sec);
    s.action = a.action;
    s.prep_a = a.prep_accel_mps2;
    return s;
  }

  bool rxConnectorNow(const smpc_lane_change::TargetVehicleSet& t) const {
    return t.distance_to_lane_end <= 30.0 && ego_speed_mps_ <= 1.0;
  }

  // 지금 승인하면 waypoint 가 쓸 방식. 맞는 미리보기가 있으면 그 mode(waypoint 가 그대로 실행), 없으면 추정.
  bool rxConnectorForAccept(const smpc_lane_change::TargetVehicleSet& t) const {
    if (referencePreviewFreshFor(t)) return reference_preview_.mode == waypoint_system::PathSwitchPreview::CONNECTOR;
    return rxConnectorNow(t);
  }

  // 실행 중 모니터 (tree_reactive.py monitor_core + _update_rear_belief).
  // connector = 이 차선 변경의 실행 방식. 실행 중에는 승인 때 고정한 값을 넘긴다: waypoint 는 connector 를 끝까지
  // 따라가는데 속도(≤ 1 m/s)로 다시 추정하면 출발 직후 blend(시간 비례 횡이동, cap 없음)로 바뀌어 위험이 튄다
  // (MORAI H_s50_d60 v7: 출발 0.6 s 뒤 0.05 → 0.43, 정지 출발–복귀 6 회).
  bool rxMonitorReturn(const smpc_lane_change::TargetVehicleSet& t, double now, std::string* detail, bool connector) {
    const double W = lane_width_m_;
    const double d_t = std::abs(t.ego_d_target);
    // 뒤차 관측 갱신
    const double intr = 0.5 * W + 0.5 * ego_width_m_ - d_t;
    if (intr < rx_.rear_react_intrusion_m) {
      rx_intr_since_ = -1.0;
      rx_p_rear_post_ = -1.0;
    } else {
      if (rx_intr_since_ < 0.0) rx_intr_since_ = now;
      const auto tvs = rxVehicles(t);
      const RxTv* rear = nullptr;
      for (const auto& x : tvs) if (x.role == "rear") { rear = &x; break; }
      if (rear && now - rx_intr_since_ >= rx_.rear_react_delay_s + rx_.rear_detect_s) {
        if (rear->v.accel <= -0.5) rx_p_rear_post_ = rx_.p_exc_floor * 0.2;
        else if (rear->v.accel > -0.1 && rear->v.v_long > ego_speed_mps_ + 0.5) rx_p_rear_post_ = 0.9;
      }
    }
    if (d_t <= tree_abort_before_abs_d_m_) return false;
    const double T_rem = std::max(0.5, egoChangeBlendDuration() * d_t / W);
    const double join = (have_reference_preview_ && reference_preview_.mode == waypoint_system::PathSwitchPreview::CONNECTOR &&
                         std::isfinite(reference_preview_.connector_join_arc_m)) ? reference_preview_.connector_join_arc_m : 20.0;
    auto br = rxBranches(t, rx_p_rear_post_);
    for (auto& b : br) b.tb = "now";
    RxSpec go;
    go.change = true;
    go.t_sw = 0.0;
    go.dur = T_rem;
    go.action = BehaviorAction::kChangeNow;
    RxResult r = rxRollout(t, {go}, br, ego_speed_mps_, currentTargetSpeedMps(), connector, join, ego_speed_mps_, T_rem);
    double go_peak = 0.0, back_peak = 0.0;
    bool back_ok = true;
    for (int b = 0; b < r.B; ++b) back_ok &= r.allowed[static_cast<std::size_t>((b * 2 + 1) * r.N)] != 0;
    for (int k = 0; k < r.K; ++k) {
      double g = 0.0, bk = 0.0;
      for (int b = 0; b < r.B; ++b) {
        g += br[static_cast<std::size_t>(b)].p * r.at(b, 0, 0, k);
        bk += br[static_cast<std::size_t>(b)].p * r.at(b, 1, 0, k);
      }
      go_peak = std::max(go_peak, g);
      back_peak = std::max(back_peak, bk);
    }
    const double eps = activeRiskEpsilon(t);
    // 계속 쪽 위험. 기본은 '끝까지 그냥 계속'(가지마다 나중 대응 없음). monitor_tree 면 승인(rxSolveRoot)처럼 가지별
    // 나중 대응을 넣는다: 앞차 급감속은 front_detect_s 뒤에 보이면 복귀, 뒤차 비양보는 이미 관측했으면 지금, 아니면
    // 침범 + 반응 지연 + 알아채는 시간 뒤 복귀(실행 중 침범이 이미 시작됐어도 지금부터 센다: 보수적). 대응 없이 계속이면
    // 앞차가 가까울 때 급감속 가지(p 0.05) 하나가 충돌 ≈ 1 이라 0.05 가 깔려, 승인은 괜찮다고 본 장면을 모니터가 2ε 근처로 본다.
    double cont_peak = go_peak;
    if (rx_.monitor_tree) {
      auto br2 = rxBranches(t, rx_p_rear_post_);
      for (auto& b : br2) {
        if (b.tb == "rear" && rx_p_rear_post_ >= 0.0) b.tb = "now";
      }
      RxResult r2 = rxRollout(t, {go}, br2, ego_speed_mps_, currentTargetSpeedMps(), connector, join, ego_speed_mps_, T_rem);
      cont_peak = rxSolveRoot(r2, br2, 0, 0.0, eps).peak_risk;
    }
    const bool ret = back_ok && cont_peak > tree_abort_risk_factor_ * eps && back_peak < cont_peak;
    if (detail) {
      std::ostringstream os;
      os.setf(std::ios::fixed);
      os.precision(4);
      os << "rx_monitor go_peak=" << go_peak << " back_peak=" << back_peak << " back_ok=" << back_ok
         << " p_rear_post=" << rx_p_rear_post_;
      if (rx_.monitor_tree) os << " cont_peak=" << cont_peak;
      *detail = os.str();
    }
    return ret;
  }

  // treeSelect 의 반응형 판(tree_reactive.py decide 의 평가 부분).
  void treeSelectReactive(const smpc_lane_change::TargetVehicleSet& t,
                          const ActionSpec& change_now_spec, bool common_ok, double target_gap_cost,
                          double allowed_risk, bool* request_change, CandidateEval* best_future,
                          bool* have_future, std::string* detail) {
    *request_change = false;
    *have_future = false;
    if (!reactive_drop_common_ && !common_ok) {
      if (detail) *detail = "rx common_gate_fail";
      return;
    }
    const ros::WallTime wall0 = ros::WallTime::now();
    std::vector<ActionSpec> specs;
    specs.push_back(change_now_spec);
    for (const auto& c : buildBehaviorCandidates(t)) {
      if (c.action != BehaviorAction::kChangeNow && candidateFitsHorizon(c)) specs.push_back(c);
    }
    std::vector<RxSpec> rs;
    std::vector<uint8_t> compat(specs.size(), 1);
    const double base_cap = egoPredictionTargetSpeedMps();
    for (std::size_t i = 0; i < specs.size(); ++i) {
      rs.push_back(rxSpecOf(specs[i]));
      if (i > 0 && !preparationLongitudinalModelCompatible(t, specs[i], base_cap)) compat[i] = 0;
    }
    const bool connector = rxConnectorForAccept(t);
    const double join = (have_reference_preview_ && reference_preview_.mode == waypoint_system::PathSwitchPreview::CONNECTOR &&
                         std::isfinite(reference_preview_.connector_join_arc_m)) ? reference_preview_.connector_join_arc_m : 20.0;
    const auto br = rxBranches(t, -1.0);
    RxResult r = rxRollout(t, rs, br, ego_speed_mps_, currentTargetSpeedMps(), connector, join, ego_speed_mps_, -1.0);
    // 승인–모니터 일치: 같은 상태에서 모니터가 곧바로 복귀를 고르면 지금 바꾸기는 승인하지 않는다.
    bool now_monitor_return = false;
    if (tree_monitor_enabled_) {
      const double keep_intr = rx_intr_since_, keep_p = rx_p_rear_post_;
      now_monitor_return = rxMonitorReturn(t, ros::Time::now().toSec(), nullptr, connector);
      rx_intr_since_ = keep_intr;
      rx_p_rear_post_ = keep_p;
    }
    // 기존 경로(buildEgoTrajectory)처럼 맞는 path-switch 미리보기가 없으면 지금 바꾸기는 승인하지 않는다. 미리보기 없이
    // 명령하면 waypoint 가 blend 없이 곧바로 경로를 바꿔(약 2.2 s) 예측한 blend(약 4.5 s)보다 훨씬 빨리 옆 차선에 들어간다
    // (MORAI H_s30_d100: 직전 차선 변경이 끝난 직후 미리보기가 다시 맞춰지기 전에 승인, 옆 차와 접촉).
    const bool preview_missing = ego_reference_preview_enabled_ && ego_reference_preview_required_for_change_ &&
        !referencePreviewFreshFor(t);
    auto root_cost_of = [&](const ActionSpec& sp) {
      return lane_change_base_cost_ + target_gap_cost +
          behavior_delay_cost_weight_ * std::max(0.0, sp.lane_change_start_sec) +
          behavior_accel_cost_weight_ * sp.prep_accel_mps2 * sp.prep_accel_mps2;
    };
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(3);
    os << "rx br=";
    for (const auto& b : br) os << b.rear << "/" << b.front << ":" << b.p << ",";
    double best_cost = std::numeric_limits<double>::infinity();
    int best_index = -1;
    std::vector<int> best_combo;
    for (std::size_t i = 0; i < specs.size(); ++i) {
      if (!r.valid[i] || !compat[i]) continue;
      if (i == 0 && now_monitor_return) {
        os << " change_now_monitor_return";
        continue;
      }
      if (i == 0 && preview_missing) {
        os << " change_now_no_preview";
        continue;
      }
      const TreeResult tr = rxSolveRoot(r, br, static_cast<int>(i), root_cost_of(specs[i]), allowed_risk);
      if (tr.feasible && tr.cost < best_cost) {
        best_cost = tr.cost;
        best_index = static_cast<int>(i);
        best_combo = tr.combo;
      }
    }
    os << " roots=" << specs.size() << " ms=" << (ros::WallTime::now() - wall0).toSec() * 1000.0;
    if (best_index < 0) {
      os << " none_feasible";
      if (detail) *detail = os.str();
      return;
    }
    os << " pick=" << specs[static_cast<std::size_t>(best_index)].name << " cost=" << best_cost << " resp=";
    for (const int q : best_combo) os << q;
    if (detail) *detail = os.str();
    if (best_index == 0) {
      *request_change = true;
      rx_lc_connector_ = connector;
      return;
    }
    std::vector<smpc_lane_change::TargetVehicle> vehicles;
    for (const auto& x : rxVehicles(t)) if (x.role != "ctx") vehicles.push_back(x.v);
    CandidateEval ev = evaluateCandidate(t, vehicles, specs[static_cast<std::size_t>(best_index)], allowed_risk, target_gap_cost);
    ev.feasible = true;
    ev.cost = best_cost;
    *best_future = ev;
    *have_future = true;
  }

  void laneChangeStatusCallback(const smpc_lane_change::LaneChangeStatus::ConstPtr& msg) {
    lc_status_ = *msg;
    have_lc_status_ = true;
  }

  void timerCallback(const ros::TimerEvent&) {
    if (!enabled_ || !have_ego_) return;

    const bool fresh_targets = have_targets_ &&
        (ros::Time::now() - targets_stamp_).toSec() <= target_timeout_sec_;
    if (!fresh_targets) return;

    const auto& t = targets_;
    ActionSpec keep_spec;
    keep_spec.action = BehaviorAction::kKeep;
    keep_spec.name = "keep";
    CandidateEval keep_candidate;
    keep_candidate.spec = keep_spec;
    keep_candidate.target_speed_mps = egoPredictionTargetSpeedMps();
    keep_candidate.cost = 0.0;

    if (!lanes_loaded_) {
      publishBehaviorOutputs(t, keep_candidate, false, false, 1.0,
                             "smpc_v3_no_lane_csv_keep");
      publishKeep(t, "smpc_v3_no_lane_csv_keep", 0.0, lane_end_emergency_cost_, 1.0);
      publishDebug("v3 disabled: lane CSVs not loaded");
      return;
    }

    auto vehicles = uniqueVehicles(t);
    const bool passing_front = targetFrontIsPassingVehicle(t);
    const int passing_vehicle_id =
        passing_front ? t.target_front.unique_id : -1;
    const auto* effective_target_rear =
        targetRearForGap(t, passing_vehicle_id);
    if (effective_target_rear) {
      const bool already_present = std::any_of(
          vehicles.begin(), vehicles.end(),
          [&](const smpc_lane_change::TargetVehicle& v) {
            return v.unique_id == effective_target_rear->unique_id;
          });
      if (!already_present) vehicles.push_back(*effective_target_rear);
    }
    if ((tree_planner_enabled_ || baseline_tree_monitor_) && tree_monitor_enabled_) {
      // 차선 변경 중 재계획(rear_abort 대체). supervisor 가 tree_abort_enabled 일 때 이 값을 따른다.
      std_msgs::Bool abort_msg;
      abort_msg.data = false;
      if (have_lc_status_ && lc_status_.active &&
          lc_status_.source_lane_id == t.current_lane_id &&
          lc_status_.target_lane_id == t.target_lane_id) {
        std::string detail;
        abort_msg.data = reactive_tree_enabled_
            ? rxMonitorReturn(t, ros::Time::now().toSec(), &detail, rx_lc_connector_)
            : treeMonitorReturn(t, vehicles, effective_target_rear, activeRiskEpsilon(t), &detail);
        std_msgs::String dm;
        dm.data = std::string("tree_monitor return=") + (abort_msg.data ? "1 " : "0 ") + detail;
        tree_debug_pub_.publish(dm);
      }
      tree_abort_pub_.publish(abort_msg);
    }
    ActionSpec change_now_spec;
    change_now_spec.action = BehaviorAction::kChangeNow;
    change_now_spec.lane_change_start_sec = 0.0;
    change_now_spec.lane_change_duration_sec = egoChangeBlendDuration();
    change_now_spec.change_left = true;
    change_now_spec.name = "change_now";

    const auto keep_traj = buildEgoTrajectory(t, keep_spec);
    const auto change_traj = buildEgoTrajectory(t, change_now_spec);
    // 이후 behavior 후보 평가가 같은 mutable 필드를 덮어쓰므로 여기서 잡아둔다.
    const std::string change_traj_build_detail = ego_traj_build_detail_;
    publishPredictionPath(keep_traj, keep_path_pub_, t.header);
    publishPredictionPath(change_traj, change_path_pub_, t.header);
    publishPredictionMarkers(t, keep_traj, change_traj, vehicles);

    const TrajectoryEval keep_eval = evaluateTrajectory(keep_traj, vehicles, &t, false);
    const TrajectoryEval change_eval = evaluateTrajectory(change_traj, vehicles, &t, true);
    keep_candidate.trajectory = keep_eval;

    const double current_safe_gap = std::max(
        current_front_safe_gap_m_, standstill_gap_m_ + time_headway_sec_ * ego_speed_mps_);
    const double current_gap = t.current_front.valid
        ? frontBumperGap(t.current_front)
        : std::numeric_limits<double>::infinity();
    const double target_front_gap = t.target_front.valid
        ? frontBumperGap(t.target_front)
        : std::numeric_limits<double>::infinity();
    const double target_rear_gap = effective_target_rear
        ? rearBumperGap(*effective_target_rear)
        : std::numeric_limits<double>::infinity();

    const bool current_front_priority =
        egoLanePriorityEligible(t, t.current_front, false);
    const double effective_current_gap = current_front_priority
        ? std::numeric_limits<double>::infinity()
        : current_gap;
    const bool current_blocked = t.current_front.valid &&
        !current_front_priority &&
        current_gap < current_front_trigger_gap_m_;
    const bool endpoint_trigger =
        t.lane_change_prepare || t.lane_change_urgent || t.emergency_stop_required;
    const double allowed_risk = activeRiskEpsilon(t);
    const double active_target_front_safe_gap = activeTargetFrontSafeGap(t);
    const double active_target_rear_safe_gap = activeTargetRearSafeGap(t);
    const double effective_target_rear_safe_gap =
        effective_target_rear
            ? targetRearRequiredGap(t, *effective_target_rear,
                                    egoChangeBlendDuration(), ego_speed_mps_,
                                    ros::Time::now())
            : active_target_rear_safe_gap;
    const bool target_front_ttc_safe =
        passing_front || targetFrontTtcSafe(t.target_front);
    const bool target_front_headway_safe =
        passing_front || targetFrontHeadwaySafe(t.target_front);
    const bool target_rear_ttc_safe =
        !effective_target_rear || targetRearTtcSafe(*effective_target_rear);
    const bool target_rear_headway_safe =
        !effective_target_rear || targetRearHeadwaySafe(*effective_target_rear);
    const bool target_rear_priority =
        effective_target_rear &&
        effective_target_rear_safe_gap < active_target_rear_safe_gap;

    const bool target_lane_is_left = t.target_lane_id == t.current_lane_id + 1;
    // A passing vehicle waives the full front gap, but it must already be
    // ahead of the ego front by the entry clearance now.  The projected gate
    // only checks steps after the ego reaches the target corridor, when the
    // passer is far ahead, so it never limited this: lc_gt 2026-09-15 0->1
    // merges were approved with the passer still 3 m alongside (GT min
    // clearance 0.70-1.56 m).
    const double pass_entry_clearance_now =
        std::max(0.0, pass_gap_entry_front_clearance_m_) +
        std::max(0.0, pass_gap_entry_length_margin_m_) +
        std::max(0.0, pass_gap_entry_relative_speed_time_sec_) *
            std::max(0.0, t.target_front.v_long - ego_speed_mps_);
    const bool deterministic_gap_safe =
        (passing_front
             ? target_front_gap >= pass_entry_clearance_now
             : (!t.target_front.valid ||
                target_front_gap > active_target_front_safe_gap)) &&
        (!effective_target_rear ||
         target_rear_gap > effective_target_rear_safe_gap) &&
        target_front_ttc_safe && target_front_headway_safe &&
        target_rear_ttc_safe && target_rear_headway_safe;
    const bool pass_gap_scene_observed =
        targetLanePairObservedForPassGap(t, effective_target_rear);
    const bool target_lane_scene_observed =
        targetLaneSceneObservedForCommit(t) || pass_gap_scene_observed;
    const bool observed_target_lane_intrusion_safe =
        observedTargetLaneIntrusionSafe(t, passing_vehicle_id);
    const bool change_time_risk_safe = timeIndexedRiskSafe(change_eval, allowed_risk);
    const bool change_legacy_risk_safe = change_eval.valid &&
        (!behavior_require_legacy_risk_gate_ || change_eval.risk <= allowed_risk);
    const bool change_clearance_safe = clearanceSafe(change_eval);
    // 기존 판단 + 게이트 정리(2×2 의 A/B): 확률 위험 제약만 남긴다. 디버그 플래그는 원래 값을 낸다.
    const bool baseline_drop_gates = !tree_planner_enabled_ && !baseline_keep_gates_;
    const bool chance_gap_safe = change_eval.valid && change_time_risk_safe &&
        change_legacy_risk_safe && (change_clearance_safe || baseline_drop_gates);
    // CHANGE_NOW must satisfy the same all-step target-lane front/rear
    // gap, TTC, and rear-headway gate as a delayed behavior candidate.  An
    // instantaneous gate alone is unsafe for a fast target-rear vehicle:
    // it can be acceptable at t=0 but close the gap while the ego reference
    // is entering the lane.
    // 호출 직전에 반드시 지운다.  change_eval.valid 가 false 면 단축 평가로
    // 호출이 생략되는데, 그러면 후보 평가(evaluateCandidate)가 다른 spec 으로
    // 남긴 기록을 CHANGE_NOW 의 것으로 잘못 읽게 된다.
    proj_fail_ = ProjGateFail{};
    const bool change_projected_gap_ttc_safe = change_eval.valid &&
        projectedTargetGapTtcSafe(t, change_traj, change_now_spec,
                                  effective_target_rear, passing_front);
    const ProjGateFail change_proj_fail = proj_fail_;
    const bool change_acc_stable = accStableForChangeCommit();
    const bool change_feasible =
        target_lane_is_left && change_eval.valid && (deterministic_gap_safe || baseline_drop_gates) &&
        chance_gap_safe && (change_projected_gap_ttc_safe || baseline_drop_gates) &&
        target_lane_scene_observed && observed_target_lane_intrusion_safe &&
        change_acc_stable;

    double keep_cost = risk_cost_weight_ * keep_eval.risk +
        gapPenalty(effective_current_gap, current_safe_gap) +
        laneEndPressureCost(t);
    if (!keep_eval.valid) keep_cost += lane_end_emergency_cost_;
    if (current_blocked) keep_cost += blocked_cost_weight_;

    const double target_gap_cost =
        gapPenalty(target_front_gap, active_target_front_safe_gap) +
        gapPenalty(target_rear_gap, effective_target_rear_safe_gap);
    double change_cost = lane_change_base_cost_ +
        risk_cost_weight_ * change_eval.risk +
        target_gap_cost;
    if (!target_lane_is_left) change_cost += lane_end_emergency_cost_;
    if (!change_eval.valid) change_cost += lane_end_emergency_cost_;
    if (!deterministic_gap_safe && !baseline_drop_gates) change_cost += lane_end_urgent_cost_;
    if (!chance_gap_safe) change_cost += lane_end_urgent_cost_;
    if (!change_projected_gap_ttc_safe && !baseline_drop_gates) change_cost += lane_end_urgent_cost_;

    const bool desire_change = endpoint_trigger || current_blocked ||
        (change_cost + decision_hysteresis_cost_ < keep_cost);
    bool request_change = desire_change && change_feasible;

    CandidateEval change_now_candidate;
    change_now_candidate.spec = change_now_spec;
    change_now_candidate.trajectory = change_eval;
    change_now_candidate.target_speed_mps = egoPredictionTargetSpeedMps();
    change_now_candidate.time_risk_safe = change_time_risk_safe;
    change_now_candidate.legacy_risk_safe = change_legacy_risk_safe;
    change_now_candidate.clearance_safe = change_clearance_safe;
    change_now_candidate.projected_gap_ttc_safe = change_projected_gap_ttc_safe;
    change_now_candidate.feasible = change_feasible;
    change_now_candidate.cost = change_cost;

    // Behavior policy is deliberately lexicographic, not only cost-based:
    // 1) CHANGE_NOW at the current longitudinal state (handled above),
    // 2) HOLD_THEN_CHANGE if any safe hold-speed candidate exists,
    // 3) YIELD_DECEL_THEN_CHANGE only when no hold-speed candidate exists.
    // This prevents an earlier YIELD candidate from displacing a later but
    // safe speed-maintaining lane change merely because its scalar cost is
    // lower.  Cost still selects the best t_LC within one action tier.
    CandidateEval best_hold_candidate;
    CandidateEval best_yield_candidate;
    CandidateEval best_accel_candidate;
    bool have_hold_candidate = false;
    bool have_yield_candidate = false;
    bool have_accel_candidate = false;
    // Do not issue a longitudinal YIELD/ACCEL command merely because the
    // target-lane scene is still warming up.  Until the rear scene is
    // continuously observed, HOLD/WAIT preserves the ordinary ACC policy and
    // the next receding-horizon cycle will re-evaluate with real data.
    if (desire_change && !change_feasible && target_lane_is_left &&
        target_lane_scene_observed && observed_target_lane_intrusion_safe &&
        change_acc_stable &&
        behavior_planner_enabled_) {
      const auto candidates = buildBehaviorCandidates(t);
      for (const auto& candidate : candidates) {
        if (candidate.action == BehaviorAction::kChangeNow) continue;
        const CandidateEval evaluated =
            evaluateCandidate(t, vehicles, candidate, allowed_risk, target_gap_cost);
        if (!evaluated.feasible) continue;
        switch (candidate.action) {
          case BehaviorAction::kHoldThenChange:
            if (!have_hold_candidate || evaluated.cost < best_hold_candidate.cost) {
              best_hold_candidate = evaluated;
              have_hold_candidate = true;
            }
            break;
          case BehaviorAction::kYieldDecelThenChange:
            if (!have_yield_candidate || evaluated.cost < best_yield_candidate.cost) {
              best_yield_candidate = evaluated;
              have_yield_candidate = true;
            }
            break;
          case BehaviorAction::kAccelThenChange:
            if (!have_accel_candidate || evaluated.cost < best_accel_candidate.cost) {
              best_accel_candidate = evaluated;
              have_accel_candidate = true;
            }
            break;
          case BehaviorAction::kKeep:
          case BehaviorAction::kWaitForGap:
          case BehaviorAction::kChangeNow:
            break;
        }
      }
    }
    CandidateEval best_future_candidate;
    bool have_future_candidate = false;
    bool yield_candidate_confirmation_pending = false;
    const double target_rear_closing = effective_target_rear
        ? conservativeRearClosingSpeed(
              *effective_target_rear, ego_speed_mps_, ros::Time::now())
        : 0.0;
    const bool rear_closing_pressure = effective_target_rear &&
        target_rear_closing >= std::max(0.0, behavior_rear_accel_min_closing_mps_);
    if (rear_closing_pressure && have_accel_candidate) {
      clearYieldCandidateConfirmation();
      best_future_candidate = best_accel_candidate;
      have_future_candidate = true;
    } else if (have_hold_candidate) {
      clearYieldCandidateConfirmation();
      best_future_candidate = best_hold_candidate;
      have_future_candidate = true;
    } else if (!rear_closing_pressure && have_yield_candidate) {
      if (yieldCandidateConfirmed(t)) {
        best_future_candidate = best_yield_candidate;
        have_future_candidate = true;
      } else {
        // Keep observing at the ordinary ACC target.  Do not let a single
        // target-lane association frame create a short brake pulse.
        yield_candidate_confirmation_pending = true;
      }
    } else if (have_accel_candidate) {
      clearYieldCandidateConfirmation();
      best_future_candidate = best_accel_candidate;
      have_future_candidate = true;
    } else {
      clearYieldCandidateConfirmation();
    }
    if (tree_planner_enabled_) {
      // 트리 플래너: G4(확률 위험 + 모든 mode clearance) 와 단계별 선택을 트리 평가로 바꾼다.
      // tree_keep_gates=false 면 G3/G5 도 걷어낸다(연구 방향 2, hard safety 는 트리 안의 위험 제약).
      std::string tree_detail;
      const bool common_ok = target_lane_is_left && target_lane_scene_observed &&
          observed_target_lane_intrusion_safe && change_acc_stable;
      const bool change_now_gates_ok = deterministic_gap_safe && change_projected_gap_ttc_safe;
      bool tree_change = false;
      clearYieldCandidateConfirmation();
      yield_candidate_confirmation_pending = false;
      if (desire_change) {
        if (reactive_tree_enabled_) {
          treeSelectReactive(t, change_now_spec, common_ok, target_gap_cost, allowed_risk, &tree_change,
                             &best_future_candidate, &have_future_candidate, &tree_detail);
        } else {
          treeSelect(t, vehicles, effective_target_rear, passing_front, change_now_spec, change_traj,
                     change_now_gates_ok, common_ok, target_gap_cost, allowed_risk, &tree_change,
                     &best_future_candidate, &have_future_candidate, &tree_detail);
        }
      } else {
        have_future_candidate = false;
      }
      request_change = desire_change && tree_change;
      std_msgs::String tm;
      tm.data = tree_detail;
      tree_debug_pub_.publish(tm);
    }
    const bool future_feasible =
        (tree_planner_enabled_ ? request_change : change_feasible) || have_future_candidate;

    std::ostringstream debug;
    debug.setf(std::ios::fixed);
    debug.precision(3);
    const double p_current_front = t.current_front.valid
        ? adaptiveEventProbability(t.current_front)
        : -1.0;
    const double p_target_front = t.target_front.valid
        ? adaptiveEventProbability(t.target_front)
        : -1.0;
    const double p_target_rear = effective_target_rear
        ? adaptiveEventProbability(*effective_target_rear)
        : -1.0;
    const bool reference_preview_ready = referencePreviewFreshFor(t);
    debug << "v3 risk legacy_keep=" << keep_eval.risk
          << " legacy_change=" << change_eval.risk
          << " step_peak_keep=" << keep_eval.peak_step_risk
          << " change=" << change_eval.peak_step_risk
          << " change_peak_t=" << prediction_dt_sec_ * (change_eval.peak_step + 1)
          << " change_first_violation=" << firstRiskViolationTimeSec(change_eval, allowed_risk)
          << " clear keep=" << keep_eval.min_signed_clearance
          << " change=" << change_eval.min_signed_clearance
          << " p cf=" << p_current_front
          << " tf=" << p_target_front
          << " tr=" << p_target_rear
          << " gaps cur=" << current_gap
          << " tf=" << target_front_gap
          << " tr=" << target_rear_gap
          << " req_gap tf=" << active_target_front_safe_gap
          << " tr=" << active_target_rear_safe_gap
          << " eff_tr=" << effective_target_rear_safe_gap
          << " ttc_ok tf=" << (target_front_ttc_safe ? 1 : 0)
          << " tf_hw=" << (target_front_headway_safe ? 1 : 0)
          << " tr=" << (target_rear_ttc_safe ? 1 : 0)
          << " rear_hw_ok=" << (target_rear_headway_safe ? 1 : 0)
          << " projected_gap_ttc_ok=" << (change_projected_gap_ttc_safe ? 1 : 0)
          << " baseline_gates=" << (baseline_drop_gates ? 0 : 1)
          << " pgtc_fail="
          << (!change_proj_fail.valid
                  ? "none"
                  : (change_proj_fail.front
                         ? (change_proj_fail.ttc ? "front_ttc" : "front_gap")
                         : (change_proj_fail.ttc ? "rear_ttc" : "rear_gap")))
          << " pgtc_t=" << change_proj_fail.t_sec
          << " pgtc_gap=" << change_proj_fail.gap
          << " pgtc_req=" << change_proj_fail.required
          << " pgtc_id=" << change_proj_fail.unique_id
          << " pgtc_mode="
          << (change_proj_fail.mode.empty() ? "-" : change_proj_fail.mode.c_str())
          << " pgtc_egov=" << change_proj_fail.ego_v
          << " pgtc_tvv=" << change_proj_fail.tv_v
          << " pgtc_egoyaw=" << change_proj_fail.ego_yaw
          << " pgtc_tvyaw=" << change_proj_fail.tv_yaw
          << " acc_commit_ok=" << (change_acc_stable ? 1 : 0)
          << " ego_accel=" << ego_longitudinal_accel_mps2_
          << " scene_obs=" << (target_lane_scene_observed ? 1 : 0)
          << " pass_gap=" << (passing_front ? 1 : 0)
          << " pass_scene=" << (pass_gap_scene_observed ? 1 : 0)
          << " pass_id=" << passing_vehicle_id
          << " gap_rear_id="
          << (effective_target_rear ? effective_target_rear->unique_id : -1)
          << " cutin_safe=" << (observed_target_lane_intrusion_safe ? 1 : 0)
          << " low_spd_rear=" << (lowSpeedRearSafetyActive() ? 1 : 0)
          << " risk_eps=" << allowed_risk
          << " rear_priority=" << (target_rear_priority ? 1 : 0)
          << " side_priority=" << (current_front_priority ? 1 : 0)
          << " vehicles=" << vehicles.size()
          << " held=" << held_injected_count_
          << " nearby=" << t.nearby_vehicles.size()
          << " v_ego=" << ego_speed_mps_
          << " rear_closing=" << target_rear_closing
          << " v_ref=" << currentTargetSpeedMps()
          << " v_rollout=" << egoPredictionTargetSpeedMps()
          << " v_post=" << changePostCompletionSpeedMps(t, egoPredictionTargetSpeedMps())
          << " ego_rollout=" << (ego_prediction_use_mission_speed_rollout_ ? 1 : 0)
          << " ref_preview=" << (reference_preview_ready ? 1 : 0)
          << "/" << referencePreviewModeName(reference_preview_.mode)
          << " ego_traj=" << change_traj_build_detail
          << " ref_pts=" << reference_preview_.source.poses.size()
          << ":" << reference_preview_.target.poses.size()
          << ":" << reference_preview_.connector.poses.size()
          << " ref_join=" << reference_preview_.connector_join_arc_m
          << " eval_detail="
          << (change_eval.detail.empty() ? "ok" : change_eval.detail)
          << " cost keep=" << keep_cost
          << " change=" << change_cost
          << " feasible=" << (change_feasible ? 1 : 0)
          << " desire=" << (desire_change ? 1 : 0)
          << " future=" << (future_feasible ? 1 : 0)
          << " yield_confirm_pending=" << (yield_candidate_confirmation_pending ? 1 : 0);
    if (have_future_candidate) {
      debug << " selected=" << behaviorActionName(best_future_candidate.spec.action)
            << " t_lc=" << best_future_candidate.spec.lane_change_start_sec
            << " a_prep=" << best_future_candidate.spec.prep_accel_mps2
            << " v_cap=" << best_future_candidate.target_speed_mps
            << " c=" << best_future_candidate.cost
            << " peak=" << best_future_candidate.trajectory.peak_step_risk;
    }
    publishDebug(debug.str());

    if (request_change) {
      clearYieldCandidateConfirmation();
      std::string decision_reason;
      if (endpoint_trigger) {
        decision_reason = "smpc_v2_lane_end_traj_safe";
      } else if (current_blocked) {
        decision_reason = "smpc_v2_current_front_blocked_traj_safe";
      } else {
        decision_reason = "smpc_v3_lower_expected_traj_cost";
      }
      publishBehaviorOutputs(t, change_now_candidate, true, true, allowed_risk,
                             decision_reason);
      publishChange(t, decision_reason, keep_cost, change_cost, change_eval.risk);
      return;
    }

    if (desire_change && have_future_candidate) {
      const std::string preparation_reason =
          std::string("smpc_v3_prepare_") + behaviorActionName(best_future_candidate.spec.action);
      publishBehaviorOutputs(t, best_future_candidate, false, true, allowed_risk,
                             preparation_reason);
      publishKeep(t, preparation_reason, keep_cost, best_future_candidate.cost,
                  best_future_candidate.trajectory.risk);
      return;
    }

    // 트리 모드에서는 트리가 거절했으면(request_change=false, 미래 후보 없음) 기존 판정과 무관하게 WAIT 를
    // 보내 ACC 차선 끝 정지를 켠다(리뷰 #5). 기존 모드는 dd1cf54 그대로.
    if (desire_change && (tree_planner_enabled_ ? !request_change : !change_feasible)) {
      // Changing is desired, but neither CHANGE_NOW nor a future candidate
      // has passed its safety gates.  This differs from ordinary KEEP_CRUISE:
      // preserve the longitudinal policy and keep observing, without claiming
      // a safe t_LC or treating the lack of a request as a rejection.
      CandidateEval wait_candidate = keep_candidate;
      wait_candidate.spec.action = BehaviorAction::kWaitForGap;
      wait_candidate.spec.prep_accel_mps2 = 0.0;
      wait_candidate.spec.lane_change_start_sec = -1.0;
      wait_candidate.spec.lane_change_duration_sec = 0.0;
      wait_candidate.spec.change_left = false;
      wait_candidate.spec.name = "wait_for_gap";
      wait_candidate.target_speed_mps = egoPredictionTargetSpeedMps();

      std::string wait_reason = "smpc_v3_wait_for_gap_change_unsafe";
      if (tree_planner_enabled_ && change_feasible) {
        wait_reason = "smpc_v3_wait_for_gap_tree_reject";
      } else if (yield_candidate_confirmation_pending) {
        wait_reason = "smpc_v3_wait_for_gap_yield_candidate_unconfirmed";
      } else if (!target_lane_is_left) wait_reason = "smpc_v3_wait_for_gap_invalid_target_lane";
      else if (!change_eval.valid) wait_reason = "smpc_v3_wait_for_gap_no_change_trajectory";
      else if (!change_acc_stable) wait_reason = "smpc_v3_wait_for_acc_stabilization";
      else if (!deterministic_gap_safe) wait_reason = "smpc_v3_wait_for_gap_deterministic_gap_unsafe";
      else if (!chance_gap_safe) wait_reason = "smpc_v3_wait_for_gap_chance_risk_high";
      else if (!change_projected_gap_ttc_safe) {
        wait_reason = "smpc_v3_wait_for_gap_projected_gap_ttc_unsafe";
      } else if (!target_lane_scene_observed) {
        wait_reason = "smpc_v3_wait_for_gap_target_lane_scene_unobserved";
      } else if (!observed_target_lane_intrusion_safe) {
        wait_reason = "smpc_v3_wait_for_gap_observed_target_cutin";
      }
      publishBehaviorOutputs(t, wait_candidate, false, false, allowed_risk, wait_reason);
      publishKeep(t, wait_reason, keep_cost, change_cost, change_eval.risk);
      return;
    }

    publishBehaviorOutputs(t, keep_candidate, change_feasible, future_feasible,
                           allowed_risk, "smpc_v3_keep_lower_expected_traj_cost");
    publishKeep(t, "smpc_v3_keep_lower_expected_traj_cost",
                keep_cost, change_cost, keep_eval.risk);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber targets_sub_, odom_sub_, mission_speed_sub_, acc_target_speed_sub_,
      behavior_status_sub_, reference_preview_sub_;
  ros::Publisher decision_pub_, behavior_plan_pub_, behavior_request_pub_;
  ros::Publisher debug_pub_, debug_marker_pub_, prediction_marker_pub_;
  ros::Publisher keep_path_pub_, change_path_pub_;
  ros::Publisher tree_abort_pub_, tree_debug_pub_;
  ros::Subscriber lc_status_sub_;
  smpc_lane_change::LaneChangeStatus lc_status_;
  bool have_lc_status_{false};
  bool tree_planner_enabled_{false};
  bool tree_keep_gates_{true};
  bool tree_monitor_enabled_{true};
  // research/docs/plan_v2.md 반응형 트리(기본 꺼짐 = 지금 동작). 값은 Python reactive.ReactiveParams 와 같다.
  bool reactive_tree_enabled_{false};
  bool reactive_drop_common_{true};
  bool reactive_single_{false};
  struct RxParams {
    double plant_tau{0.7}, plant_a_max{3.0}, plant_a_min{-6.0}, plant_a_stop{-9.0}, plant_jerk{12.0};
    double acc_up{4.0}, acc_dn{7.0}, mission_ramp{0.0}, cruise_max{0.0}, lane_end_decel{2.5};
    double rear_react_intrusion_m{0.3}, rear_react_delay_s{1.0}, rear_detect_s{0.5};
    double rear_idm_s0{3.0}, rear_idm_T{1.0}, rear_idm_a{1.0}, rear_idm_b{2.0}, rear_max_brake{3.0};
    double front_brake{-2.5}, front_accel{1.0}, front_detect_s{0.3}, accel_decay_s{1.5};
    double front_idm_s0{3.0}, front_idm_T{1.0}, front_idm_a{1.0}, front_idm_b{2.0};
    double p_exc_floor{0.05}, follow_sigma_cap_s{1.0}, abort_before_abs_d{1.75};
    bool connector_cap{false};   // ACC 의 정지 connector 속도 상한(MORAI 대회 스택, 파이썬 시뮬 world 도 같다)
    bool monitor_tree{true};     // 실행 중 모니터의 계속 위험도 승인처럼 가지별 나중 대응을 넣어 푼다(false = 옛 규칙)
    // 자차가 지나친 목표 차선 앞차(tf·tf2)를 뒤차로 본다(11-06): 뒤차 가지(yield/nonyield)를 함께 쓰고, yield 면
    // 반응 지연 뒤 자차를 IDM 으로 따르며, 지나친 뒤엔 앞차 예외(급감속·가속)가 없고 σ 도 뒤차처럼(상한 없음).
    bool passed_front_as_rear{false};
    // 실제 ACC 의 merge_keep_rolling 하한(acc_speed_planner applyMergeRollingFloor)과 차선 끝 대기 조건(mergeWaitStopRequired)을
    // 예측 ACC 에 넣는다(11-06). 차선 끝 정지 제한은 대기 상태일 때만 건다.
    bool merge_floor{false};
    double floor_urgent_distance_m{120.0}, floor_urgent_time_s{5.0};   // target_selector lane_end_urgent_*
  } rx_;
  // acc_planner.yaml 의 같은 키(이 노드에는 ns 'acc' 로 올린다)
  struct RxAcc {
    double standstill_gap_m{5.0}, time_headway_sec{1.2}, closing_speed_gap_time_sec{1.6}, gap_gain{1.05};
    double max_follow_speed_drop_mps{8.0}, emergency_gap_m{5.0}, critical_gap_m{3.0}, critical_ttc_sec{2.2};
    double soft_ttc_sec{7.0}, hard_ttc_sec{2.8}, target_front_acc_entry_abs_d_m{0.75};
    double target_front_overlap_soft_acc_start_abs_d_m{2.7}, lane_end_merge_wait_stop_distance_m{25.0};
    double lane_change_lane_end_margin_m{3.0}, lane_change_lane_end_min_cap_mps{3.0};
    double lane_change_lane_end_cap_release_abs_d_m{1.5}, closing_decel_boost_start_mps{1.0};
    double closing_decel_boost_full_mps{6.0}, lead_required_decel_limit_gain{1.5};
    double lead_required_decel_limit_headway_sec{1.2}, rapid_lead_ttc_sec{3.0}, rapid_lead_min_closing_speed_mps{4.0};
    bool lead_required_decel_limit_enabled{true}, hard_brake_bypass_filter{true}, rapid_lead_brake_enabled{true};
    bool merge_keep_rolling_enabled{true};
    double merge_keep_rolling_min_speed_mps{11.1}, merge_keep_rolling_front_block_gap_m{8.0};
    double merge_keep_rolling_low_speed_lead_mps{11.1}, merge_keep_rolling_low_speed_lead_max_gap_m{80.0};
    double min_target_front_gap_m{11.0}, lane_end_stop_prepare_distance_m{45.0}, lane_end_stop_prepare_time_sec{4.0};
    std::vector<int> lane_end_stop_lane_ids{3};
  } rx_acc_;
  double rx_intr_since_{-1.0};
  double rx_p_rear_post_{-1.0};
  bool rx_lc_connector_{false};   // 마지막 지금 바꾸기 승인이 connector 출발이었나(실행 중 모니터가 같은 방식으로 본다)
  bool baseline_keep_gates_{true};
  bool baseline_tree_monitor_{false};
  std::vector<double> tree_onsets_sec_{0.0, 1.0, 2.0, 3.0};
  double tree_rear_detect_delay_sec_{0.5};
  double tree_front_detect_delay_sec_{0.3};
  double tree_escape_accel_mps2_{1.5};
  double tree_return_time_full_sec_{3.1};
  double tree_return_decel_mps2_{-1.5};
  double tree_brake_decel_mps2_{-3.0};
  double tree_abort_before_abs_d_m_{1.75};
  std::vector<double> tree_response_cost_{0.0, 1.0, 30.0, 35.0, 5.0};
  double tree_risk_weight_{80.0};
  double tree_abort_risk_factor_{2.0};
  bool tree_executable_responses_only_{true};
  bool tree_plant_return_{true};
  // 한 판단 주기 안에서만 유효한 TV 예측 캐시(treeSelect / treeMonitorReturn 시작에서 비운다).
  mutable std::unordered_map<std::uint64_t, std::pair<int, PredPose>> tree_pred_cache_;
  ros::Timer timer_;

  smpc_lane_change::TargetVehicleSet targets_;
  ros::Time targets_stamp_;
  bool have_targets_{false};
  bool have_ego_{false};
  bool have_mission_speed_target_{false};
  bool have_acc_target_speed_{false};
  bool have_behavior_status_{false};
  bool have_reference_preview_{false};
  double ego_x_{0.0};
  double ego_y_{0.0};
  double ego_yaw_{0.0};
  double ego_speed_mps_{0.0};
  double ego_longitudinal_accel_mps2_{0.0};
  ros::Time ego_speed_stamp_;
  ros::Time acc_commit_stable_since_;

  std::map<int, smpc_lane_change::LanePath> lanes_;
  std::unordered_map<int, VehicleHistory> vehicle_history_;
  // 목표 차선 판단 구간에서 "사라진" 차량.  트래커 출력이 끊겨도 차가 실제로
  // 사라진 것은 아니다 (lc_gt 백 15개: 실차 트랙 끊김 383건 중 289건이 0.3 s
  // 이하, 최대 0.77 s, 1 s 초과 0건).  마지막 관측을 등속으로 밀어 잠깐 유지한다.
  struct HeldVehicle {
    smpc_lane_change::TargetVehicle vehicle;
    ros::Time last_seen_stamp;
    int current_lane_id{-1};
    int target_lane_id{-1};
  };
  std::unordered_map<int, HeldVehicle> held_target_vehicles_;
  int held_injected_count_{0};
  // projectedTargetGapTtcSafe 의 실패 지점 계측 (2026-09-17).  게이트는 불린
  // 하나만 내보내서 앞/뒤 어느 항이 걸렸는지 분리할 수 없었다.  판단에는
  // 영향이 없는 출력 전용 기록이다.
  struct ProjGateFail {
    bool valid{false};
    bool front{false};
    bool ttc{false};
    double t_sec{0.0};
    double gap{0.0};
    double required{0.0};
    int unique_id{-1};
    // 실패를 유발한 TV 모드 (nominal / front_brake / rear_accel ...).
    // 일반 앞차에는 front_brake 가 결정론적으로 적용되므로, 어느 모드가
    // 거부권을 행사했는지 구분해야 원인을 가릴 수 있다.
    std::string mode;
    // 실패 스텝에서의 예측 자차 속도와 TV 전진속도.  "자차가 빨라서 붙었다" 와
    // "앞차가 제동해서 붙었다" 를 산술로 가르려면 둘 다 필요하다.
    double ego_v{0.0};
    double tv_v{0.0};
    // 어느 쪽 yaw 가 퇴화했는지 가리기 위해 둘 다 기록한다.
    // tv_forward_speed = tv.v * cos(tv.yaw - ego.yaw) 이므로 두 값에 모두 의존.
    double ego_yaw{0.0};
    double tv_yaw{0.0};
  };
  mutable ProjGateFail proj_fail_;
  bool vanished_target_hold_enabled_{false};
  double vanished_target_hold_sec_{0.50};
  int vanished_target_hold_min_hits_{20};
  double vanished_target_hold_zone_rear_m_{60.0};
  double vanished_target_hold_zone_front_m_{30.0};
  double vanished_target_hold_margin_m_{1.5};
  double vanished_target_hold_alongside_sec_{0.0};
  double vanished_target_hold_alongside_ds_m_{4.0};
  TargetLaneSceneHistory target_lane_scene_history_;
  std::string waypoint_directory_;
  bool lanes_loaded_{false};

  bool enabled_{true};
  double publish_rate_hz_{20.0};
  double target_timeout_sec_{0.5};
  int horizon_steps_{40};
  double prediction_dt_sec_{0.2};
  double risk_epsilon_{0.04};
  double urgent_risk_epsilon_{0.15};
  bool risk_include_nearby_vehicles_{true};

  double ego_length_m_{4.635};
  double ego_width_m_{1.892};
  double ego_front_extent_m_{3.845};
  double ego_rear_extent_m_{0.790};
  std::string mission_speed_topic_{"/control_feedback/mission_speed_target_mps"};
  double mission_speed_timeout_sec_{0.5};
  double mission_speed_fallback_mps_{10.0};
  double mission_speed_target_mps_{10.0};
  ros::Time mission_speed_stamp_;
  bool ego_prediction_use_acc_target_speed_{true};
  std::string ego_prediction_acc_target_speed_topic_{"/smpc/target_speed_mps"};
  double ego_prediction_acc_target_speed_timeout_sec_{0.5};
  double acc_target_speed_mps_{0.0};
  ros::Time acc_target_speed_stamp_;
  smpc_lane_change::BehaviorLongitudinalStatus behavior_status_;
  ros::Time behavior_status_stamp_;
  ReferencePreview reference_preview_;
  double standstill_gap_m_{5.0};
  double time_headway_sec_{1.0};
  double current_front_trigger_gap_m_{35.0};
  double current_front_safe_gap_m_{7.0};
  double target_front_safe_gap_m_{14.0};
  double target_rear_safe_gap_m_{9.0};
  double prepare_merge_front_safe_gap_m_{12.0};
  double prepare_merge_rear_safe_gap_m_{7.0};
  double target_front_min_ttc_sec_{3.0};
  double target_front_min_headway_sec_{1.2};
  bool target_front_headway_use_closing_speed_{false};
  bool pass_gap_headway_use_closing_speed_{false};
  bool change_rollout_post_completion_use_pre_cap_floor_{true};
  bool change_rollout_blend_speed_cap_enabled_{false};
  bool change_rollout_blend_speed_interp_enabled_{false};
  bool change_rollout_blend_stage_enabled_{false};
  double change_rollout_soft_acc_start_abs_d_m_{2.7};
  double change_rollout_launch_speed_threshold_mps_{1.0};
  double change_rollout_blend_cap_decel_mps2_{4.0};
  double target_rear_min_ttc_sec_{3.0};
  double target_rear_min_headway_sec_{1.2};
  double change_commit_scene_observation_sec_{1.5};
  bool change_commit_allow_lane_end_wait_decel_{false};
  double change_commit_vehicle_observation_sec_{0.6};
  double change_commit_observation_gap_timeout_sec_{0.35};
  double change_commit_acc_stable_sec_{0.50};
  double change_commit_acc_speed_error_mps_{1.50};
  double change_commit_acc_max_decel_mps2_{1.00};
  bool pass_gap_enabled_{true};
  double pass_gap_min_relative_speed_mps_{2.0};
  double pass_gap_front_max_distance_m_{40.0};
  double pass_gap_entry_front_clearance_m_{2.0};
  double pass_gap_entry_relative_speed_time_sec_{0.0};
  double pass_gap_entry_length_margin_m_{0.0};
  double pass_gap_lane_pair_observation_sec_{1.5};
  double rear_gap_perception_control_delay_sec_{0.50};
  double new_rear_track_guard_sec_{0.80};
  double new_rear_track_max_gap_m_{40.0};
  double new_rear_track_closing_upper_mps_{15.0};
  double tv_lane_projection_extension_before_m_{100.0};
  bool observed_cutin_safety_enabled_{true};
  bool observed_cutin_assume_no_lane_change_{false};
  double observed_cutin_min_lateral_speed_mps_{0.5};
  double observed_cutin_max_entry_time_sec_{4.5};
  double observed_cutin_max_rear_distance_m_{60.0};
  double urgent_merge_front_safe_gap_m_{9.0};
  double urgent_merge_rear_safe_gap_m_{5.5};
  double emergency_merge_front_safe_gap_m_{6.5};
  double emergency_merge_rear_safe_gap_m_{4.5};
  double emergency_merge_risk_epsilon_{0.25};
  bool merge_priority_enabled_{true};
  double merge_priority_rear_safe_gap_m_{4.5};
  double merge_priority_rear_min_gap_m_{4.0};
  double merge_priority_rear_max_closing_mps_{3.5};
  double merge_priority_rear_min_ttc_sec_{1.8};
  double merge_priority_rear_risk_scale_{0.10};
  double merge_priority_nearby_rear_risk_scale_{0.05};
  bool low_speed_rear_safety_enabled_{true};
  double low_speed_rear_ego_speed_mps_{2.0};
  double low_speed_rear_safe_gap_m_{10.0};
  double low_speed_rear_min_ttc_sec_{3.0};
  bool ego_lane_priority_enabled_{true};
  double ego_lane_priority_current_max_abs_d_m_{1.25};
  double ego_lane_priority_front_hard_gap_m_{8.0};
  double ego_lane_priority_side_risk_scale_{0.08};

  double lane_width_m_{3.5};
  double lane_change_duration_sec_{2.5};
  bool use_waypoint_linear_blend_{true};
  int waypoint_blend_steps_{17};
  double waypoint_blend_rate_hz_{25.0};
  double waypoint_blend_duration_sec_{0.0};
  double lane_center_decay_tau_sec_{1.2};
  bool ego_reference_preview_enabled_{true};
  bool ego_reference_preview_required_for_change_{true};
  std::string ego_reference_preview_topic_{"/path_switch_preview"};
  double ego_reference_preview_timeout_sec_{0.50};
  double ego_reference_preview_duration_scale_{1.0};
  double ego_reference_preview_motion_start_speed_mps_{0.20};
  double ego_reference_preview_motion_start_distance_m_{0.30};
  double stopped_launch_connector_speed_cap_mps_{2.0};
  std::vector<double> stopped_launch_connector_speed_cap_by_target_lane_mps_;
  double stopped_launch_connector_max_accel_mps2_{1.5};
  double stopped_launch_connector_max_preswitch_distance_m_{0.50};
  bool ego_reference_preview_require_target_completion_{true};
  bool ego_reference_preview_allow_truncation_{true};
  double change_commit_scene_rear_max_range_m_{0.0};
  bool pass_gap_skip_ramp_while_receding_{false};
  bool ego_brake_reaction_enabled_{false};
  double ego_brake_reaction_delay_sec_{0.5};
  double ego_brake_reaction_max_decel_mps2_{2.5};
  mutable std::string ego_traj_build_detail_{"none"};
  double ego_reference_preview_target_completion_max_abs_d_m_{0.75};

  bool behavior_planner_enabled_{true};
  bool behavior_execute_decel_hold_{false};
  bool behavior_execute_accel_{false};
  bool behavior_require_legacy_risk_gate_{false};
  double behavior_min_signed_clearance_m_{0.0};
  double behavior_execution_delay_sec_{0.20};
  double behavior_post_merge_buffer_sec_{1.5};
  double behavior_decel_accel_mps2_{-1.0};
  double behavior_yield_comfort_max_decel_mps2_{0.8};
  double behavior_yield_max_speed_drop_mps_{3.0};
  double behavior_yield_confirm_sec_{0.6};
  double behavior_hold_accel_mps2_{0.0};
  double behavior_accel_accel_mps2_{1.0};
  double behavior_rear_accel_min_closing_mps_{1.0};
  double behavior_rear_accel_gain_{0.25};
  double behavior_rear_accel_min_mps2_{0.40};
  double behavior_rear_accel_emergency_ttc_sec_{2.0};
  std::vector<double> behavior_yield_decel_candidates_mps2_{-0.8};
  double behavior_delay_cost_weight_{1.0};
  double behavior_accel_cost_weight_{0.25};
  std::vector<double> behavior_lc_start_times_sec_{0.4, 0.8, 1.2, 1.6, 2.0};
  std::string behavior_status_topic_{"/smpc/behavior_longitudinal_status"};
  double behavior_status_timeout_sec_{0.35};
  uint32_t behavior_plan_id_{0};
  uint32_t last_behavior_plan_id_{0};
  uint32_t behavior_command_revision_{0};
  uint32_t last_behavior_command_plan_id_{0};
  uint32_t last_behavior_command_revision_{0};
  uint32_t previous_behavior_command_plan_id_{0};
  uint32_t previous_behavior_command_revision_{0};
  BehaviorAction last_behavior_action_{BehaviorAction::kKeep};
  double behavior_yield_episode_start_speed_mps_{
      std::numeric_limits<double>::quiet_NaN()};
  YieldCandidateConfirmation yield_candidate_confirmation_;
  int last_behavior_current_lane_id_{std::numeric_limits<int>::min()};
  int last_behavior_target_lane_id_{std::numeric_limits<int>::min()};
  BehaviorAction last_behavior_command_action_{BehaviorAction::kKeep};
  bool last_behavior_command_active_{false};
  bool previous_behavior_command_active_{false};
  int last_behavior_command_current_lane_id_{std::numeric_limits<int>::min()};
  int last_behavior_command_target_lane_id_{std::numeric_limits<int>::min()};
  double last_behavior_command_lane_change_start_sec_{
      std::numeric_limits<double>::quiet_NaN()};
  double last_behavior_command_speed_cap_mps_{
      std::numeric_limits<double>::quiet_NaN()};
  double previous_behavior_command_speed_cap_mps_{
      std::numeric_limits<double>::quiet_NaN()};

  double front_brake_probability_{0.25};
  double rear_accel_probability_{0.25};
  bool rear_accel_probability_by_accel_enabled_{false};
  std::vector<double> rear_accel_probability_accel_breaks_mps2_;
  std::vector<double> rear_accel_probability_by_accel_;
  double min_front_brake_probability_{0.05};
  double max_front_brake_probability_{0.65};
  double min_rear_accel_probability_{0.05};
  double max_rear_accel_probability_{0.65};
  double front_brake_accel_mps2_{-2.0};
  double rear_accel_mps2_{1.5};
  bool rear_accel_max_speed_enabled_{false};
  double rear_accel_max_speed_mps_{-1.0};
  double rear_accel_max_speed_margin_mps_{0.0};
  double nominal_target_accel_mps2_{0.0};
  double ego_assumed_accel_mps2_{0.0};
  bool ego_prediction_use_mission_speed_rollout_{true};
  double ego_prediction_max_accel_mps2_{2.5};
  bool change_rollout_target_lane_speed_enabled_{false};
  double change_rollout_release_fraction_{1.0};
  double ego_prediction_max_decel_mps2_{2.0};
  double probability_ttc_safe_sec_{5.0};
  double probability_ttc_critical_sec_{1.5};
  double probability_rel_speed_scale_mps_{8.0};
  double probability_accel_scale_mps2_{3.0};
  double observed_accel_alpha_{0.35};
  double max_observed_accel_mps2_{6.0};

  bool lateral_mode_enabled_{true};
  double lateral_mode_probability_{0.03};
  double max_lateral_mode_probability_{0.08};
  double lateral_mode_v_threshold_mps_{0.90};
  double yaw_lateral_threshold_rad_{0.28};
  double yaw_lateral_probability_gain_{0.04};
  double max_lateral_prediction_mps_{0.45};
  bool lateral_mode_allow_nearby_{false};
  bool lateral_mode_allow_rear_{false};
  bool lateral_mode_allow_yaw_only_{false};
  double lateral_mode_min_abs_d_m_{1.40};
  bool lateral_mode_require_yaw_agreement_{true};
  bool lateral_mode_require_away_from_center_{true};
  double tv_yaw_rate_weight_{0.05};
  double tv_csv_yaw_weight_{0.90};
  double tv_no_lateral_probability_floor_{0.90};
  double tv_yaw_rate_horizon_sec_{0.8};
  double max_abs_tv_yaw_rate_{0.6};

  double prediction_sigma_base_m_{0.5};
  double prediction_sigma_growth_mps_{0.35};
  double prediction_nominal_sigma_base_m_{0.5};
  double prediction_nominal_sigma_growth_mps_{0.35};
  double prediction_front_brake_sigma_base_m_{0.5};
  double prediction_front_brake_sigma_growth_mps_{0.35};
  double prediction_rear_accel_sigma_base_m_{0.5};
  double prediction_rear_accel_sigma_growth_mps_{0.35};
  double prediction_lateral_sigma_base_m_{0.25};
  double prediction_lateral_sigma_growth_mps_{0.12};
  double collision_margin_m_{0.45};

  double lane_change_base_cost_{0.5};
  double risk_cost_weight_{80.0};
  double gap_cost_weight_{2.5};
  double blocked_cost_weight_{35.0};
  double lane_end_prepare_cost_{60.0};
  double lane_end_urgent_cost_{180.0};
  double lane_end_emergency_cost_{500.0};
  double decision_hysteresis_cost_{0.1};
  bool debug_publish_marker_{true};
  double debug_marker_lifetime_sec_{0.5};
  double debug_marker_height_m_{4.0};
  bool debug_prediction_publish_markers_{true};
  double debug_prediction_marker_lifetime_sec_{0.5};
  double debug_prediction_line_width_m_{0.25};
  double debug_prediction_z_m_{0.35};
  double debug_prediction_footprint_height_m_{0.15};
  int debug_prediction_ego_footprint_stride_{5};
  int debug_prediction_target_footprint_stride_{5};
  int debug_prediction_target_max_vehicles_{20};
  bool debug_prediction_show_target_modes_{true};
  bool debug_prediction_show_labels_{true};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_decision");
  SmpcDecisionNode node;
  ros::spin();
  return 0;
}
