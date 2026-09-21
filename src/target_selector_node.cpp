#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <map>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <geometry_msgs/Point.h>
#include <geometry_msgs/Quaternion.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <visualization_msgs/MarkerArray.h>
#include <waypoint_maker/State.h>
#include <morai_msgs/ObjectStatusList.h>
#include <lidar_ttc_tracker/TrackedObjectArray.h>

#include <smpc_lane_change/TargetVehicle.h>
#include <smpc_lane_change/TargetVehicleSet.h>
#include <smpc_lane_change/LaneChangeStatus.h>
#include <smpc_lane_change/lane_path.hpp>

namespace fs = std::filesystem;
using smpc_lane_change::LanePath;
using smpc_lane_change::Projection;
using smpc_lane_change::TargetVehicle;
using smpc_lane_change::TargetVehicleSet;

class TargetSelectorNode {
 public:
  TargetSelectorNode() : nh_(), pnh_("~") {
    pnh_.param<std::string>("waypoint_directory", waypoint_directory_, "");
    pnh_.param<std::string>("target_source", target_source_, "tracked_objects");
    pnh_.param<std::string>("object_topic", object_topic_, "/Object_topic");
    pnh_.param<std::string>("tracked_objects_topic", tracked_objects_topic_, "/tracked_objects");
    pnh_.param<std::string>("odom_topic", odom_topic_, "/odom");
    pnh_.param<std::string>("state_topic", state_topic_, "/gps_state");
    pnh_.param("lane_half_width", lane_half_width_, 2.2);
    pnh_.param("max_front_range", max_front_range_, 120.0);
    pnh_.param("max_rear_range", max_rear_range_, 80.0);
    pnh_.param("target_lane_projection_extension_before_m",
               target_lane_projection_extension_before_m_, 100.0);
    pnh_.param("min_tracked_hits", min_tracked_hits_, 3);
    pnh_.param("tracked_dynamic_only", tracked_dynamic_only_, false);
    pnh_.param("tracked_vehicle_filter_enabled", tracked_vehicle_filter_enabled_, true);
    pnh_.param("tracked_vehicle_min_confidence", tracked_vehicle_min_confidence_, 0.60);
    pnh_.param("tracked_vehicle_confidence_hold_enabled",
               tracked_vehicle_confidence_hold_enabled_, false);
    pnh_.param("tracked_vehicle_confidence_hold_min_hits",
               tracked_vehicle_confidence_hold_min_hits_, 20);
    pnh_.param("tracked_vehicle_confidence_hold_stale_sec",
               tracked_vehicle_confidence_hold_stale_sec_, 0.50);
    pnh_.param("tracked_vehicle_confidence_hold_timeout_sec",
               tracked_vehicle_confidence_hold_timeout_sec_, 0.35);
    pnh_.param("tracked_unknown_obstacle_min_confidence",
               tracked_unknown_obstacle_min_confidence_, 0.42);
    pnh_.param("min_object_length_m", min_object_length_m_, 1.0);
    pnh_.param("max_object_length_m", max_object_length_m_, 8.0);
    pnh_.param("min_object_width_m", min_object_width_m_, 0.4);
    pnh_.param("max_object_width_m", max_object_width_m_, 3.5);
    pnh_.param("align_tracked_yaw_with_velocity", align_tracked_yaw_with_velocity_, true);
    pnh_.param("min_yaw_align_speed_mps", min_yaw_align_speed_mps_, 1.0);
    pnh_.param("bbox_overlap_margin_m", bbox_overlap_margin_m_, 0.2);
    pnh_.param("representative_lane_center_filter_enabled",
               representative_lane_center_filter_enabled_,
               true);
    // 길고 납작한 트랙은 대표 차량(current_front/target_front/target_rear)으로
    // 승격시키지 않는다.  VLP-16 링이 여러 물체를 가로질러 한 줄로 그은 선분이
    // 실제보다 긴 박스로 나오는 경우이고, 위치와 속도가 함께 틀린다.
    // lc_gt 14-18 실측(대표 599건): 이 조건에 걸린 27건 중 current_front 13건은
    // 모두 속도 부호/크기가 GT 와 어긋났고 그중 10건이 ACC 하드 브레이크를
    // 유발했다(목표속도 0).  target_front 14건은 실차 손실 0 으로 GT 미대응을
    // 9.1% -> 4.6% 로 줄였고 target_rear 는 해당 없음(0건).
    // 실제 대형차는 근거리에서 높이가 측정되므로 걸리지 않는다(길이>=8 m 이면서
    // 높이>0.25 m 인 트랙 52건의 GT 미대응은 13.5%).
    // 0 이면 비활성.  nearby_vehicles 에는 그대로 남는다.
    pnh_.param("representative_flat_max_length_m",
               representative_flat_max_length_m_, 0.0);
    pnh_.param("representative_flat_height_m",
               representative_flat_height_m_, 0.25);
    pnh_.param("representative_lane_center_max_abs_d_m",
               representative_lane_center_max_abs_d_m_,
               1.35);
    pnh_.param("sticky_role_selection_enabled", sticky_role_selection_enabled_, true);
    pnh_.param("sticky_role_size_hysteresis_m", sticky_role_size_hysteresis_m_, 0.15);
    sticky_role_size_hysteresis_m_ = std::max(0.0, sticky_role_size_hysteresis_m_);
    pnh_.param("target_lane_offset", target_lane_offset_, 1);
    pnh_.param("yaw_rate_alpha", yaw_rate_alpha_, 0.15);
    pnh_.param("max_abs_yaw_rate", max_abs_yaw_rate_, 0.6);
    pnh_.param("stabilize_tracked_yaw", stabilize_tracked_yaw_, true);
    pnh_.param("tracked_yaw_filter_alpha", tracked_yaw_filter_alpha_, 0.18);
    pnh_.param("tracked_yaw_history_timeout_sec", tracked_yaw_history_timeout_sec_, 1.0);
    pnh_.param("tracked_yaw_max_step_rad", tracked_yaw_max_step_rad_, 0.07);
    pnh_.param("tracked_yaw_velocity_blend_weight", tracked_yaw_velocity_blend_weight_, 0.25);
    pnh_.param("tracked_yaw_lane_blend_weight", tracked_yaw_lane_blend_weight_, 0.90);
    pnh_.param("tracked_yaw_lane_max_error_rad", tracked_yaw_lane_max_error_rad_, 1.57);
    pnh_.param("tracked_yaw_lateral_reduction_gain", tracked_yaw_lateral_reduction_gain_, 0.0);
    pnh_.param("tracked_yaw_lateral_speed_reduce_mps",
               tracked_yaw_lateral_speed_reduce_mps_, 0.8);
    pnh_.param("tracked_yaw_lane_prior_hits", tracked_yaw_lane_prior_hits_, 12);
    pnh_.param("tracked_yaw_lane_change_boundary_abs_d_m",
               tracked_yaw_lane_change_boundary_abs_d_m_, 1.0);
    pnh_.param("tracked_yaw_lane_change_lateral_speed_mps",
               tracked_yaw_lane_change_lateral_speed_mps_, 0.35);
    pnh_.param("tracked_yaw_lane_change_weight_scale",
               tracked_yaw_lane_change_weight_scale_, 0.15);
    pnh_.param("tracked_yaw_lane_change_confirm_hits",
               tracked_yaw_lane_change_confirm_hits_, 3);
    pnh_.param("tracked_yaw_lane_change_release_hits",
               tracked_yaw_lane_change_release_hits_, 5);
    pnh_.param("tracked_yaw_lane_change_min_track_hits",
               tracked_yaw_lane_change_min_track_hits_,
               tracked_yaw_lane_prior_hits_);
    pnh_.param("tracked_yaw_confidence_enabled", tracked_yaw_confidence_enabled_, true);
    pnh_.param("tracked_yaw_confidence_low", tracked_yaw_confidence_low_, 0.35);
    pnh_.param("tracked_yaw_confidence_high", tracked_yaw_confidence_high_, 0.70);
    pnh_.param("tracked_yaw_low_confidence_velocity_blend_weight",
               tracked_yaw_low_confidence_velocity_blend_weight_, 0.70);
    pnh_.param("tracked_yaw_low_confidence_lane_blend_weight",
               tracked_yaw_low_confidence_lane_blend_weight_, 0.40);
    pnh_.param("tracked_yaw_velocity_min_hits", tracked_yaw_velocity_min_hits_, 3);
    pnh_.param("final_lane_id", final_lane_id_, 3);
    pnh_.param("lane_end_prepare_time_sec", lane_end_prepare_time_sec_, 8.0);
    pnh_.param("lane_end_urgent_time_sec", lane_end_urgent_time_sec_, 5.0);
    pnh_.param("lane_end_emergency_time_sec", lane_end_emergency_time_sec_, 1.0);
    pnh_.param("lane_end_prepare_distance_m", lane_end_prepare_distance_m_, 200.0);
    pnh_.param("lane_end_urgent_distance_m", lane_end_urgent_distance_m_, 120.0);
    pnh_.param("lane_end_emergency_distance_m", lane_end_emergency_distance_m_, 15.0);
    pnh_.param("nearby_front_range", nearby_front_range_, 150.0);
    pnh_.param("nearby_rear_range", nearby_rear_range_, 60.0);
    pnh_.param("nearby_lane_margin_m", nearby_lane_margin_m_, 1.0);
    pnh_.param("max_nearby_vehicles", max_nearby_vehicles_, 40);
    pnh_.param("lane_yaw_align_when_slow", lane_yaw_align_when_slow_, true);
    pnh_.param("lane_yaw_align_max_speed_mps", lane_yaw_align_max_speed_mps_, 1.5);
    pnh_.param("lane_yaw_align_weight", lane_yaw_align_weight_, 0.85);
    pnh_.param("debug_publish_csv_paths", debug_publish_csv_paths_, true);
    pnh_.param("debug_show_csv_file_path", debug_show_csv_file_path_, true);
    pnh_.param("debug_show_ego_vehicle", debug_show_ego_vehicle_, true);
    pnh_.param("debug_show_nearby_vehicles", debug_show_nearby_vehicles_, true);
    pnh_.param("debug_show_nearby_labels", debug_show_nearby_labels_, false);
    pnh_.param("debug_use_vehicle_mesh", debug_use_vehicle_mesh_, true);
    pnh_.param<std::string>("debug_vehicle_mesh_resource", debug_vehicle_mesh_resource_,
                            "package://smpc_lane_change/meshes/debug_vehicle.stl");
    pnh_.param("debug_marker_lifetime_sec", debug_marker_lifetime_sec_, 0.5);
    pnh_.param("debug_vehicle_height_m", debug_vehicle_height_m_, 1.5);
    pnh_.param("debug_label_height_m", debug_label_height_m_, 2.5);
    pnh_.param("debug_show_id_labels", debug_show_id_labels_, true);
    pnh_.param("debug_id_label_scale", debug_id_label_scale_, 2.2);
    pnh_.param("debug_id_label_height_m", debug_id_label_height_m_, 4.2);
    pnh_.param("debug_csv_path_z_m", debug_csv_path_z_m_, 0.05);
    pnh_.param("debug_max_csv_points_per_lane", debug_max_csv_points_per_lane_, 2500);

    if (!loadLanes()) {
      ROS_FATAL("[target_selector] Failed to load lane CSVs from %s", waypoint_directory_.c_str());
      ros::shutdown();
      return;
    }

    if (target_source_ == "tracked_objects" || target_source_ == "lidar") {
      tracked_objects_sub_ = nh_.subscribe(
          tracked_objects_topic_, 1, &TargetSelectorNode::trackedObjectsCallback, this);
      ROS_INFO("[target_selector] using LiDAR tracked objects from %s", tracked_objects_topic_.c_str());
    } else if (target_source_ == "object_topic" || target_source_ == "morai") {
      object_sub_ = nh_.subscribe(object_topic_, 1, &TargetSelectorNode::objectCallback, this);
      ROS_WARN("[target_selector] using MORAI Object_topic fallback from %s", object_topic_.c_str());
    } else {
      ROS_WARN("[target_selector] unknown target_source='%s'; falling back to LiDAR tracked_objects",
               target_source_.c_str());
      target_source_ = "tracked_objects";
      tracked_objects_sub_ = nh_.subscribe(
          tracked_objects_topic_, 1, &TargetSelectorNode::trackedObjectsCallback, this);
    }
    odom_sub_ = nh_.subscribe(odom_topic_, 10, &TargetSelectorNode::odomCallback, this);
    state_sub_ = nh_.subscribe(state_topic_, 10, &TargetSelectorNode::stateCallback, this);
    status_sub_ = nh_.subscribe("/smpc/lane_change_status", 10, &TargetSelectorNode::statusCallback, this);
    target_pub_ = nh_.advertise<TargetVehicleSet>("/smpc/targets", 1);
    marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/smpc/target_markers", 1);
    csv_path_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("/smpc/csv_path_markers", 1, true);

    if (debug_publish_csv_paths_) {
      publishCsvPathMarkers();
    }

    ROS_INFO("[target_selector] loaded %zu lane CSVs", lanes_.size());
  }

 private:
  struct YawHistory {
    bool initialized{false};
    double yaw{0.0};
    double filtered_rate{0.0};
    ros::Time stamp;
  };

  struct LaneChangeCandidateHistory {
    bool initialized{false};
    bool active{false};
    int evidence_hits{0};
    int stable_hits{0};
    ros::Time stamp;
  };

  // Vehicle id currently occupying one of the three featured roles, so the
  // object filter can tell a role holder from a new candidate.
  struct RoleSelection {
    int held_id{-1};
  };

  struct CandidateObject {
    int id{-1};
    double x{0.0};
    double y{0.0};
    double yaw{0.0};
    double yaw_confidence{1.0};
    double yaw_rate{0.0};
    double vx{0.0};
    double vy{0.0};
    double v_long{0.0};
    double v_lat{0.0};
    double length{0.0};
    double width{0.0};
    // LiDAR 트래커에서만 채워진다.  /Object_topic (MORAI GT) 경로는 NaN 으로
    // 남아 flat-track 판정이 적용되지 않는다.
    double height{std::numeric_limits<double>::quiet_NaN()};
    bool velocity_in_map{true};
    double vehicle_confidence{1.0};
    bool is_vehicle{true};
    int track_hits{std::numeric_limits<int>::max()};
  };

  struct LaneFootprint {
    double half_s{0.0};
    double half_d{0.0};
  };

  // 한 번 차량으로 확정된 트랙은 확신도가 잠깐 떨어져도 목록에 남긴다.
  // lc_gt 2026-09-15-19-33-03 37.4~39.6 s: 자차 옆에 붙은 id31 의 확신도가
  // 0.10~0.33 으로 내려가 2.2 s 동안 /smpc/targets 에서 빠졌고 (그동안에도
  // hits 96->107 로 계속 새로 검출됨), SMPC 는 빈 차선으로 보고 승인했다.
  // 백 15개: 이렇게 빠진 옆차선 성숙 트랙 269 프레임 중 239 (89%) 가 GT 실차.
  struct ConfidenceHold {
    bool confirmed{false};
    int last_hits{-1};
    ros::Time last_stamp;
    ros::Time last_growth_stamp;
  };

  // hits 증가(새 검출)가 이어지는 동안만 유지한다.  트래커가 관성으로만 끌고
  // 가는 트랙은 유지하지 않는다.  오래 안 보이면 번호 재사용일 수 있어 초기화.
  bool updateConfidenceHold(const CandidateObject& obj, const ros::Time& now) {
    if (!tracked_vehicle_confidence_hold_enabled_ || obj.id < 0) return false;
    auto& h = confidence_hold_[obj.id];
    const double timeout =
        std::max(0.0, tracked_vehicle_confidence_hold_timeout_sec_);
    if (!h.last_stamp.isZero() && (now - h.last_stamp).toSec() > timeout) {
      h = ConfidenceHold{};
    }
    if (h.last_hits < 0 || obj.track_hits > h.last_hits) h.last_growth_stamp = now;
    if (obj.is_vehicle &&
        obj.vehicle_confidence >= tracked_vehicle_min_confidence_ &&
        obj.track_hits >= tracked_vehicle_confidence_hold_min_hits_) {
      h.confirmed = true;
    }
    h.last_hits = obj.track_hits;
    h.last_stamp = now;
    if (!h.confirmed || h.last_growth_stamp.isZero()) return false;
    return (now - h.last_growth_stamp).toSec() <=
        std::max(0.0, tracked_vehicle_confidence_hold_stale_sec_);
  }

  void pruneConfidenceHold(const ros::Time& now) {
    const double keep = std::max(
        1.0, 4.0 * std::max(0.0, tracked_vehicle_confidence_hold_timeout_sec_));
    for (auto it = confidence_hold_.begin(); it != confidence_hold_.end();) {
      if (!it->second.last_stamp.isZero() &&
          (now - it->second.last_stamp).toSec() > keep) {
        it = confidence_hold_.erase(it);
      } else {
        ++it;
      }
    }
  }

  bool loadLanes() {
    if (waypoint_directory_.empty() || !fs::is_directory(waypoint_directory_)) return false;
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(waypoint_directory_)) {
      if (entry.is_regular_file() && entry.path().extension() == ".csv") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
      try { return std::stoi(a.stem().string()) < std::stoi(b.stem().string()); }
      catch (...) { return a.filename().string() < b.filename().string(); }
    });

    for (const auto& file : files) {
      int lane_id = -1;
      try { lane_id = std::stoi(file.stem().string()); } catch (...) { continue; }
      LanePath lane;
      if (lane.loadCsv(file.string())) {
        lane_csv_paths_[lane_id] = file.string();
        lanes_[lane_id] = std::move(lane);
      }
    }
    return !lanes_.empty();
  }

  void odomCallback(const nav_msgs::Odometry::ConstPtr& msg) {
    ego_x_ = msg->pose.pose.position.x;
    ego_y_ = msg->pose.pose.position.y;
    ego_yaw_ = yawFromQuaternion(msg->pose.pose.orientation);
    ego_speed_mps_ = std::hypot(msg->twist.twist.linear.x, msg->twist.twist.linear.y);
    ego_ready_ = true;
  }

  void stateCallback(const waypoint_maker::State::ConstPtr& msg) {
    gps_lane_id_ = msg->lane_number;
    if (!lane_change_active_) current_lane_id_ = gps_lane_id_;
    lane_ready_ = true;
  }

  void statusCallback(const smpc_lane_change::LaneChangeStatus::ConstPtr& msg) {
    lane_change_active_ = msg->active;
    if (lane_change_active_) {
      current_lane_id_ = msg->source_lane_id;
      forced_target_lane_id_ = msg->target_lane_id;
    } else {
      current_lane_id_ = gps_lane_id_;
      forced_target_lane_id_ = -1;
    }
  }

  double updateYawRate(int id, double yaw, const ros::Time& stamp) {
    auto& h = yaw_history_[id];
    if (!h.initialized) {
      h.initialized = true;
      h.yaw = yaw;
      h.stamp = stamp;
      return 0.0;
    }
    const double dt = (stamp - h.stamp).toSec();
    double raw = 0.0;
    if (dt > 1e-3 && dt < 1.0) raw = smpc_lane_change::wrapToPi(yaw - h.yaw) / dt;
    raw = std::clamp(raw, -max_abs_yaw_rate_, max_abs_yaw_rate_);
    h.filtered_rate = yaw_rate_alpha_ * raw + (1.0 - yaw_rate_alpha_) * h.filtered_rate;
    h.yaw = yaw;
    h.stamp = stamp;
    return h.filtered_rate;
  }

  void normalizeBoxAxes(CandidateObject& obj) const {
    if (obj.width > obj.length) {
      std::swap(obj.length, obj.width);
      obj.yaw = smpc_lane_change::wrapToPi(obj.yaw + 0.5 * M_PI);
    } else {
      obj.yaw = smpc_lane_change::wrapToPi(obj.yaw);
    }
  }

  void alignYawWithVelocity(CandidateObject& obj) const {
    if (!align_tracked_yaw_with_velocity_ || !obj.velocity_in_map) return;
    const double speed = std::hypot(obj.vx, obj.vy);
    if (speed < min_yaw_align_speed_mps_) return;

    const double heading_dot = std::cos(obj.yaw) * obj.vx + std::sin(obj.yaw) * obj.vy;
    if (heading_dot < 0.0) {
      obj.yaw = smpc_lane_change::wrapToPi(obj.yaw + M_PI);
    }
  }

  static double angleLerp(double a, double b, double alpha) {
    return smpc_lane_change::wrapToPi(a + alpha * smpc_lane_change::wrapToPi(b - a));
  }

  static double closestYawEquivalent(double yaw, double reference) {
    const double y0 = smpc_lane_change::wrapToPi(yaw);
    const double y1 = smpc_lane_change::wrapToPi(yaw + M_PI);
    return std::abs(smpc_lane_change::wrapToPi(y0 - reference)) <=
                   std::abs(smpc_lane_change::wrapToPi(y1 - reference))
               ? y0
               : y1;
  }

  static double objectSpeed(const CandidateObject& obj) {
    if (obj.velocity_in_map) return std::hypot(obj.vx, obj.vy);
    return std::hypot(obj.v_long, obj.v_lat);
  }

  double trackedYawReliability(const CandidateObject& obj) const {
    if (!tracked_yaw_confidence_enabled_ || !obj.velocity_in_map) return 1.0;
    const double q = std::isfinite(obj.yaw_confidence)
        ? std::clamp(obj.yaw_confidence, 0.0, 1.0)
        : 0.0;
    const double low = std::clamp(tracked_yaw_confidence_low_, 0.0, 0.99);
    const double high = std::max(low + 1e-3,
                                 std::clamp(tracked_yaw_confidence_high_, 0.0, 1.0));
    return std::clamp((q - low) / (high - low), 0.0, 1.0);
  }

  bool nearestLaneProjection(const CandidateObject& obj, Projection& best) const {
    bool found = false;
    double best_abs_d = std::numeric_limits<double>::infinity();
    for (const auto& lane_pair : lanes_) {
      const Projection p = lane_pair.second.project(obj.x, obj.y);
      if (!p.valid) continue;
      const double abs_d = std::abs(p.d);
      if (!found || abs_d < best_abs_d) {
        found = true;
        best_abs_d = abs_d;
        best = p;
      }
    }
    return found;
  }

  double lateralSpeedInLane(const CandidateObject& obj,
                            const Projection& projection) const {
    if (obj.velocity_in_map) {
      const double c = std::cos(projection.yaw);
      const double s = std::sin(projection.yaw);
      return -obj.vx * s + obj.vy * c;
    }
    return obj.v_lat;
  }

  bool hasLaneChangeYawEvidence(const CandidateObject& obj,
                                const Projection& nearest_projection) const {
    // Near a lane boundary or with measurable lateral motion, lane heading
    // should only be a weak prior. This is evidence, not an immediate mode
    // switch; transient LiDAR center/velocity noise is debounced below.
    const double boundary_threshold =
        std::max(0.0, tracked_yaw_lane_change_boundary_abs_d_m_);
    const double lateral_speed_threshold =
        std::max(0.0, tracked_yaw_lane_change_lateral_speed_mps_);
    const bool near_boundary = boundary_threshold > 0.0 &&
        std::abs(nearest_projection.d) >= boundary_threshold;
    const bool laterally_moving = lateral_speed_threshold > 0.0 &&
        std::abs(lateralSpeedInLane(obj, nearest_projection)) >= lateral_speed_threshold;
    return near_boundary || laterally_moving;
  }

  bool laneChangeYawCandidateActive(int id) const {
    const auto it = lane_change_candidate_history_.find(id);
    return it != lane_change_candidate_history_.end() && it->second.active;
  }

  bool updateLaneChangeYawCandidate(const CandidateObject& obj,
                                    const Projection* nearest_projection,
                                    const ros::Time& stamp) {
    if (obj.id < 0) return false;
    auto& h = lane_change_candidate_history_[obj.id];
    // The first few Kalman velocity/centroid estimates are not a motion
    // measurement yet.  Do not let their lateral noise bypass the initial
    // directed lane-yaw prior by falsely declaring a lane-change candidate.
    const bool mature_track = obj.track_hits >=
        std::max(1, tracked_yaw_lane_change_min_track_hits_);
    const bool evidence = mature_track && nearest_projection &&
        hasLaneChangeYawEvidence(obj, *nearest_projection);
    const double dt = h.initialized ? (stamp - h.stamp).toSec() : 0.0;
    if (!h.initialized || dt <= 1e-3 || dt > tracked_yaw_history_timeout_sec_) {
      h.initialized = true;
      h.active = false;
      h.evidence_hits = 0;
      h.stable_hits = 0;
    }

    if (evidence) {
      h.evidence_hits += 1;
      h.stable_hits = 0;
      if (h.evidence_hits >= std::max(1, tracked_yaw_lane_change_confirm_hits_)) {
        h.active = true;
      }
    } else {
      h.evidence_hits = 0;
      if (h.active) {
        h.stable_hits += 1;
        if (h.stable_hits >= std::max(1, tracked_yaw_lane_change_release_hits_)) {
          h.active = false;
          h.stable_hits = 0;
        }
      } else {
        h.stable_hits = 0;
      }
    }
    h.stamp = stamp;
    return h.active;
  }

  void pruneStaleLaneChangeYawCandidates(const ros::Time& stamp) {
    const double max_age = std::max(2.0, 2.0 * tracked_yaw_history_timeout_sec_);
    for (auto it = lane_change_candidate_history_.begin();
         it != lane_change_candidate_history_.end();) {
      if ((stamp - it->second.stamp).toSec() > max_age) {
        it = lane_change_candidate_history_.erase(it);
      } else {
        ++it;
      }
    }
  }

  bool hasFreshYawHistory(int id, const ros::Time& stamp) const {
    const auto it = yaw_history_.find(id);
    if (it == yaw_history_.end() || !it->second.initialized) return false;
    const double dt = (stamp - it->second.stamp).toSec();
    return dt >= 0.0 && dt <= tracked_yaw_history_timeout_sec_;
  }

  double fusedTrackedYawMeasurement(const CandidateObject& obj,
                                    const Projection* lane_projection,
                                    bool use_lane_prior,
                                    bool lane_change_candidate,
                                    bool has_fresh_history) const {
    double yaw = smpc_lane_change::wrapToPi(obj.yaw);
    const double speed = objectSpeed(obj);
    const double yaw_reliability = trackedYawReliability(obj);
    // `yaw_reliability` is continuous inside [low, high].  Only q<=low must
    // take hard fallbacks; the transition band changes blend weights smoothly.
    const bool pca_unreliable = yaw_reliability <= 1e-6;
    const bool pca_degraded = yaw_reliability < (1.0 - 1e-6);
    const bool velocity_available = obj.velocity_in_map &&
        speed >= min_yaw_align_speed_mps_;
    // Preserve the prior high-quality / confidence-disabled behavior. Extra
    // hit confirmation is required only when confidence says PCA is degraded.
    const bool reliable_velocity = velocity_available &&
        (!pca_degraded ||
         obj.track_hits >= std::max(1, tracked_yaw_velocity_min_hits_));
    const bool lane_is_nearby = lane_projection &&
        std::abs(lane_projection->d) <=
            lane_half_width_ + bbox_overlap_margin_m_ + nearby_lane_margin_m_;

    // A newly observed, non-maneuvering vehicle has no trustworthy PCA yaw
    // history yet.  Initialize it from the directed lane tangent so a single
    // partial scan cannot inject a 90-degree OBB yaw error into the EMA.
    // Do not do this for a vehicle that is already crossing a lane boundary.
    if (lane_projection && use_lane_prior && !lane_change_candidate) {
      return smpc_lane_change::wrapToPi(lane_projection->yaw);
    }

    // A low-quality PCA angle must not dominate a lane-change candidate. If
    // motion is reliable, use its directed heading; otherwise preserve fresh
    // history (or a lane tangent only when there is no history).
    if (pca_unreliable && lane_change_candidate && !reliable_velocity) {
      const auto it = yaw_history_.find(obj.id);
      if (has_fresh_history && it != yaw_history_.end() && it->second.initialized) {
        yaw = it->second.yaw;
      } else if (lane_is_nearby) {
        yaw = smpc_lane_change::wrapToPi(lane_projection->yaw);
      }
    }

    if (reliable_velocity) {
      const double velocity_yaw = std::atan2(obj.vy, obj.vx);
      if (pca_unreliable) {
        // Do not blend a known-bad PCA axis into a directed velocity heading.
        yaw = velocity_yaw;
      } else {
        yaw = closestYawEquivalent(yaw, velocity_yaw);
        const double velocity_weight = std::clamp(
            tracked_yaw_velocity_blend_weight_ * yaw_reliability +
                tracked_yaw_low_confidence_velocity_blend_weight_ *
                    (1.0 - yaw_reliability),
            0.0, 1.0);
        yaw = angleLerp(yaw, velocity_yaw, velocity_weight);
      }
    } else if (pca_unreliable && lane_is_nearby && !lane_change_candidate) {
      // Slow/stopped, low-quality clusters have no reliable physical heading.
      // Use the directed lane tangent before footprint/lane assignment.
      return smpc_lane_change::wrapToPi(lane_projection->yaw);
    }

    if (lane_is_nearby &&
        tracked_yaw_lane_blend_weight_ > 0.0 &&
        lane_projection) {
      double lane_yaw = lane_projection->yaw;
      if (std::cos(smpc_lane_change::wrapToPi(yaw - lane_yaw)) < 0.0) {
        lane_yaw = smpc_lane_change::wrapToPi(lane_yaw + M_PI);
      }

      const double lane_error = std::abs(smpc_lane_change::wrapToPi(yaw - lane_yaw));
      const bool allow_low_confidence_lane_prior =
          pca_unreliable && !lane_change_candidate;
      if (lane_error <= tracked_yaw_lane_max_error_rad_ || allow_low_confidence_lane_prior) {
        const double lateral_speed = lateralSpeedInLane(obj, *lane_projection);
        const double reduce_scale = std::max(0.1, tracked_yaw_lateral_speed_reduce_mps_);
        const double lateral_alpha =
            std::clamp(std::abs(lateral_speed) / reduce_scale, 0.0, 1.0);
        const double reduction_gain =
            std::clamp(tracked_yaw_lateral_reduction_gain_, 0.0, 1.0);
        const double maneuver_weight_scale = lane_change_candidate
            ? std::clamp(tracked_yaw_lane_change_weight_scale_, 0.0, 1.0)
            : 1.0;
        const double confidence_aware_lane_weight =
            (pca_unreliable && reliable_velocity)
                ? std::clamp(tracked_yaw_low_confidence_lane_blend_weight_, 0.0, 1.0)
                : tracked_yaw_lane_blend_weight_ +
                      (1.0 - tracked_yaw_lane_blend_weight_) *
                          (1.0 - yaw_reliability);
        const double lane_weight = std::clamp(
            confidence_aware_lane_weight * maneuver_weight_scale *
                (1.0 - reduction_gain * lateral_alpha),
            0.0,
            1.0);
        yaw = angleLerp(yaw, lane_yaw, lane_weight);
      }
    }

    return smpc_lane_change::wrapToPi(yaw);
  }

  double updateYawHistoryFiltered(int id, double measured_yaw, const ros::Time& stamp) {
    auto& h = yaw_history_[id];
    measured_yaw = smpc_lane_change::wrapToPi(measured_yaw);

    if (!h.initialized) {
      h.initialized = true;
      h.yaw = measured_yaw;
      h.filtered_rate = 0.0;
      h.stamp = stamp;
      return measured_yaw;
    }

    const double dt = (stamp - h.stamp).toSec();
    if (dt <= 1e-3 || dt > tracked_yaw_history_timeout_sec_) {
      h.yaw = measured_yaw;
      h.filtered_rate = 0.0;
      h.stamp = stamp;
      return measured_yaw;
    }

    const double alpha = std::clamp(tracked_yaw_filter_alpha_, 0.0, 1.0);
    const double max_step = std::max(0.0, tracked_yaw_max_step_rad_);
    double step = alpha * smpc_lane_change::wrapToPi(measured_yaw - h.yaw);
    if (max_step > 1e-6) {
      step = std::clamp(step, -max_step, max_step);
    }

    const double prev_yaw = h.yaw;
    h.yaw = smpc_lane_change::wrapToPi(h.yaw + step);
    double raw_rate = smpc_lane_change::wrapToPi(h.yaw - prev_yaw) / dt;
    raw_rate = std::clamp(raw_rate, -max_abs_yaw_rate_, max_abs_yaw_rate_);
    h.filtered_rate = yaw_rate_alpha_ * raw_rate + (1.0 - yaw_rate_alpha_) * h.filtered_rate;
    h.stamp = stamp;
    return h.yaw;
  }

  void stabilizeTrackedYaw(CandidateObject& obj, const ros::Time& stamp) {
    if (!stabilize_tracked_yaw_ || obj.id < 0) {
      obj.yaw_rate = updateYawRate(obj.id, obj.yaw, stamp);
      return;
    }

    Projection nearest_projection;
    const bool has_lane_projection = nearestLaneProjection(obj, nearest_projection);
    const bool lane_change_candidate = updateLaneChangeYawCandidate(
        obj, has_lane_projection ? &nearest_projection : nullptr, stamp);
    const bool has_fresh_history = hasFreshYawHistory(obj.id, stamp);
    const bool use_lane_prior =
        obj.track_hits <= std::max(1, tracked_yaw_lane_prior_hits_) ||
        !has_fresh_history;
    const double measured_yaw = fusedTrackedYawMeasurement(
        obj, has_lane_projection ? &nearest_projection : nullptr, use_lane_prior,
        lane_change_candidate, has_fresh_history);
    obj.yaw = updateYawHistoryFiltered(obj.id, measured_yaw, stamp);
    obj.yaw_rate = yaw_history_[obj.id].filtered_rate;
  }

  double yawAlignedToLaneWhenSlow(const CandidateObject& obj,
                                  const Projection& projection) const {
    double yaw = smpc_lane_change::wrapToPi(obj.yaw);
    if (!lane_yaw_align_when_slow_) return yaw;
    if (laneChangeYawCandidateActive(obj.id)) return yaw;

    const double max_speed = std::max(0.1, lane_yaw_align_max_speed_mps_);
    const double speed = objectSpeed(obj);
    if (speed >= max_speed) return yaw;

    double lane_yaw = projection.yaw;
    if (std::cos(smpc_lane_change::wrapToPi(yaw - lane_yaw)) < 0.0) {
      lane_yaw = smpc_lane_change::wrapToPi(lane_yaw + M_PI);
    }

    const double slow_alpha = std::clamp((max_speed - speed) / max_speed, 0.0, 1.0);
    const double confidence_aware_weight = lane_yaw_align_weight_ +
        (1.0 - lane_yaw_align_weight_) * (1.0 - trackedYawReliability(obj));
    const double alpha = std::clamp(confidence_aware_weight * slow_alpha, 0.0, 1.0);
    return angleLerp(yaw, lane_yaw, alpha);
  }

  static LaneFootprint footprintOnLane(const CandidateObject& obj, double lane_yaw) {
    const double yaw_diff = smpc_lane_change::wrapToPi(obj.yaw - lane_yaw);
    const double c = std::abs(std::cos(yaw_diff));
    const double s = std::abs(std::sin(yaw_diff));
    LaneFootprint footprint;
    footprint.half_s = 0.5 * (obj.length * c + obj.width * s);
    footprint.half_d = 0.5 * (obj.length * s + obj.width * c);
    return footprint;
  }

  bool holdsFeaturedRole(int id) const {
    return id >= 0 && (id == current_front_selection_.held_id ||
                       id == target_front_selection_.held_id ||
                       id == target_rear_selection_.held_id);
  }

  // 길고 납작한 트랙인가.  height 가 NaN(정보 없음)이면 판정하지 않는다.
  bool isFlatLongTrack(const CandidateObject& obj) const {
    if (representative_flat_max_length_m_ <= 0.0) return false;
    if (!std::isfinite(obj.height) || !std::isfinite(obj.length)) return false;
    return obj.length >= representative_flat_max_length_m_ &&
           obj.height <= representative_flat_height_m_;
  }

  bool passesObjectFilter(const CandidateObject& obj) const {
    if (!std::isfinite(obj.x) || !std::isfinite(obj.y) ||
        !std::isfinite(obj.length) || !std::isfinite(obj.width)) {
      return false;
    }
    // A partially observed vehicle's measured extent swings across the minimum
    // between scans, and dropping the object here removes it from every role at
    // once.  Let a vehicle that already holds a role shrink further before it
    // is discarded, so one physical target is not torn out of target_front /
    // target_rear and reinstated a frame later.
    const double size_relief =
        (sticky_role_selection_enabled_ && holdsFeaturedRole(obj.id))
            ? sticky_role_size_hysteresis_m_ : 0.0;
    if (obj.length < min_object_length_m_ - size_relief ||
        obj.length > max_object_length_m_ + size_relief) {
      return false;
    }
    if (obj.width < min_object_width_m_ - size_relief ||
        obj.width > max_object_width_m_ + size_relief) {
      return false;
    }
    return true;
  }

  TargetVehicle makeVehicle(const CandidateObject& obj,
                            const Projection& projection,
                            double ego_s,
                            int lane_id,
                            const std::string& role) const {
    TargetVehicle out;
    out.valid = true;
    out.unique_id = obj.id;
    out.role = role;
    out.lane_id = lane_id;
    out.track_hits = obj.track_hits;
    out.x = obj.x;
    out.y = obj.y;
    out.s = projection.s;
    out.d = projection.d;
    out.delta_s = projection.s - ego_s;
    const LaneFootprint footprint = footprintOnLane(obj, projection.yaw);
    out.half_s = footprint.half_s;
    out.half_d = footprint.half_d;
    out.front_delta_s = out.delta_s + footprint.half_s;
    out.rear_delta_s = out.delta_s - footprint.half_s;
    out.yaw = yawAlignedToLaneWhenSlow(obj, projection);
    out.yaw_rate = obj.yaw_rate;
    if (obj.velocity_in_map) {
      const double c = std::cos(projection.yaw);
      const double s = std::sin(projection.yaw);
      out.v_long = obj.vx * c + obj.vy * s;
      out.v_lat = -obj.vx * s + obj.vy * c;
    } else {
      out.v_long = obj.v_long;
      out.v_lat = obj.v_lat;
    }
    out.length = obj.length;
    out.width = obj.width;
    return out;
  }

  bool makeNearbyVehicle(const CandidateObject& obj, TargetVehicle& out) const {
    bool found = false;
    double best_score = std::numeric_limits<double>::infinity();
    TargetVehicle best;

    for (const auto& lane_pair : lanes_) {
      const int lane_id = lane_pair.first;
      const auto& lane = lane_pair.second;
      const Projection vehicle_projection = lane.projectExtended(
          obj.x, obj.y, target_lane_projection_extension_before_m_);
      const Projection ego_projection = lane.projectExtended(
          ego_x_, ego_y_, target_lane_projection_extension_before_m_);
      if (!vehicle_projection.valid || !ego_projection.valid) continue;

      const LaneFootprint footprint = footprintOnLane(obj, vehicle_projection.yaw);
      const double lane_clearance = std::abs(vehicle_projection.d) - footprint.half_d;
      if (lane_clearance > lane_half_width_ + bbox_overlap_margin_m_ + nearby_lane_margin_m_) {
        continue;
      }

      TargetVehicle candidate =
          makeVehicle(obj, vehicle_projection, ego_projection.s, lane_id, "nearby");
      if (candidate.front_delta_s < -nearby_rear_range_ ||
          candidate.rear_delta_s > nearby_front_range_) {
        continue;
      }

      if (candidate.rear_delta_s > 0.0) {
        candidate.role = "nearby_front";
      } else if (candidate.front_delta_s < 0.0) {
        candidate.role = "nearby_rear";
      } else {
        candidate.role = "nearby_overlap";
      }

      const double score =
          std::max(0.0, lane_clearance) +
          0.01 * std::abs(candidate.delta_s) +
          0.001 * std::abs(lane_id - current_lane_id_);
      if (!found || score < best_score) {
        found = true;
        best_score = score;
        best = candidate;
      }
    }

    if (!found) return false;
    out = best;
    return true;
  }

  static TargetVehicle invalidVehicle(const std::string& role, int lane_id) {
    TargetVehicle v;
    v.valid = false;
    v.unique_id = -1;
    v.role = role;
    v.lane_id = lane_id;
    return v;
  }

  void publishTargetsFromCandidates(const std_msgs::Header& header,
                                    const std::vector<CandidateObject>& objects) {
    if (!ego_ready_ || !lane_ready_) return;
    const bool lane_change_required_now = current_lane_id_ < final_lane_id_;
    const int target_lane_id = forced_target_lane_id_ >= 0
        ? forced_target_lane_id_
        : (lane_change_required_now ? current_lane_id_ + target_lane_offset_ : current_lane_id_);
    auto current_it = lanes_.find(current_lane_id_);
    auto target_it = lanes_.find(target_lane_id);
    if (current_it == lanes_.end()) {
      ROS_WARN_THROTTLE(1.0, "[target_selector] invalid current lane=%d", current_lane_id_);
      return;
    }
    if (target_it == lanes_.end()) {
      ROS_WARN_THROTTLE(1.0, "[target_selector] invalid target lane=%d from current=%d", target_lane_id, current_lane_id_);
      return;
    }

    const Projection ego_current = current_it->second.project(ego_x_, ego_y_);
    const Projection ego_target = target_it->second.projectExtended(
        ego_x_, ego_y_, target_lane_projection_extension_before_m_);
    if (!ego_current.valid || !ego_target.valid) return;

    // An id held for a lane pair says nothing about a different lane pair.
    if (current_lane_id_ != held_role_current_lane_id_ ||
        target_lane_id != held_role_target_lane_id_) {
      current_front_selection_ = RoleSelection{};
      target_front_selection_ = RoleSelection{};
      target_rear_selection_ = RoleSelection{};
      held_role_current_lane_id_ = current_lane_id_;
      held_role_target_lane_id_ = target_lane_id;
    }

    TargetVehicle current_front = invalidVehicle("current_front", current_lane_id_);
    TargetVehicle target_front = invalidVehicle("target_front", target_lane_id);
    TargetVehicle target_rear = invalidVehicle("target_rear", target_lane_id);
    double current_front_ds = max_front_range_;
    double target_front_ds = max_front_range_;
    double target_rear_abs_ds = max_rear_range_;
    std::vector<TargetVehicle> nearby_vehicles;
    nearby_vehicles.reserve(objects.size());

    const ros::Time hold_now = header.stamp.isZero() ? ros::Time::now() : header.stamp;
    pruneConfidenceHold(hold_now);

    for (const auto& obj : objects) {
      // The LiDAR tracker keeps every geometry cluster alive.  Only confirmed
      // vehicles may influence the SMPC multi-vehicle prediction.  A partly
      // vehicle-like but unconfirmed track is retained as a conservative
      // current/target-lane obstacle for ACC and merge-gap safety; clear
      // structures below this threshold are ignored.
      const bool confidence_held = updateConfidenceHold(obj, hold_now);
      const bool confirmed_vehicle =
          !tracked_vehicle_filter_enabled_ ||
          (obj.is_vehicle && obj.vehicle_confidence >= tracked_vehicle_min_confidence_) ||
          confidence_held;
      const bool unknown_safety_obstacle =
          tracked_vehicle_filter_enabled_ && !confirmed_vehicle &&
          obj.vehicle_confidence >= tracked_unknown_obstacle_min_confidence_;
      if (!confirmed_vehicle && !unknown_safety_obstacle) continue;

      if (confirmed_vehicle) {
        TargetVehicle nearby;
        if (makeNearbyVehicle(obj, nearby)) {
          nearby_vehicles.push_back(nearby);
        }
      }

      const Projection pc = current_it->second.project(obj.x, obj.y);
      const Projection pt = target_it->second.projectExtended(
          obj.x, obj.y, target_lane_projection_extension_before_m_);
      const LaneFootprint fc = pc.valid ? footprintOnLane(obj, pc.yaw) : LaneFootprint{};
      const LaneFootprint ft = pt.valid ? footprintOnLane(obj, pt.yaw) : LaneFootprint{};

      const bool on_current_lane = pc.valid &&
          (std::abs(pc.d) - fc.half_d) <= (lane_half_width_ + bbox_overlap_margin_m_);
      const bool on_target_lane = pt.valid &&
          (std::abs(pt.d) - ft.half_d) <= (lane_half_width_ + bbox_overlap_margin_m_);
      const double current_lane_clearance = std::abs(pc.d) - fc.half_d;
      const double target_lane_clearance = std::abs(pt.d) - ft.half_d;
      const bool assign_current_lane = on_current_lane &&
          (!on_target_lane || current_lane_clearance <= target_lane_clearance);
      const bool assign_target_lane = on_target_lane &&
          (!on_current_lane || target_lane_clearance < current_lane_clearance);
      const bool flat_long_track = isFlatLongTrack(obj);
      const bool representative_current_lane = assign_current_lane &&
          !flat_long_track &&
          (!representative_lane_center_filter_enabled_ ||
           std::abs(pc.d) <= representative_lane_center_max_abs_d_m_);
      const bool representative_target_lane = assign_target_lane &&
          !flat_long_track &&
          (!representative_lane_center_filter_enabled_ ||
           std::abs(pt.d) <= representative_lane_center_max_abs_d_m_);

      if (representative_current_lane) {
        const double ds = pc.s - ego_current.s;
        const double front_edge_ds = ds + fc.half_s;
        const double rear_edge_ds = ds - fc.half_s;
        if (front_edge_ds > 0.0) {
          const double front_gap = std::max(0.0, rear_edge_ds);
          if (front_gap < current_front_ds) {
            current_front_ds = front_gap;
            current_front = makeVehicle(obj, pc, ego_current.s, current_lane_id_, "current_front");
          }
        }
      }

      if (representative_target_lane) {
        const double ds = pt.s - ego_target.s;
        const double front_edge_ds = ds + ft.half_s;
        const double rear_edge_ds = ds - ft.half_s;
        if (front_edge_ds >= 0.0) {
          const double front_gap = std::max(0.0, rear_edge_ds);
          if (front_gap < target_front_ds) {
            target_front_ds = front_gap;
            target_front = makeVehicle(obj, pt, ego_target.s, target_lane_id, "target_front");
          }
        }
        if (rear_edge_ds <= 0.0) {
          const double rear_gap = std::max(0.0, -front_edge_ds);
          if (rear_gap < target_rear_abs_ds) {
            target_rear_abs_ds = rear_gap;
            target_rear = makeVehicle(obj, pt, ego_target.s, target_lane_id, "target_rear");
          }
        }
      }
    }

    current_front_selection_.held_id = current_front.valid ? current_front.unique_id : -1;
    target_front_selection_.held_id = target_front.valid ? target_front.unique_id : -1;
    target_rear_selection_.held_id = target_rear.valid ? target_rear.unique_id : -1;

    TargetVehicleSet out;
    out.header = header;
    out.header.frame_id = "map";
    out.current_lane_id = current_lane_id_;
    out.target_lane_id = target_lane_id;
    out.ego_s_current = ego_current.s;
    out.ego_d_current = ego_current.d;
    out.ego_d_target = ego_target.d;
    out.current_lane_length = current_it->second.length();
    out.distance_to_lane_end = std::max(0.0, out.current_lane_length - out.ego_s_current);
    out.time_to_lane_end = ego_speed_mps_ > 0.1
        ? out.distance_to_lane_end / ego_speed_mps_
        : std::numeric_limits<double>::infinity();

    out.lane_change_required = current_lane_id_ < final_lane_id_;
    out.lane_change_prepare = out.lane_change_required &&
        (out.distance_to_lane_end <= lane_end_prepare_distance_m_ ||
         out.time_to_lane_end <= lane_end_prepare_time_sec_);
    out.lane_change_urgent = out.lane_change_required &&
        (out.distance_to_lane_end <= lane_end_urgent_distance_m_ ||
         out.time_to_lane_end <= lane_end_urgent_time_sec_);
    out.emergency_stop_required = out.lane_change_required &&
        (out.distance_to_lane_end <= lane_end_emergency_distance_m_ ||
         out.time_to_lane_end <= lane_end_emergency_time_sec_);

    out.current_front = current_front;
    out.target_front = target_front;
    out.target_rear = target_rear;
    std::sort(nearby_vehicles.begin(), nearby_vehicles.end(),
              [](const TargetVehicle& a, const TargetVehicle& b) {
                return std::abs(a.delta_s) < std::abs(b.delta_s);
              });
    if (max_nearby_vehicles_ > 0 &&
        nearby_vehicles.size() > static_cast<std::size_t>(max_nearby_vehicles_)) {
      nearby_vehicles.resize(static_cast<std::size_t>(max_nearby_vehicles_));
    }
    out.nearby_vehicles = nearby_vehicles;
    target_pub_.publish(out);
    publishMarkers(out);
  }

  void objectCallback(const morai_msgs::ObjectStatusList::ConstPtr& msg) {
    std::vector<CandidateObject> objects;
    objects.reserve(msg->npc_list.size());

    for (const auto& npc : msg->npc_list) {
      CandidateObject obj;
      obj.id = npc.unique_id;
      obj.x = npc.position.x;
      obj.y = npc.position.y;
      obj.yaw = npc.heading * M_PI / 180.0;
      // MORAI ObjectStatus.msg documents velocity in km/h. x/y are vehicle-local longitudinal/lateral components.
      obj.velocity_in_map = false;
      obj.v_long = npc.velocity.x / 3.6;
      obj.v_lat = npc.velocity.y / 3.6;
      // Some MORAI versions/documentation swap size x/y labels. Vehicle length is safely the larger horizontal dimension.
      obj.length = std::max(npc.size.x, npc.size.y);
      obj.width = std::min(npc.size.x, npc.size.y);
      obj.vehicle_confidence = 1.0;
      obj.is_vehicle = true;
      normalizeBoxAxes(obj);
      obj.yaw_rate = updateYawRate(obj.id, obj.yaw, msg->header.stamp);
      if (passesObjectFilter(obj)) objects.push_back(obj);
    }

    publishTargetsFromCandidates(msg->header, objects);
  }

  void trackedObjectsCallback(const lidar_ttc_tracker::TrackedObjectArray::ConstPtr& msg) {
    std::vector<CandidateObject> objects;
    objects.reserve(msg->objects.size());

    for (const auto& tracked : msg->objects) {
      if (static_cast<int>(tracked.hits) < min_tracked_hits_) continue;
      if (tracked_dynamic_only_ && !tracked.is_dynamic) continue;

      CandidateObject obj;
      obj.id = static_cast<int>(tracked.id);
      obj.x = tracked.x;
      obj.y = tracked.y;
      obj.yaw = tracked.yaw;
      obj.yaw_confidence = std::isfinite(tracked.yaw_confidence)
          ? std::clamp(tracked.yaw_confidence, 0.0, 1.0)
          : 0.0;
      obj.velocity_in_map = true;
      obj.vx = tracked.vx;
      obj.vy = tracked.vy;
      obj.length = tracked.size_x;
      obj.width = tracked.size_y;
      obj.height = tracked.size_z;
      obj.vehicle_confidence = std::clamp(tracked.vehicle_confidence, 0.0, 1.0);
      obj.is_vehicle = tracked.is_vehicle;
      obj.track_hits = static_cast<int>(tracked.hits);
      normalizeBoxAxes(obj);
      alignYawWithVelocity(obj);
      stabilizeTrackedYaw(obj, msg->header.stamp);
      if (passesObjectFilter(obj)) objects.push_back(obj);
    }

    pruneStaleLaneChangeYawCandidates(msg->header.stamp);

    publishTargetsFromCandidates(msg->header, objects);
  }

  static double yawFromQuaternion(const geometry_msgs::Quaternion& q) {
    const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
    const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    return std::atan2(siny_cosp, cosy_cosp);
  }

  static geometry_msgs::Point point(double x, double y, double z) {
    geometry_msgs::Point p;
    p.x = x;
    p.y = y;
    p.z = z;
    return p;
  }

  static void setColor(visualization_msgs::Marker& marker,
                       double r, double g, double b, double a) {
    marker.color.r = r;
    marker.color.g = g;
    marker.color.b = b;
    marker.color.a = a;
  }

  static void setRoleColor(visualization_msgs::Marker& marker, int role_index, double alpha) {
    switch (role_index) {
      case 0:
        setColor(marker, 1.0, 0.20, 0.15, alpha);
        break;
      case 1:
        setColor(marker, 0.10, 0.90, 0.30, alpha);
        break;
      case 2:
        setColor(marker, 0.20, 0.45, 1.00, alpha);
        break;
      case 3:
        setColor(marker, 1.0, 0.85, 0.20, alpha);
        break;
      case 4:
        setColor(marker, 0.85, 0.95, 1.00, alpha);
        break;
      default:
        setColor(marker, 1.0, 1.0, 1.0, alpha);
        break;
    }
  }

  static void setLaneColor(visualization_msgs::Marker& marker, int lane_id, double alpha) {
    const int color_index = ((lane_id % 6) + 6) % 6;
    switch (color_index) {
      case 0:
        setColor(marker, 0.15, 0.85, 1.00, alpha);
        break;
      case 1:
        setColor(marker, 1.00, 0.65, 0.15, alpha);
        break;
      case 2:
        setColor(marker, 0.35, 1.00, 0.35, alpha);
        break;
      case 3:
        setColor(marker, 1.00, 0.35, 0.75, alpha);
        break;
      case 4:
        setColor(marker, 0.75, 0.55, 1.00, alpha);
        break;
      default:
        setColor(marker, 1.00, 1.00, 0.25, alpha);
        break;
    }
  }

  ros::Duration targetMarkerLifetime() const {
    return ros::Duration(std::max(0.0, debug_marker_lifetime_sec_));
  }

  void appendVehicleBody(visualization_msgs::MarkerArray& array,
                         const std_msgs::Header& header,
                         const std::string& ns,
                         int id,
                         int role_index,
                         double x,
                         double y,
                         double yaw,
                         double length,
                         double width) const {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = (debug_use_vehicle_mesh_ && !debug_vehicle_mesh_resource_.empty())
        ? visualization_msgs::Marker::MESH_RESOURCE
        : visualization_msgs::Marker::CUBE;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = 0.5 * debug_vehicle_height_m_;
    marker.pose.orientation.z = std::sin(0.5 * yaw);
    marker.pose.orientation.w = std::cos(0.5 * yaw);
    marker.scale.x = std::max(0.1, length);
    marker.scale.y = std::max(0.1, width);
    marker.scale.z = std::max(0.1, debug_vehicle_height_m_);
    marker.lifetime = targetMarkerLifetime();
    setRoleColor(marker, role_index, 0.85);
    if (marker.type == visualization_msgs::Marker::MESH_RESOURCE) {
      marker.mesh_resource = debug_vehicle_mesh_resource_;
      marker.mesh_use_embedded_materials = false;
    }
    array.markers.push_back(marker);
  }

  void appendHeadingArrow(visualization_msgs::MarkerArray& array,
                          const std_msgs::Header& header,
                          const std::string& ns,
                          int id,
                          int role_index,
                          double x,
                          double y,
                          double yaw,
                          double length) const {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::ARROW;
    marker.pose.orientation.w = 1.0;
    marker.points.push_back(point(x, y, debug_vehicle_height_m_ + 0.25));
    marker.points.push_back(point(x + std::cos(yaw) * std::max(1.0, 0.55 * length),
                                  y + std::sin(yaw) * std::max(1.0, 0.55 * length),
                                  debug_vehicle_height_m_ + 0.25));
    marker.scale.x = 0.20;
    marker.scale.y = 0.45;
    marker.scale.z = 0.45;
    marker.lifetime = targetMarkerLifetime();
    setRoleColor(marker, role_index, 0.9);
    array.markers.push_back(marker);
  }

  void appendTextLabel(visualization_msgs::MarkerArray& array,
                       const std_msgs::Header& header,
                       const std::string& ns,
                       int id,
                       int role_index,
                       double x,
                       double y,
                       const std::string& text) const {
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = ns;
    marker.id = id;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.pose.position.x = x;
    marker.pose.position.y = y;
    marker.pose.position.z = debug_label_height_m_;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = 1.0;
    marker.text = text;
    marker.lifetime = targetMarkerLifetime();
    setRoleColor(marker, role_index, 1.0);
    array.markers.push_back(marker);
  }

  // Track id on its own, larger and above the detail text, so an id switch is
  // readable from a normal RViz viewpoint without selecting the marker.
  void appendIdLabel(visualization_msgs::MarkerArray& array,
                     const std_msgs::Header& header,
                     const TargetVehicle& vehicle,
                     int role_index) const {
    if (!debug_show_id_labels_ || vehicle.unique_id < 0) return;
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "smpc_id_labels";
    // unique_id keys the marker, so a featured and a nearby vehicle can never
    // claim the same slot.
    marker.id = vehicle.unique_id;
    marker.action = visualization_msgs::Marker::ADD;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.pose.position.x = vehicle.x;
    marker.pose.position.y = vehicle.y;
    marker.pose.position.z = debug_id_label_height_m_;
    marker.pose.orientation.w = 1.0;
    marker.scale.z = std::max(0.1, debug_id_label_scale_);
    marker.text = "#" + std::to_string(vehicle.unique_id);
    marker.lifetime = targetMarkerLifetime();
    setRoleColor(marker, role_index, 1.0);
    array.markers.push_back(marker);
  }

  std::string targetLabel(const TargetVehicle& vehicle) const {
    std::ostringstream ss;
    ss.setf(std::ios::fixed);
    ss << std::setprecision(1)
       << vehicle.role << " id=" << vehicle.unique_id << "\n"
       << "lane=" << vehicle.lane_id
       << " ds=" << vehicle.delta_s << "m"
       << " v=" << vehicle.v_long << "m/s";
    return ss.str();
  }

  void appendTargetVehicle(visualization_msgs::MarkerArray& array,
                           const std_msgs::Header& header,
                           const TargetVehicle& vehicle,
                           int role_index) const {
    appendVehicleBody(array, header, "smpc_target_vehicles", role_index, role_index,
                      vehicle.x, vehicle.y, vehicle.yaw, vehicle.length, vehicle.width);
    appendHeadingArrow(array, header, "smpc_target_heading", role_index, role_index,
                       vehicle.x, vehicle.y, vehicle.yaw, vehicle.length);
    appendTextLabel(array, header, "smpc_target_labels", role_index, role_index,
                    vehicle.x, vehicle.y, targetLabel(vehicle));
    appendIdLabel(array, header, vehicle, role_index);
  }

  bool isFeaturedVehicle(const TargetVehicle& vehicle,
                         const TargetVehicleSet& set) const {
    if (vehicle.unique_id < 0) return false;
    const TargetVehicle featured[3] = {set.current_front, set.target_front, set.target_rear};
    for (const auto& selected : featured) {
      if (selected.valid && selected.unique_id == vehicle.unique_id) return true;
    }
    return false;
  }

  void appendNearbyVehicle(visualization_msgs::MarkerArray& array,
                           const std_msgs::Header& header,
                           const TargetVehicle& vehicle,
                           int marker_id) const {
    appendVehicleBody(array, header, "smpc_nearby_vehicles", marker_id, 4,
                      vehicle.x, vehicle.y, vehicle.yaw, vehicle.length, vehicle.width);
    appendHeadingArrow(array, header, "smpc_nearby_heading", marker_id, 4,
                       vehicle.x, vehicle.y, vehicle.yaw, vehicle.length);
    if (debug_show_nearby_labels_) {
      appendTextLabel(array, header, "smpc_nearby_labels", marker_id, 4,
                      vehicle.x, vehicle.y, targetLabel(vehicle));
    }
    appendIdLabel(array, header, vehicle, 4);
  }

  void publishCsvPathMarkers() const {
    visualization_msgs::MarkerArray array;
    std_msgs::Header header;
    header.frame_id = "map";
    header.stamp = ros::Time::now();

    visualization_msgs::Marker clear;
    clear.header = header;
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);

    for (const auto& lane_pair : lanes_) {
      const int lane_id = lane_pair.first;
      const auto& lane = lane_pair.second;
      const auto& points = lane.points();
      if (points.empty()) continue;

      visualization_msgs::Marker line;
      line.header = header;
      line.ns = "smpc_csv_paths";
      line.id = lane_id;
      line.type = visualization_msgs::Marker::LINE_STRIP;
      line.action = visualization_msgs::Marker::ADD;
      line.pose.orientation.w = 1.0;
      line.scale.x = 0.35;
      setLaneColor(line, lane_id, 0.95);

      const int max_points = std::max(2, debug_max_csv_points_per_lane_);
      const std::size_t stride = points.size() > static_cast<std::size_t>(max_points)
          ? static_cast<std::size_t>(std::ceil(points.size() / static_cast<double>(max_points)))
          : 1;
      for (std::size_t i = 0; i < points.size(); i += stride) {
        line.points.push_back(point(points[i].x, points[i].y, debug_csv_path_z_m_));
      }
      const auto& last = points.back();
      if (line.points.empty() ||
          line.points.back().x != last.x ||
          line.points.back().y != last.y) {
        line.points.push_back(point(last.x, last.y, debug_csv_path_z_m_));
      }
      array.markers.push_back(line);

      visualization_msgs::Marker label;
      label.header = header;
      label.ns = "smpc_csv_path_labels";
      label.id = lane_id;
      label.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
      label.action = visualization_msgs::Marker::ADD;
      label.pose.position.x = points.front().x;
      label.pose.position.y = points.front().y;
      label.pose.position.z = 2.0;
      label.pose.orientation.w = 1.0;
      label.scale.z = 1.2;
      auto path_it = lane_csv_paths_.find(lane_id);
      label.text = "lane " + std::to_string(lane_id);
      if (debug_show_csv_file_path_ && path_it != lane_csv_paths_.end()) {
        label.text += "\n" + path_it->second;
      }
      setLaneColor(label, lane_id, 1.0);
      array.markers.push_back(label);
    }

    csv_path_marker_pub_.publish(array);
  }

  void publishMarkers(const TargetVehicleSet& set) {
    visualization_msgs::MarkerArray array;
    visualization_msgs::Marker clear;
    clear.header = set.header;
    clear.action = visualization_msgs::Marker::DELETEALL;
    array.markers.push_back(clear);

    if (debug_show_ego_vehicle_ && ego_ready_) {
      appendVehicleBody(array, set.header, "smpc_ego_vehicle", 0, 3,
                        ego_x_, ego_y_, ego_yaw_, 4.8, 1.9);
      appendHeadingArrow(array, set.header, "smpc_ego_heading", 0, 3,
                         ego_x_, ego_y_, ego_yaw_, 4.8);
      std::ostringstream ss;
      ss.setf(std::ios::fixed);
      ss << std::setprecision(1)
         << "ego\nlane=" << current_lane_id_
         << " v=" << ego_speed_mps_ << "m/s";
      appendTextLabel(array, set.header, "smpc_ego_label", 0, 3,
                      ego_x_, ego_y_, ss.str());
    }

    const TargetVehicle vehicles[3] = {set.current_front, set.target_front, set.target_rear};
    for (int i = 0; i < 3; ++i) {
      if (!vehicles[i].valid) continue;
      appendTargetVehicle(array, set.header, vehicles[i], i);
    }

    if (debug_show_nearby_vehicles_) {
      int marker_index = 0;
      for (const auto& vehicle : set.nearby_vehicles) {
        if (!vehicle.valid || isFeaturedVehicle(vehicle, set)) continue;
        const int marker_id = vehicle.unique_id >= 0 ? vehicle.unique_id : 1000 + marker_index;
        appendNearbyVehicle(array, set.header, vehicle, marker_id);
        ++marker_index;
      }
    }
    marker_pub_.publish(array);
  }

  ros::NodeHandle nh_, pnh_;
  ros::Subscriber object_sub_, tracked_objects_sub_, odom_sub_, state_sub_, status_sub_;
  ros::Publisher target_pub_, marker_pub_, csv_path_marker_pub_;
  std::string waypoint_directory_, target_source_, object_topic_, tracked_objects_topic_, odom_topic_, state_topic_;
  std::map<int, LanePath> lanes_;
  std::map<int, std::string> lane_csv_paths_;
  std::unordered_map<int, YawHistory> yaw_history_;
  std::unordered_map<int, ConfidenceHold> confidence_hold_;
  bool tracked_vehicle_confidence_hold_enabled_{false};
  int tracked_vehicle_confidence_hold_min_hits_{20};
  double tracked_vehicle_confidence_hold_stale_sec_{0.50};
  double tracked_vehicle_confidence_hold_timeout_sec_{0.35};
  std::unordered_map<int, LaneChangeCandidateHistory> lane_change_candidate_history_;
  bool ego_ready_{false}, lane_ready_{false};
  double ego_x_{0.0}, ego_y_{0.0}, ego_yaw_{0.0}, ego_speed_mps_{0.0};
  int current_lane_id_{0};
  int gps_lane_id_{0};
  int forced_target_lane_id_{-1};
  bool lane_change_active_{false};
  double lane_half_width_{2.2}, max_front_range_{120.0}, max_rear_range_{80.0};
  double target_lane_projection_extension_before_m_{100.0};
  int min_tracked_hits_{3};
  bool tracked_dynamic_only_{false};
  bool tracked_vehicle_filter_enabled_{true};
  double tracked_vehicle_min_confidence_{0.60};
  double tracked_unknown_obstacle_min_confidence_{0.42};
  double min_object_length_m_{1.0};
  double max_object_length_m_{8.0};
  double min_object_width_m_{0.4};
  double max_object_width_m_{3.5};
  bool align_tracked_yaw_with_velocity_{true};
  double min_yaw_align_speed_mps_{1.0};
  double bbox_overlap_margin_m_{0.2};
  bool representative_lane_center_filter_enabled_{true};
  double representative_lane_center_max_abs_d_m_{1.35};
  double representative_flat_max_length_m_{0.0};
  double representative_flat_height_m_{0.25};
  bool sticky_role_selection_enabled_{true};
  double sticky_role_size_hysteresis_m_{0.15};
  RoleSelection current_front_selection_;
  RoleSelection target_front_selection_;
  RoleSelection target_rear_selection_;
  int held_role_current_lane_id_{std::numeric_limits<int>::min()};
  int held_role_target_lane_id_{std::numeric_limits<int>::min()};
  int target_lane_offset_{1};
  double yaw_rate_alpha_{0.15}, max_abs_yaw_rate_{0.6};
  bool stabilize_tracked_yaw_{true};
  double tracked_yaw_filter_alpha_{0.18};
  double tracked_yaw_history_timeout_sec_{1.0};
  double tracked_yaw_max_step_rad_{0.07};
  double tracked_yaw_velocity_blend_weight_{0.25};
  double tracked_yaw_lane_blend_weight_{0.90};
  double tracked_yaw_lane_max_error_rad_{1.57};
  double tracked_yaw_lateral_reduction_gain_{0.0};
  double tracked_yaw_lateral_speed_reduce_mps_{0.8};
  int tracked_yaw_lane_prior_hits_{12};
  double tracked_yaw_lane_change_boundary_abs_d_m_{1.0};
  double tracked_yaw_lane_change_lateral_speed_mps_{0.35};
  double tracked_yaw_lane_change_weight_scale_{0.15};
  int tracked_yaw_lane_change_confirm_hits_{3};
  int tracked_yaw_lane_change_release_hits_{5};
  int tracked_yaw_lane_change_min_track_hits_{12};
  bool tracked_yaw_confidence_enabled_{true};
  double tracked_yaw_confidence_low_{0.35};
  double tracked_yaw_confidence_high_{0.70};
  double tracked_yaw_low_confidence_velocity_blend_weight_{0.70};
  double tracked_yaw_low_confidence_lane_blend_weight_{0.40};
  int tracked_yaw_velocity_min_hits_{3};
  int final_lane_id_{3};
  double lane_end_prepare_time_sec_{8.0};
  double lane_end_urgent_time_sec_{5.0};
  double lane_end_emergency_time_sec_{1.0};
  double lane_end_prepare_distance_m_{200.0};
  double lane_end_urgent_distance_m_{120.0};
  double lane_end_emergency_distance_m_{15.0};
  double nearby_front_range_{150.0};
  double nearby_rear_range_{60.0};
  double nearby_lane_margin_m_{1.0};
  int max_nearby_vehicles_{40};
  bool lane_yaw_align_when_slow_{true};
  double lane_yaw_align_max_speed_mps_{1.5};
  double lane_yaw_align_weight_{0.85};
  bool debug_publish_csv_paths_{true};
  bool debug_show_csv_file_path_{true};
  bool debug_show_ego_vehicle_{true};
  bool debug_show_nearby_vehicles_{true};
  bool debug_show_nearby_labels_{false};
  bool debug_use_vehicle_mesh_{true};
  std::string debug_vehicle_mesh_resource_{"package://smpc_lane_change/meshes/debug_vehicle.stl"};
  double debug_marker_lifetime_sec_{0.5};
  double debug_vehicle_height_m_{1.5};
  double debug_label_height_m_{2.5};
  bool debug_show_id_labels_{true};
  double debug_id_label_scale_{2.2};
  double debug_id_label_height_m_{4.2};
  double debug_csv_path_z_m_{0.05};
  int debug_max_csv_points_per_lane_{2500};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_target_selector");
  TargetSelectorNode node;
  ros::spin();
  return 0;
}
