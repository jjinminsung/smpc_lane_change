// C++ 트리 평가(treeEvaluate, treeBranches)가 파이썬 기준 구현(research/sim/smpc_sim/tree.py)과
// 같은지 확인하는 하네스. risk_harness.cpp 와 같은 방식으로 노드 소스를 include 한다.
//
// 입력 (공백 구분):
//   ego_speed <v>   mission <v>   targets <cur> <tgt> <ego_d_target>   rear <uid|-1>
//   vehicles <M>   (risk_harness 와 같은 줄 형식)
//   roots <R>      다음마다:  root <change 0/1> <root_cost> <K>  + K 줄 (t x y yaw v)
// 출력: 한 줄 JSON {branches:[[name,p,tb]], roots:[{feasible,cost,peak,combo}]}
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <geometry_msgs/Point.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
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

#define private public
#define main smpc_decision_original_main
#include "../../src/smpc_decision_node.cpp"
#undef main
#undef private

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_tree_harness");
  if (argc < 2) return 2;
  SmpcDecisionNode node;
  if (node.lanes_.empty()) {
    std::cerr << "no lanes\n";
    return 3;
  }
  std::ifstream in(argv[1]);
  std::string key;
  smpc_lane_change::TargetVehicleSet t;
  std::vector<smpc_lane_change::TargetVehicle> vehicles;
  int rear_uid = -1;
  struct Root { bool change; double cost; std::vector<SmpcDecisionNode::PredPose> traj; };
  std::vector<Root> roots;
  while (in >> key) {
    if (key == "ego_speed") {
      in >> node.ego_speed_mps_;
    } else if (key == "mission") {
      double m;
      in >> m;
      node.mission_speed_target_mps_ = m;
      node.have_mission_speed_target_ = true;
      node.mission_speed_timeout_sec_ = 0.0;
    } else if (key == "targets") {
      in >> t.current_lane_id >> t.target_lane_id >> t.ego_d_target;
    } else if (key == "ego_xy") {
      in >> node.ego_x_ >> node.ego_y_;
    } else if (key == "rear") {
      in >> rear_uid;
    } else if (key == "vehicles") {
      int m = 0;
      in >> m;
      for (int i = 0; i < m; ++i) {
        smpc_lane_change::TargetVehicle v;
        double accel_obs = 0.0;
        in >> v.unique_id >> v.role >> v.lane_id >> v.s >> v.d >> v.delta_s >> v.yaw >>
            v.yaw_rate >> v.v_long >> v.length >> v.width >> v.half_s >> v.front_delta_s >>
            v.rear_delta_s >> accel_obs;
        v.valid = true;
        auto& h = node.vehicle_history_[v.unique_id];
        h.initialized = true;
        h.v_long = v.v_long;
        h.filtered_accel = accel_obs;
        vehicles.push_back(v);
        if (v.role == "target_front") t.target_front = v;
        if (v.role == "target_rear") t.target_rear = v;
        if (v.role == "current_front") t.current_front = v;
      }
    } else if (key == "roots") {
      int r = 0;
      in >> r;
      for (int i = 0; i < r; ++i) {
        std::string tag;
        Root root;
        int change = 0, k = 0;
        in >> tag >> change >> root.cost >> k;
        root.change = change != 0;
        for (int j = 0; j < k; ++j) {
          SmpcDecisionNode::PredPose p;
          in >> p.t >> p.x >> p.y >> p.yaw >> p.v;
          p.valid = true;
          root.traj.push_back(p);
        }
        roots.push_back(root);
      }
    }
  }
  const smpc_lane_change::TargetVehicle* rear = nullptr;
  for (const auto& v : vehicles) {
    if (v.unique_id == rear_uid) rear = &v;
  }
  const auto branches = node.treeBranches(t, rear);
  std::cout << std::setprecision(17) << "{\"branches\":[";
  for (std::size_t b = 0; b < branches.size(); ++b) {
    if (b) std::cout << ",";
    std::cout << "[\"" << branches[b].name << "\"," << branches[b].probability << ","
              << (std::isfinite(branches[b].t_b) ? branches[b].t_b : 1e300) << "]";
  }
  std::cout << "],\"roots\":[";
  for (std::size_t i = 0; i < roots.size(); ++i) {
    const auto r = node.treeEvaluate(roots[i].traj, roots[i].change, t, vehicles, branches,
                                     roots[i].cost, node.risk_epsilon_);
    if (i) std::cout << ",";
    std::cout << "{\"feasible\":" << (r.feasible ? "true" : "false") << ",\"cost\":"
              << (std::isfinite(r.cost) ? r.cost : 1e300) << ",\"peak\":" << r.peak_risk << ",\"combo\":[";
    for (std::size_t b = 0; b < r.combo.size(); ++b) std::cout << (b ? "," : "") << r.combo[b];
    std::cout << "]}";
  }
  std::cout << "]";
  std::string detail;
  const bool ret = node.treeMonitorReturn(t, vehicles, rear, node.risk_epsilon_, &detail);
  std::cout << ",\"monitor_return\":" << (ret ? "true" : "false") << ",\"monitor_detail\":\"" << detail << "\"}" << std::endl;
  return 0;
}
