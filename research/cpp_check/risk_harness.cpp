// 파이썬 이식(research/sim/smpc_sim/risk.py)이 원본 C++ 과 같은 값을 내는지 확인하는 하네스.
// 원본 노드 소스를 수정 없이 include 해 evaluateTrajectory 를 직접 호출한다.
//
// 입력 시나리오 파일 형식 (공백 구분):
//   ego_speed <v>
//   ego <K>            다음 K 줄: t x y yaw v
//   vehicles <M>       다음 M 줄: uid role lane_id s d delta_s yaw yaw_rate v_long
//                                  length width half_s front_delta_s rear_delta_s accel_obs
// 출력: 한 줄 JSON.
//
// 노드가 쓰는 헤더를 먼저 include 해 둔다. 그래야 아래 #define private public 이
// 표준·ROS 헤더가 아니라 노드 클래스에만 적용된다(include guard 로 재포함 안 됨).
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
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <smpc_lane_change/BehaviorLongitudinalRequest.h>
#include <smpc_lane_change/BehaviorLongitudinalStatus.h>
#include <smpc_lane_change/BehaviorPlan.h>
#include <smpc_lane_change/LaneChangeDecision.h>
#include <smpc_lane_change/TargetVehicle.h>
#include <smpc_lane_change/TargetVehicleSet.h>
#include <smpc_lane_change/lane_path.hpp>
#include <waypoint_system/PathSwitchPreview.h>

#define private public
#define main smpc_decision_original_main
#include "../../src/smpc_decision_node.cpp"
#undef main
#undef private

namespace {

void printArray(std::ostream& os, const std::vector<double>& a) {
  os << "[";
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (i) os << ",";
    if (std::isinf(a[i])) os << (a[i] > 0 ? "1e300" : "-1e300");
    else os << a[i];
  }
  os << "]";
}

}  // namespace

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_risk_harness");
  if (argc < 2) {
    std::cerr << "usage: risk_harness <scenario.txt>\n";
    return 2;
  }
  SmpcDecisionNode node;
  if (node.lanes_.empty()) {
    std::cerr << "no lanes loaded (check ~waypoint_directory)\n";
    return 3;
  }

  std::ifstream in(argv[1]);
  std::string key;
  double ego_speed = 0.0;
  std::vector<SmpcDecisionNode::PredPose> traj;
  std::vector<smpc_lane_change::TargetVehicle> vehicles;
  while (in >> key) {
    if (key == "ego_speed") {
      in >> ego_speed;
    } else if (key == "ego") {
      int k = 0;
      in >> k;
      for (int i = 0; i < k; ++i) {
        SmpcDecisionNode::PredPose p;
        in >> p.t >> p.x >> p.y >> p.yaw >> p.v;
        p.valid = true;
        traj.push_back(p);
      }
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
      }
    }
  }
  node.ego_speed_mps_ = ego_speed;

  const auto eval = node.evaluateTrajectory(traj, vehicles, nullptr, false);
  std::cout << std::setprecision(17) << "{\"valid\":" << (eval.valid ? "true" : "false")
            << ",\"detail\":\"" << eval.detail << "\""
            << ",\"peak\":" << eval.peak_step_risk << ",\"legacy\":" << eval.risk
            << ",\"risk_by_step\":";
  printArray(std::cout, eval.risk_by_step);
  std::cout << ",\"min_clear_by_step\":";
  printArray(std::cout, eval.min_signed_clearance_by_step);
  std::cout << ",\"traces\":[";
  bool first = true;
  for (const auto& vt : eval.vehicle_risk_traces) {
    for (const auto& mt : vt.modes) {
      if (!first) std::cout << ",";
      first = false;
      std::cout << "{\"uid\":" << vt.unique_id << ",\"mode\":\"" << mt.name
                << "\",\"p\":" << mt.probability << ",\"cond\":";
      printArray(std::cout, mt.conditional_risk_by_step);
      std::cout << "}";
    }
  }
  std::cout << "]}" << std::endl;
  return 0;
}
