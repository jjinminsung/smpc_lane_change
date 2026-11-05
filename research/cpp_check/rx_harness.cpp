// C++ 반응형 예측(rxRollout)이 파이썬 기준 구현(research/sim/smpc_sim/reactive.py joint_rollout)과
// 같은지 확인하는 하네스. tree_harness.cpp 와 같은 방식으로 노드 소스를 include 한다.
//
// 입력 (공백 구분):
//   ego_speed <v>  mission <v>  yield_ref <v>  targets <cur> <tgt> <ego_d_target> <distance_to_lane_end>
//   flags <required> <prepare> <urgent> <emergency>  (0/1, 없으면 모두 0)
//   ego_xy <x> <y>  mode <connector 0/1> <join_arc> <mid_T_rem (<0 이면 판단)>
//   vehicles <M>  다음마다: uid role lane s d delta_s v_long length width rear_delta_s front_delta_s accel
//   specs <N>     다음마다: change t_sw dur action prep_a      (action: keep hold yield accel change_now)
//   branches <B>  다음마다: rear front tb p                   (tb 가 없으면 '-')
// 출력: 한 줄 JSON {valid, allowed, risk, ego_s, ego_d, ego_v, tv_s}
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
#include <set>
#include <sstream>
#include <string>
#include <tuple>
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

template <typename T>
static void dump(const char* name, const std::vector<T>& v, bool comma = true) {
  std::cout << "\"" << name << "\":[";
  for (std::size_t i = 0; i < v.size(); ++i) std::cout << (i ? "," : "") << static_cast<double>(v[i]);
  std::cout << "]" << (comma ? "," : "");
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_rx_harness");
  if (argc < 2) return 2;
  SmpcDecisionNode node;
  if (node.lanes_.empty()) {
    std::cerr << "no lanes\n";
    return 3;
  }
  std::ifstream in(argv[1]);
  std::string key;
  smpc_lane_change::TargetVehicleSet t;
  double mission = 25.0, yield_ref = 0.0, join = 20.0, mid = -1.0;
  int connector = 0;
  std::vector<SmpcDecisionNode::RxSpec> specs;
  std::vector<SmpcDecisionNode::RxBranch> br;
  while (in >> key) {
    if (key == "ego_speed") {
      in >> node.ego_speed_mps_;
    } else if (key == "mission") {
      in >> mission;
    } else if (key == "yield_ref") {
      in >> yield_ref;
    } else if (key == "targets") {
      in >> t.current_lane_id >> t.target_lane_id >> t.ego_d_target >> t.distance_to_lane_end;
    } else if (key == "flags") {      // lane_change_required prepare urgent emergency_stop_required (0/1)
      int a = 0, b = 0, c = 0, d = 0;
      in >> a >> b >> c >> d;
      t.lane_change_required = a != 0;
      t.lane_change_prepare = b != 0;
      t.lane_change_urgent = c != 0;
      t.emergency_stop_required = d != 0;
    } else if (key == "ego_xy") {
      in >> node.ego_x_ >> node.ego_y_;
    } else if (key == "mode") {
      in >> connector >> join >> mid;
    } else if (key == "vehicles") {
      int m = 0;
      in >> m;
      for (int i = 0; i < m; ++i) {
        smpc_lane_change::TargetVehicle v;
        in >> v.unique_id >> v.role >> v.lane_id >> v.s >> v.d >> v.delta_s >> v.v_long >> v.length >> v.width >>
            v.rear_delta_s >> v.front_delta_s >> v.accel;
        v.valid = true;
        if (v.role == "target_front") t.target_front = v;
        if (v.role == "target_rear") t.target_rear = v;
        if (v.role == "current_front") t.current_front = v;
        t.nearby_vehicles.push_back(v);
      }
    } else if (key == "specs") {
      int n = 0;
      in >> n;
      for (int i = 0; i < n; ++i) {
        SmpcDecisionNode::RxSpec s;
        int change = 0;
        std::string act;
        in >> change >> s.t_sw >> s.dur >> act >> s.prep_a;
        s.change = change != 0;
        if (!s.change) s.t_sw = std::numeric_limits<double>::infinity();
        if (act == "hold") s.action = SmpcDecisionNode::BehaviorAction::kHoldThenChange;
        else if (act == "yield") s.action = SmpcDecisionNode::BehaviorAction::kYieldDecelThenChange;
        else if (act == "accel") s.action = SmpcDecisionNode::BehaviorAction::kAccelThenChange;
        else if (act == "change_now") s.action = SmpcDecisionNode::BehaviorAction::kChangeNow;
        else s.action = SmpcDecisionNode::BehaviorAction::kKeep;
        specs.push_back(s);
      }
    } else if (key == "branches") {
      int b = 0;
      in >> b;
      for (int i = 0; i < b; ++i) {
        SmpcDecisionNode::RxBranch x;
        in >> x.rear >> x.front >> x.tb >> x.p;
        if (x.tb == "-") x.tb.clear();
        br.push_back(x);
      }
    }
  }
  auto r = node.rxRollout(t, specs, br, node.ego_speed_mps_, mission, connector != 0, join, yield_ref, mid);
  std::cout << std::setprecision(17) << "{";
  dump("valid", r.valid);
  dump("allowed", r.allowed);
  dump("risk", r.risk);
  dump("ego_s", r.ego_s);
  dump("ego_d", r.ego_d);
  dump("ego_v", r.ego_v);
  std::cout << "\"tv_s\":[";
  for (std::size_t i = 0; i < r.tv_s.size(); ++i) {
    std::cout << (i ? "," : "") << "[";
    for (std::size_t j = 0; j < r.tv_s[i].size(); ++j) std::cout << (j ? "," : "") << r.tv_s[i][j];
    std::cout << "]";
  }
  std::cout << "]}" << std::endl;
  return 0;
}
