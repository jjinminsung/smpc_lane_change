// MORAI bag 의 틱을 C++ 반응형 모니터(rxMonitorReturn)·판단(rxRollout)에 그대로 다시 넣어 본다.
// research/analysis/rx_replay_ticks.py 가 만든 틱 파일을 읽는다. 승인–모니터 불일치(plan_v2 단계 7) 원인 찾기용.
//
// 입력 (공백 구분, 틱마다):
//   tick <t> ego <x> <y> <yaw> <v> mission <v> blend <T>
//   targets <cur> <tgt> <ego_s_current> <ego_d_current> <ego_d_target> <lane_len> <dist_end> <time_to_end>
//   veh <slot> <uid> <lane> <x> <y> <s> <d> <delta_s> <yaw> <v_long> <v_lat> <length> <width>
//       <half_s> <half_d> <front_delta_s> <rear_delta_s> <hits> <accel> <role>      (slot: cf tf tr nb)
//   end
// 출력: 틱마다 한 줄 JSON {t, go, back, ret, vehicles, branch_go, peak_k, ego, tvs}
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

int main(int argc, char** argv) {
  ros::init(argc, argv, "smpc_rx_replay");
  if (argc < 2) return 2;
  SmpcDecisionNode node;
  if (node.lanes_.empty()) {
    std::cerr << "no lanes\n";
    return 3;
  }
  std::ifstream in(argv[1]);
  std::string key;
  smpc_lane_change::TargetVehicleSet t;
  double tick_t = 0.0;
  std::cout << std::setprecision(6);
  while (in >> key) {
    if (key == "tick") {
      t = smpc_lane_change::TargetVehicleSet();
      double mission = 0.0, blend = 0.0;
      std::string k;
      in >> tick_t >> k >> node.ego_x_ >> node.ego_y_ >> node.ego_yaw_ >> node.ego_speed_mps_ >> k >> mission >> k >> blend;
      node.mission_speed_target_mps_ = mission;
      node.have_mission_speed_target_ = true;
      node.mission_speed_stamp_ = ros::Time::now();
      node.reference_preview_ = SmpcDecisionNode::ReferencePreview();
      node.reference_preview_.valid = true;
      node.reference_preview_.mode = waypoint_system::PathSwitchPreview::BLEND;
      node.reference_preview_.blend_duration_sec = blend;
      node.have_reference_preview_ = true;
    } else if (key == "targets") {
      in >> t.current_lane_id >> t.target_lane_id >> t.ego_s_current >> t.ego_d_current >> t.ego_d_target >>
          t.current_lane_length >> t.distance_to_lane_end >> t.time_to_lane_end;
    } else if (key == "veh") {
      std::string slot;
      smpc_lane_change::TargetVehicle v;
      in >> slot >> v.unique_id >> v.lane_id >> v.x >> v.y >> v.s >> v.d >> v.delta_s >> v.yaw >> v.v_long >> v.v_lat >>
          v.length >> v.width >> v.half_s >> v.half_d >> v.front_delta_s >> v.rear_delta_s >> v.track_hits >> v.accel >> v.role;
      v.valid = true;
      if (slot == "cf") t.current_front = v;
      else if (slot == "tf") t.target_front = v;
      else if (slot == "tr") t.target_rear = v;
      else t.nearby_vehicles.push_back(v);
    } else if (key == "end") {
      std::string detail;
      const bool ret = node.rxMonitorReturn(t, tick_t, &detail);
      // 같은 모니터 롤아웃을 다시 굴려 가지별 위험과 자차·차량 경로를 낸다.
      const double W = node.lane_width_m_;
      const double d_t = std::abs(t.ego_d_target);
      const double T_rem = std::max(0.5, node.egoChangeBlendDuration() * d_t / W);
      auto br = node.rxBranches(t, node.rx_p_rear_post_);
      for (auto& b : br) b.tb = "now";
      SmpcDecisionNode::RxSpec go;
      go.change = true;
      go.t_sw = 0.0;
      go.dur = T_rem;
      go.action = SmpcDecisionNode::BehaviorAction::kChangeNow;
      auto r = node.rxRollout(t, {go}, br, node.ego_speed_mps_, node.currentTargetSpeedMps(), node.rxConnectorNow(t), 20.0,
                              node.ego_speed_mps_, T_rem);
      const auto tvs = node.rxVehicles(t);
      int peak_k = 0;
      double peak = -1.0;
      for (int k = 0; k < r.K; ++k) {
        double g = 0.0;
        for (int b = 0; b < r.B; ++b) g += br[static_cast<std::size_t>(b)].p * r.at(b, 0, 0, k);
        if (g > peak) {
          peak = g;
          peak_k = k;
        }
      }
      const auto* tl = node.laneFor(t.target_lane_id);
      const auto ep = tl->projectExtended(node.ego_x_, node.ego_y_, node.tv_lane_projection_extension_before_m_, 0.0);
      std::cout << "{\"tgt_len\":" << tl->length() << ",\"ego_proj_s\":" << ep.s << ",\"tgt_end\":" << tl->length() - ep.s << ",";
      std::cout << "\"t\":" << tick_t << ",\"detail\":\"" << detail << "\",\"ret\":" << ret << ",\"T_rem\":" << T_rem
                << ",\"vehicles\":[";
      for (std::size_t i = 0; i < tvs.size(); ++i)
        std::cout << (i ? "," : "") << "[" << tvs[i].v.unique_id << ",\"" << tvs[i].role << "\"," << tvs[i].v.delta_s << "]";
      std::cout << "],\"branch_go\":[";
      for (int b = 0; b < r.B; ++b) {
        double pk = 0.0;
        for (int k = 0; k < r.K; ++k) pk = std::max(pk, r.at(b, 0, 0, k));
        std::cout << (b ? "," : "") << "[\"" << br[static_cast<std::size_t>(b)].rear << "/" << br[static_cast<std::size_t>(b)].front
                  << "\"," << br[static_cast<std::size_t>(b)].p << "," << pk << "]";
      }
      std::cout << "],\"peak_k\":" << peak_k << ",\"ego\":[";
      for (int k = 0; k < r.K; k += 5)
        std::cout << (k ? "," : "") << "[" << r.ego_s[static_cast<std::size_t>(k)] << "," << r.ego_d[static_cast<std::size_t>(k)] << ","
                  << r.ego_v[static_cast<std::size_t>(k)] << "]";
      std::cout << "],\"tvs\":[";
      for (std::size_t i = 0; i < tvs.size(); ++i) {
        std::cout << (i ? "," : "") << "[";
        for (int k = 0; k < r.K; k += 5) std::cout << (k ? "," : "") << r.tv_s[i][static_cast<std::size_t>(k)];
        std::cout << "]";
      }
      std::cout << "]}" << std::endl;
    }
  }
  return 0;
}
