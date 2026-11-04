// MORAI bag 의 틱을 C++ 반응형 모니터(rxMonitorReturn)·판단(rxRollout)에 그대로 다시 넣어 본다.
// research/analysis/rx_replay_ticks.py 가 만든 틱 파일을 읽는다. 승인–모니터 불일치(plan_v2 단계 7) 원인 찾기용.
//
// 입력 (공백 구분, 틱마다):
//   tick <t> ego <x> <y> <yaw> <v> mission <v> blend <T>
//   conn <0|1>                          (waypoint 가 connector 를 따라가는 중인가 = 실행 방식)
//   targets <cur> <tgt> <ego_s_current> <ego_d_current> <ego_d_target> <lane_len> <dist_end> <time_to_end>
//   veh <slot> <uid> <lane> <x> <y> <s> <d> <delta_s> <yaw> <v_long> <v_lat> <length> <width>
//       <half_s> <half_d> <front_delta_s> <rear_delta_s> <hits> <accel> <role>      (slot: cf tf tr nb)
//   end
// 출력: 틱마다 한 줄 JSON {t, go, back, ret, vehicles, branch_go, peak_k, ego, tvs, acc_*, sel}
//   sel = treeSelectReactive 의 판단 문자열(미리보기 요구는 끈다: bag 에 미리보기가 없다)
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
  bool conn = false;
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
    } else if (key == "conn") {
      int c = 0;
      in >> c;
      conn = c != 0;
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
      const bool ret = node.rxMonitorReturn(t, tick_t, &detail, conn);
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
      auto r = node.rxRollout(t, {go}, br, node.ego_speed_mps_, node.currentTargetSpeedMps(), conn, 20.0,
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
      std::cout << "]";
      // 승인 평가: treeSelectReactive 의 change_now 루트(실행 전, 실행 지연 포함)
      SmpcDecisionNode::ActionSpec cn;
      cn.action = SmpcDecisionNode::BehaviorAction::kChangeNow;
      cn.lane_change_start_sec = 0.0;
      cn.lane_change_duration_sec = node.egoChangeBlendDuration();
      cn.change_left = true;
      cn.name = "change_now";
      const auto abr = node.rxBranches(t, -1.0);
      auto ar = node.rxRollout(t, {node.rxSpecOf(cn)}, abr, node.ego_speed_mps_, node.currentTargetSpeedMps(),
                               node.rxConnectorNow(t), 20.0, node.ego_speed_mps_, -1.0);
      const auto ts = node.rxSolveRoot(ar, abr, 0, 0.0, node.activeRiskEpsilon(t));
      std::cout << ",\"acc_feasible\":" << ts.feasible << ",\"acc_peak\":" << ts.peak_risk << ",\"acc_eps\":"
                << node.activeRiskEpsilon(t) << ",\"acc_branch\":[";
      for (int b = 0; b < ar.B; ++b) {
        double pk0 = 0.0, pk1 = 0.0;
        for (int k = 0; k < ar.K; ++k) {
          pk0 = std::max(pk0, ar.at(b, 0, 0, k));
          pk1 = std::max(pk1, ar.at(b, 1, 0, k));
        }
        std::cout << (b ? "," : "") << "[\"" << abr[static_cast<std::size_t>(b)].rear << "/" << abr[static_cast<std::size_t>(b)].front
                  << "\"," << abr[static_cast<std::size_t>(b)].p << "," << pk0 << "," << pk1 << ","
                  << static_cast<int>(ar.allowed[static_cast<std::size_t>((b * 2 + 1) * ar.N)]) << "]";
      }
      const auto atvs = node.rxVehicles(t);
      std::cout << "],\"acc_ego\":[";
      for (int k = 0; k < ar.K; k += 2)
        std::cout << (k ? "," : "") << "[" << ar.ego_s[static_cast<std::size_t>(k)] << "," << ar.ego_d[static_cast<std::size_t>(k)] << ","
                  << ar.ego_v[static_cast<std::size_t>(k)] << "]";
      std::cout << "],\"acc_tvs\":[";
      for (std::size_t i = 0; i < atvs.size(); ++i) {
        std::cout << (i ? "," : "") << "[";
        for (int k = 0; k < ar.K; k += 2) std::cout << (k ? "," : "") << ar.tv_s[i][static_cast<std::size_t>(k)];
        std::cout << "]";
      }
      std::cout << "]";
      // 판단 전체(모든 루트)
      node.ego_reference_preview_required_for_change_ = false;
      const double keep_intr = node.rx_intr_since_, keep_p = node.rx_p_rear_post_;
      const bool keep_conn = node.rx_lc_connector_;
      bool rc = false, hf = false;
      SmpcDecisionNode::CandidateEval bf;
      std::string sel;
      node.treeSelectReactive(t, cn, true, 0.0, node.activeRiskEpsilon(t), &rc, &bf, &hf, &sel);
      node.rx_intr_since_ = keep_intr;
      node.rx_p_rear_post_ = keep_p;
      node.rx_lc_connector_ = keep_conn;
      std::cout << ",\"sel\":\"" << sel << "\"";
      // 루트별: 이름, 유효, 실행 가능, 최대 위험, 지평선 끝 자차 [s, d, v]
      if (std::getenv("RX_ROOTS")) {
        std::vector<SmpcDecisionNode::ActionSpec> specs{cn};
        for (const auto& c : node.buildBehaviorCandidates(t))
          if (c.action != SmpcDecisionNode::BehaviorAction::kChangeNow && node.candidateFitsHorizon(c)) specs.push_back(c);
        std::vector<SmpcDecisionNode::RxSpec> rs;
        for (const auto& c : specs) rs.push_back(node.rxSpecOf(c));
        auto rr = node.rxRollout(t, rs, abr, node.ego_speed_mps_, node.currentTargetSpeedMps(), node.rxConnectorForAccept(t),
                                       20.0, node.ego_speed_mps_, -1.0);
        std::cout << ",\"roots\":[";
        for (std::size_t i = 0; i < specs.size(); ++i) {
          const auto ts2 = node.rxSolveRoot(rr, abr, static_cast<int>(i), 0.0, node.activeRiskEpsilon(t));
          const std::size_t m0 = (0 * 2 + 0) * rr.N + i;  // 가지 0, 계속
          const std::size_t last = m0 * rr.K + rr.K - 1;
          std::cout << (i ? "," : "") << "[\"" << specs[i].name << "\"," << static_cast<int>(rr.valid[i]) << "," << ts2.feasible << ","
                    << ts2.peak_risk << "," << specs[i].lane_change_start_sec << "," << specs[i].prep_accel_mps2 << ",["
                    << rr.ego_s[last] << "," << rr.ego_d[last] << "," << rr.ego_v[last] << "]]";
        }
        std::cout << "]";
      }
      // 가지 하나의 매 스텝(계속): 모니터·승인 롤아웃 각각 [k, 위험, 자차 s, d, v, 차량별 s...]
      if (const char* bs = std::getenv("RX_BRANCH")) {
        const int b = std::atoi(bs);
        auto dump = [&](const char* name, SmpcDecisionNode::RxResult& rr) {
          if (b >= rr.B) return;
          const std::size_t row = static_cast<std::size_t>((b * 2 + 0) * rr.N + 0);
          std::cout << ",\"" << name << "\":[";
          for (int k = 0; k < rr.K; ++k) {
            const std::size_t i = row * rr.K + k;
            std::cout << (k ? "," : "") << "[" << k << "," << rr.at(b, 0, 0, k) << "," << rr.ego_s[i] << "," << rr.ego_d[i] << ","
                      << rr.ego_v[i];
            for (const auto& s : rr.tv_s) std::cout << "," << s[i];
            std::cout << "]";
          }
          std::cout << "]";
        };
        dump("mon_b", r);
        dump("acc_b", ar);
      }
      std::cout << "}" << std::endl;
    }
  }
  return 0;
}
