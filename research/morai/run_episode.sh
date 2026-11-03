#!/bin/bash
# MORAI 연구 주행 한 번: (시나리오는 미리 로드) 스택 기동 → 녹화 → 종료 판단 → 정리.
#   run_episode.sh <tag> <method: baseline|tree> <keep_gates: true|false> [extra roslaunch args...]
# 시나리오 로드는 load_scenario.py 가 따로 한다(GUI). 프로세스는 PID 로만 내린다(pkill -f 금지).
set -u
TAG=$1; METHOD=$2; KEEP=$3; shift 3
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/bags/research}
mkdir -p "$OUT"
source /opt/ros/noetic/setup.bash
source $HOME/catkin_ws/devel/setup.bash
LOG=$OUT/${TAG}.launch.log
roslaunch smpc_lane_change research_morai.launch method:=$METHOD tree_keep_gates:=$KEEP "$@" > "$LOG" 2>&1 &
LP=$!
# 마스터가 뜰 때까지
for i in $(seq 1 30); do rostopic list > /dev/null 2>&1 && break; sleep 1; done
rosbag record -O "$OUT/${TAG}.bag" --lz4 \
  /Object_topic /Ego_topic /odom /gps_state /CollisionData /Competition_topic /ctrl_cmd \
  /path_number /path_switch_connector_active \
  /smpc/targets /smpc/decision /smpc/lane_change_status /smpc/smpc_decision_debug \
  /smpc/lane_change_supervisor_debug /smpc/behavior_plan /smpc/behavior_longitudinal_request \
  /smpc/behavior_longitudinal_status /smpc/target_speed_mps /smpc/target_speed_mps_gated \
  /smpc/acc_active_lead /smpc/tree_debug /smpc/tree_abort /mission/active_section \
  /control_feedback/actual_speed_mps /control_feedback/mission_speed_target_mps \
  __name:=research_record > "$OUT/${TAG}.record.log" 2>&1 &
RP=$!
python3 "$HERE/episode_monitor.py" > "$OUT/${TAG}.monitor.json" 2> "$OUT/${TAG}.monitor.err"
RC=$?
kill -INT $RP 2>/dev/null
for i in $(seq 1 20); do kill -0 $RP 2>/dev/null || break; sleep 0.5; done
kill -INT $LP 2>/dev/null
for i in $(seq 1 40); do kill -0 $LP 2>/dev/null || break; sleep 1; done
kill -0 $LP 2>/dev/null && { kill -TERM $LP; sleep 5; }
# 남은 ROS 프로세스 확인(이 스크립트가 띄운 것만)
ps -eo pid,args | grep -E "/opt/ros/noetic/bin/(rosmaster|roslaunch)" | grep -v -E "grep|-p 11411" && echo "WARN: ROS still up"
tail -1 "$OUT/${TAG}.monitor.json"
exit $RC
