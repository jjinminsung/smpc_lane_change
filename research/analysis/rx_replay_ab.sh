#!/usr/bin/env bash
# MORAI bag 틱을 C++ 판단(treeSelectReactive)에 다시 넣어 예측 수정 켬/끔 비교. 격리 마스터 11411.
#   replay_ab.sh <bag> <t_from> <t_to> <step> <out_prefix>
# 결과: <out_prefix>.{ticks, pf1fl1, pf0fl1, pf1fl0, pf0fl0}.jsonl
set -u
BAG=$1; T0=$2; T1=$3; STEP=$4; OUT=$5
export ROS_MASTER_URI=http://localhost:11411
source /opt/ros/noetic/setup.bash; source ~/catkin_ws/devel/setup.bash
if ! rosparam list >/dev/null 2>&1; then
  setsid roscore -p 11411 >/dev/null 2>&1 < /dev/null &
  echo $! > ${OUT%/*}/roscore11411.pid
  for i in $(seq 1 30); do rosparam list >/dev/null 2>&1 && break; sleep 0.5; done
fi
N=/smpc_rx_replay; C=~/github/smpc_lane_change/config
rosparam load $C/smpc_decision.yaml $N && rosparam load $C/acc_planner.yaml $N/acc && rosparam load $C/reactive_morai.yaml $N/reactive
rosparam set $N/waypoint_directory ~/catkin_ws/src/waypoint_system/data/default
for p in tree_planner_enabled tree_monitor_enabled reactive_tree_enabled; do rosparam set $N/$p true; done
rosparam set $N/tree_keep_gates false
rosparam set $N/reactive/monitor_tree true
rosparam set $N/tree_abort_risk_factor 2.0
cd ~/bags/research && nice -n 19 python3 ~/github/smpc_lane_change/research/analysis/rx_replay_ticks.py $BAG $T0 $T1 --all $STEP > $OUT.ticks 2> $OUT.rec
for pf in 1 0; do for fl in 1 0; do
  rosparam set $N/reactive/passed_front_as_rear $([ $pf = 1 ] && echo true || echo false)
  rosparam set $N/reactive/merge_floor $([ $fl = 1 ] && echo true || echo false)
  nice -n 19 ~/github/smpc_lane_change/research/cpp_check/bin/rx_replay $OUT.ticks __name:=smpc_rx_replay 2>/dev/null | grep '^{' > $OUT.pf${pf}fl${fl}.jsonl
done; done
wc -l $OUT.*.jsonl
