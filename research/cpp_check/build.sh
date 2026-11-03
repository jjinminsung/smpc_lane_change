#!/usr/bin/env bash
# 하네스 빌드. catkin 패키지(CMakeLists)는 건드리지 않고 g++ 로 따로 만든다.
# 메시지 헤더는 ~/catkin_ws/devel/include 에서 가져온다(smpc_lane_change 빌드 후).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
WS_DEVEL="${WS_DEVEL:-$HOME/catkin_ws/devel}"
OUT="$HERE/bin"
mkdir -p "$OUT"
g++ -std=c++17 -O2 -w \
  -I"$REPO/include" -I"$WS_DEVEL/include" -I/opt/ros/noetic/include \
  "$HERE/${HARNESS:-risk_harness}.cpp" -o "$OUT/${HARNESS:-risk_harness}" \
  -L/opt/ros/noetic/lib -Wl,-rpath,/opt/ros/noetic/lib \
  -lroscpp -lrosconsole -lroscpp_serialization -lrostime -lxmlrpcpp -lcpp_common \
  -lboost_system -lpthread -lstdc++fs
echo "built $OUT/${HARNESS:-risk_harness}"
