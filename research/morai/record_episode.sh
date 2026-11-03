#!/bin/bash
# 시나리오 로드 → 녹화 시작 → 주행 한 번 → 녹화 종료.  결과: $OUT/<tag>.mp4
#   record_episode.sh <scenario> <tag> <method: baseline|tree> <keep_gates: true|false>
set -u
SC=$1; TAG=$2; METHOD=$3; KEEP=$4
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$HOME/bags/research/video}
mkdir -p "$OUT"
export DISPLAY=:1
python3 "$HERE/load_scenario.py" "$SC" || exit 2
WIN=$(xwininfo -root -tree | grep -o '0x[0-9a-f]* "Simulator"' | head -1 | cut -d' ' -f1)
set -- $(xwininfo -id "$WIN" | awk '/Absolute upper-left X/{x=$4}/Absolute upper-left Y/{y=$4}/Width/{w=$2}/Height/{h=$2}END{print x,y,w,h}')
W=$(( $3 / 2 * 2 )); H=$(( $4 / 2 * 2 ))
# NVENC: ffmpeg 4.2 는 -preset p4 를 못 받는다(캡스톤에서 확인). fast + vbr/cq.
ffmpeg -y -loglevel error -f x11grab -framerate 30 -video_size ${W}x${H} -i :1+$1,$2 \
  -vf scale=1150:-2 -c:v h264_nvenc -preset fast -rc vbr -cq 28 -b:v 0 "$OUT/${TAG}.mp4" &
FP=$!
OUT_BAG=${OUT_BAG:-$HOME/bags/research}
OUT=$OUT_BAG bash "$HERE/run_episode.sh" "$TAG" "$METHOD" "$KEEP"
RC=$?
kill -INT $FP 2>/dev/null
for i in $(seq 1 20); do kill -0 $FP 2>/dev/null || break; sleep 0.5; done
ls -la "$OUT/${TAG}.mp4"
exit $RC
