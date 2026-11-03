#!/bin/bash
# v3 = 최종 바이너리(2026-11-04 02:28, 실행 가능 대응·plant 복귀·승인-모니터 일치·WAIT·abort 디바운스).
# 기존(baseline) 주행은 트리 변경의 영향을 받지 않아 v2_ 결과를 그대로 쓴다.
cd "$(dirname "$0")"
while kill -0 ${WAIT_PID:-0} 2>/dev/null; do sleep 5; done
L=~/bags/research/v3.log
echo "=== B tree_nogate $(date +%H:%M:%S)" >> $L
python3 run_matrix.py --phase B --methods tree_nogate --prefix v3_ >> $L 2>&1
echo "=== video $(date +%H:%M:%S)" >> $L
for m in baseline tree_nogate; do
  keep=true; meth=$m; [ $m = tree_nogate ] && { keep=false; meth=tree; }
  [ -f ~/bags/research/video/S2_${m}.mp4 ] || bash record_episode.sh research_S2_rear_accel video_S2_${m} $meth $keep >> $L 2>&1
done
echo "=== C r0 $(date +%H:%M:%S)" >> $L
python3 run_matrix.py --phase C --reps 1 --prefix v3_ >> $L 2>&1
echo "=== D r0 $(date +%H:%M:%S)" >> $L
python3 run_matrix.py --phase D --reps 1 --prefix v3_ >> $L 2>&1
echo "=== C r1 $(date +%H:%M:%S)" >> $L
python3 run_matrix.py --phase C --reps 2 --start_rep 1 --prefix v3_ >> $L 2>&1
echo "=== all done $(date +%H:%M:%S)" >> $L
