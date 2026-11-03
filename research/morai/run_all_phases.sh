#!/bin/bash
# A(S2 안전 확인) → B(문서 계획 24) → D(트리 게이트 유지 비교) → C(속도 격자 48). 각 단계 로그는 ~/bags/research.
cd "$(dirname "$0")"
for ph in A B D C; do
  echo "=== phase $ph $(date +%H:%M:%S)"
  python3 run_matrix.py --phase $ph --prefix v2_ >> ~/bags/research/v2_phase$ph.log 2>&1
done
echo "=== all done $(date +%H:%M:%S)"
