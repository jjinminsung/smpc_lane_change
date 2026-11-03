#!/usr/bin/env python3
"""MORAI 기존 SMPC 가 왜 바꾸지 못했나: 목표 차선이 바로 옆(|d_t| ≤ 4.5 m)이고 바꾸기를 원한(desire=1) 틱에서
각 게이트가 막은 비율.

    python3 morai_reject_reasons.py "v2_B_S[34]*baseline*.bag"
/smpc/smpc_decision_debug 의 플래그로 다시 계산한다.
  G3 = 지금 간격(앞·뒤 간격 ≥ 요구 간격)과 TTC·헤드웨이 플래그
  G4 = 변경 궤적 위험 ≤ ε 이고 모든 mode clearance ≥ 0
  G5 = projected_gap_ttc_ok
한 틱에 여러 게이트가 같이 막을 수 있다. 'feasible=1' 은 노드가 직접 낸 판정이다.
"""
import bisect
import collections
import glob
import re
import sys
from pathlib import Path

import rosbag

BAGS = Path.home() / "bags" / "research"


def gates(d):
    gaps = re.search(r"gaps cur=\S+ tf=(\S+) tr=(\S+) req_gap tf=(\S+) tr=(\S+) eff_tr=(\S+)", d)
    ttc = re.search(r"ttc_ok tf=(\d) tf_hw=(\d) tr=(\d) rear_hw_ok=(\d)", d)
    g3 = (gaps is not None and ttc is not None and float(gaps.group(1)) >= float(gaps.group(3)) and
          float(gaps.group(2)) >= float(gaps.group(5)) and all(x == "1" for x in ttc.groups()))
    r = re.search(r"step_peak_keep=\S+ change=(\S+)", d)
    c = re.search(r"clear keep=\S+ change=(\S+)", d)
    e = re.search(r"risk_eps=(\S+)", d)
    g4 = r is not None and c is not None and e is not None and float(r.group(1)) <= float(e.group(1)) and \
        float(c.group(1)) >= 0.0
    return g3, g4, "projected_gap_ttc_ok=1" in d


def main():
    tot, n = collections.Counter(), 0
    for pat in sys.argv[1:] or ["v2_B_S[34]*baseline*.bag"]:
        for f in sorted(glob.glob(str(BAGS / pat))):
            tg, dbg = [], []
            with rosbag.Bag(f) as b:
                for top, m, t in b.read_messages(topics=["/smpc/smpc_decision_debug", "/smpc/targets"]):
                    if top == "/smpc/targets":
                        tg.append((t.to_sec(), m.ego_d_target))
                    elif " desire=1" in m.data:
                        dbg.append((t.to_sec(), m.data))
            ts = [x[0] for x in tg]
            for t, d in dbg:
                i = min(bisect.bisect_left(ts, t), len(tg) - 1)
                if abs(tg[i][1]) > 4.5:
                    continue
                g3, g4, g5 = gates(d)
                n += 1
                tot["G3 fail"] += not g3
                tot["G4 fail"] += not g4
                tot["G5 fail"] += not g5
                tot["G3-G5 any fail"] += not (g3 and g4 and g5)
                tot["G3-G5 all fail"] += not (g3 or g4 or g5)
                tot["node feasible=1"] += " feasible=1" in d
    print(f"desire ticks with target lane adjacent: {n}")
    for k, v in tot.items():
        print(f"  {k}: {v} ({100.0 * v / max(1, n):.1f} %)")


if __name__ == "__main__":
    main()
