#!/usr/bin/env python3
"""MORAI 기존 SMPC 가 왜 바꾸지 못했나: 바꾸기를 원한 틱(desire=1)에서 각 게이트가 막은 비율.

    python3 morai_reject_reasons.py "v2_B_S3*baseline*.bag" "v2_B_S4*baseline*.bag"
/smpc/smpc_decision_debug 의 플래그를 센다: G3 = 지금 간격·TTC(ttc_ok / rear_hw_ok / gaps 대 req_gap),
G4 = 위험 > ε 또는 clearance < 0, G5 = projected_gap_ttc_ok. 한 틱에 여러 게이트가 같이 막을 수 있다.
"""
import glob
import re
import sys
from pathlib import Path

import rosbag

BAGS = Path.home() / "bags" / "research"


def flag(d, key):
    m = re.search(key + r"=([-0-9.e+inf]+)", d)
    return float(m.group(1)) if m else None


def main():
    for pat in sys.argv[1:]:
        for f in sorted(glob.glob(str(BAGS / pat))):
            n = g3 = g4 = g5 = only_g4 = 0
            with rosbag.Bag(f) as b:
                for _, m, _ in b.read_messages(topics=["/smpc/smpc_decision_debug"]):
                    d = m.data
                    if " desire=1" not in d:
                        continue
                    n += 1
                    gaps = re.search(r"gaps cur=\S+ tf=(\S+) tr=(\S+) req_gap tf=(\S+) tr=(\S+) eff_tr=(\S+)", d)
                    ttc = re.search(r"ttc_ok tf=(\d) tf_hw=(\d) tr=(\d) rear_hw_ok=(\d)", d)
                    gap_ok = gaps is not None and float(gaps.group(1)) >= float(gaps.group(3)) and \
                        float(gaps.group(2)) >= float(gaps.group(5))
                    det_ok = gap_ok and ttc is not None and all(x == "1" for x in ttc.groups())
                    risk = re.search(r"step_peak_keep=\S+ change=(\S+)", d)
                    clear = re.search(r"clear keep=\S+ change=(\S+)", d)
                    eps = flag(d, "risk_eps")
                    g4_ok = risk is not None and clear is not None and eps is not None and \
                        float(risk.group(1)) <= eps and float(clear.group(1)) >= 0.0
                    g5_ok = flag(d, "projected_gap_ttc_ok") == 1
                    g3 += not det_ok
                    g4 += not g4_ok
                    g5 += not g5_ok
                    only_g4 += det_ok and g5_ok and not g4_ok
            if n:
                print(f"{Path(f).name}: desire ticks {n}, G3 fail {100 * g3 / n:.0f} %, G4 fail {100 * g4 / n:.0f} %, "
                      f"G5 fail {100 * g5 / n:.0f} %, G3·G5 pass but G4 fail {100 * only_g4 / n:.0f} %")


if __name__ == "__main__":
    main()
