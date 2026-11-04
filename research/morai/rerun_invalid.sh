#!/bin/bash
# run_v3.sh 가 끝난 뒤: episode_monitor 거짓 done 으로 잘린 주행 기록을 morai_runs_invalid.jsonl 로 옮기고 다시 돌린다.
cd "$(dirname "$0")"
while ps aux | grep -q "[r]un_v3.sh"; do sleep 20; done
LOG=../results/morai_runs.jsonl
python3 - <<'PY'
import json
from pathlib import Path
log = Path("../results/morai_runs.jsonl"); bad = Path("../results/morai_runs_invalid.jsonl")
keep, moved = [], []
for line in log.read_text().splitlines():
    if not line.strip():
        continue
    r = json.loads(line)
    mon, m = r.get("monitor") or {}, r.get("metrics") or {}
    (moved if (r["tag"].startswith("v3_") and mon.get("reason") == "done" and not m.get("success")) else keep).append(line)
log.write_text("\n".join(keep) + "\n")
with bad.open("a") as f:
    for line in moved:
        f.write(line + "\n")
print("moved", [json.loads(l)["tag"] for l in moved])
PY
echo "=== rerun invalid $(date +%H:%M:%S)" >> ~/bags/research/v3.log
python3 run_matrix.py --phase C --reps 2 --prefix v3_ >> ~/bags/research/v3.log 2>&1
echo "=== rerun done $(date +%H:%M:%S)" >> ~/bags/research/v3.log
