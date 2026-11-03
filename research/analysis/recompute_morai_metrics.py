#!/usr/bin/env python3
"""morai_runs.jsonl 의 metrics 를 현재 morai_metrics.py 로 다시 계산한다(원본은 .bak 로 보관)."""
import json
import shutil
from pathlib import Path

from morai_metrics import analyse

LOG = Path(__file__).resolve().parents[1] / "results" / "morai_runs.jsonl"
BAGS = Path.home() / "bags" / "research"
shutil.copy(LOG, LOG.with_suffix(".jsonl.bak"))
rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
for r in rows:
    bag = BAGS / f"{r['tag']}.bag"
    if bag.exists():
        r["metrics"] = analyse(bag)
        (BAGS / f"{r['tag']}.metrics.json").write_text(json.dumps(r["metrics"], indent=1))
LOG.write_text("".join(json.dumps(r) + "\n" for r in rows))
print("recomputed", sum(1 for r in rows if (BAGS / f"{r['tag']}.bag").exists()), "of", len(rows))
