#!/usr/bin/env python3
"""highway_spawn50 시나리오를 그대로 두고 NPC 스폰 속도·목표 속도만 바꾼 격자를 만든다.

    python3 make_scenarios.py            # research_H_s{스폰}_d{목표} 12 개 → MORAI SaveFile + scenarios/

자차 시작 위치, 스폰 지점·간격·최대 대수, 신호등은 highway_spawn50 과 같다. 바꾸는 값은 스폰 지점마다
  SpawnVelocityType = 2(Custom), Min/MaxSpawnVelocity_Custom = 스폰 속도
  DesiredVelocityType = 2(Custom), Min/MaxDesiredVelocity_Custom = 목표 속도   (highway_spawn50 은 1)
네트워크(EgoNetwork/_MN.json)는 highway_spawn50 의 rosbridge 설정이 아니라 연구 스택이 쓰는 UDP 설정
(research_S1_normal_MN.json, 대회 포트 + Object Info 7605→7505)을 복사한다.
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
MORAI = Path.home() / "Desktop/MoraiLauncher_Lin/MoraiLauncher_Lin_Data/SaveFile/Scenario/R_KR_PR_K-city_2025"
BASE = MORAI / "highway_spawn50 .json"          # 파일 이름에 공백이 있다
NET = MORAI / "EgoNetwork/research_S1_normal_MN.json"
SPAWN_KPH = (30, 50, 70)
DESIRED_KPH = (40, 60, 80, 100)


def main():
    base = json.loads(BASE.read_text())
    for s in SPAWN_KPH:
        for d in DESIRED_KPH:
            name = f"research_H_s{s}_d{d}"
            sc = json.loads(json.dumps(base))
            for sp in sc["spawnPointList"]:
                sp["SpawnVelocityType"] = 2
                sp["MinSpawnVelocity_Custom"] = sp["MaxSpawnVelocity_Custom"] = float(s)
                sp["DesiredVelocityType"] = 2
                sp["MinDesiredVelocity_Custom"] = sp["MaxDesiredVelocity_Custom"] = float(d)
            text = json.dumps(sc)
            for out in (MORAI / f"{name}.json", HERE / "scenarios" / f"{name}.json"):
                if out.exists():
                    raise SystemExit(f"이미 있음: {out}")
                out.write_text(text)
            shutil.copyfile(NET, MORAI / "EgoNetwork" / f"{name}_MN.json")
            print(name)


if __name__ == "__main__":
    main()
