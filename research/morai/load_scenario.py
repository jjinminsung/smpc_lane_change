#!/usr/bin/env python3
"""MORAI 시나리오를 이름으로 불러오고 Ego 네트워크 연결을 확인한다.

    python3 load_scenario.py research_S1_normal

좌표는 Simulator 창 2299x2096 기준(2026-11-04 화면에서 확인). 각 단계는 화면 색으로 확인하고,
실패하면 0 이 아닌 코드로 끝난다. Load Scenario 대화상자의 검색창에 이름을 넣어 한 줄만 남긴다.
Network Data 는 켜 둔다(시나리오의 _MN.json = 1020 UDP 대회 포트로 다시 연결된다).
"""
import os
import re
import subprocess
import sys
import time

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SCR = os.environ.get("MORAI_SHOT_DIR", "/tmp")
ENV = dict(os.environ, DISPLAY=":1")
X = ["python3", os.path.join(HERE, "xctl.py")]


def win():
    out = subprocess.run(["xwininfo", "-root", "-tree"], check=True, capture_output=True, text=True, env=ENV).stdout
    m = re.search(r'(0x[0-9a-f]+) "Simulator"', out)
    if not m:
        sys.exit("MORAI Simulator window not found")
    return m.group(1)


def run(*a):
    subprocess.run(X + [str(x) for x in a], check=True, env=ENV, stdout=subprocess.DEVNULL)


def shot(w, name):
    p = os.path.join(SCR, name)
    subprocess.run(X + ["shot", w, p], check=True, env=ENV)
    return Image.open(p).convert("RGB")


def px(im, x, y, r=3):
    pts = [im.getpixel((x + dx, y + dy)) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    return tuple(sum(p[i] for p in pts) // len(pts) for i in range(3))


def blue(p):
    return p[2] > 150 and p[2] - p[0] > 30


def green(p):
    return p[1] > 90 and p[1] > p[0] + 50


def main():
    name = sys.argv[1]
    w = win()
    run("resize", w, 10, 45, 2299, 2096)
    time.sleep(1.0)
    run("activate", w)
    time.sleep(0.5)
    for attempt in range(4):
        run("wclick", w, 52, 17)            # Edit
        time.sleep(0.8)
        run("wclick", w, 90, 154)           # Scenario ▸
        time.sleep(0.8)
        run("wclick", w, 455, 221)          # Load Scenario
        time.sleep(1.5)
        im = shot(w, "ls_dialog.png")
        if blue(px(im, 1130, 886, 2)) or blue(px(im, 1224, 1394, 2)) or px(im, 1224, 1394, 2)[2] > 80:
            break
        run("wclick", w, 1200, 17)
        time.sleep(0.6)
    else:
        sys.exit(2)
    run("wclick", w, 1100, 754)             # 검색창
    time.sleep(0.3)
    run("key", "a", "Control_L")
    run("key", "BackSpace")
    run("type", name)
    time.sleep(1.0)
    run("wclick", w, 1130, 886)             # 첫 줄(검색 결과)
    time.sleep(0.4)
    im = shot(w, "ls_dialog2.png")
    if not blue(px(im, 1130, 886, 2)):
        print("search result row not selected", px(im, 1130, 886, 2))
        sys.exit(3)
    if blue(px(im, 1144, 1326, 3)):        # Pause Mode 끄기
        run("wclick", w, 1144, 1326)
        time.sleep(0.4)
    if not blue(px(im, 906, 1326, 3)):     # Network Data 켜기
        run("wclick", w, 906, 1326)
        time.sleep(0.4)
    im = shot(w, "ls_dialog3.png")
    if blue(px(im, 1144, 1326, 3)) or not blue(px(im, 906, 1326, 3)):
        print("checkbox state wrong")
        sys.exit(4)
    run("wclick", w, 1224, 1394)            # Load
    time.sleep(9)
    run("wclick", w, 1072, 1394)            # 남은 대화상자 Cancel
    time.sleep(0.8)
    run("key", "F4")
    time.sleep(1.5)
    im = shot(w, "ls_net.png")
    ok = green(px(im, 976, 846, 2))
    if not ok:
        run("wclick", w, 1490, 1374)        # Connect
        time.sleep(2)
        im = shot(w, "ls_net2.png")
        ok = green(px(im, 976, 846, 2))
    run("wclick", w, 1606, 708)             # 네트워크 창 닫기
    time.sleep(0.8)
    print("loaded %s, network %s" % (name, "connected" if ok else "DISCONNECTED"))
    sys.exit(0 if ok else 5)


if __name__ == "__main__":
    main()
