#!/usr/bin/env python3
"""MORAI 1020 시나리오 다시 불러오기 + Ego 네트워크 Connect.
화면 색으로 상태를 확인하고 필요한 것만 누른다 (Connect 는 누를 때마다 켜고 끄기가 바뀐다).
좌표는 Simulator 창(2299x2160) 기준이며 2026-10-29 화면에서 확인했다."""
import os
import re
import subprocess
import sys
import time

from PIL import Image

S = os.path.dirname(os.path.abspath(__file__))
# MORAI 를 다시 켜면 창 ID 가 바뀐다
_m = re.search(r'(0x[0-9a-f]+) "Simulator"', subprocess.run(
    ['xwininfo', '-root', '-tree'], check=True, capture_output=True, text=True,
    env=dict(os.environ, DISPLAY=':1')).stdout)
if not _m:
    sys.exit('MORAI Simulator window not found: start the simulator first')
WIN = _m.group(1)
X = ['python3', os.path.join(S, 'xctl.py')]
env = dict(os.environ, DISPLAY=':1')


def run(*a):
    subprocess.run(X + [str(x) for x in a], check=True, env=env, stdout=subprocess.DEVNULL)


def shot(name):
    p = os.path.join(S, name)
    subprocess.run(X + ['shot', WIN, p], check=True, env=env)
    return Image.open(p).convert('RGB')


def is_blue(px):
    r, g, b = px
    return b > 150 and b - r > 30


def mean_px(im, x, y, r=3):
    pts = [im.getpixel((x + dx, y + dy)) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
    return tuple(sum(p[i] for p in pts) // len(pts) for i in range(3))


def win_dy():
    """가운데 정렬된 대화상자 좌표는 창 높이 2160 기준이다. 창 높이가 다르면 그 절반만큼 옮긴다."""
    out = subprocess.run(X + ['geom', WIN], check=True, env=env, capture_output=True, text=True).stdout.split()
    return (int(out[3]) - 2160) // 2


def main():
    # 막 켠 MORAI 는 800x800 이다. 아래 좌표는 2299 폭 기준이라 크기부터 맞춘다
    run('resize', WIN, 101, 64, 2299, 2096)
    time.sleep(2)
    dy = win_dy()
    print('window height offset dy=%d' % dy)
    run('activate', WIN)
    time.sleep(0.8)
    # Edit -> (Scenario 하위 메뉴가 펼쳐진 채로 열린다) -> Load Scenario -> 1020.json. 실패하면 Esc 후 다시.
    for attempt in range(4):
        run('wclick', WIN, 52, 17)
        time.sleep(1.0)
        if attempt % 2 == 0:
            run('wclick', WIN, 455, 221)  # Scenario 하위 메뉴가 펼쳐져 있을 때: Load Scenario
        else:
            run('wclick', WIN, 90, 154)   # MORAI 를 막 켰을 때: Scenario 를 누르면 불러오기 창이 열린다
        time.sleep(1.5)
        run('wclick', WIN, 956, 956 + dy)
        time.sleep(0.6)
        im = shot('rs_dialog2.png')
        sel = mean_px(im, 1000, 956 + dy, 2)
        if is_blue(sel):
            break
        print('attempt %d: load dialog not open (row color %s), retry' % (attempt + 1, sel))
        run('wclick', WIN, 1200, 17)  # 메뉴 닫기 (ESC 는 일시정지를 켠다)
        run('wclick', WIN, 1200, 17)
        time.sleep(0.8)
    else:
        sys.exit(2)
    # Network Data, Pause Mode 는 해제 상태여야 한다
    for (x, y, name) in ((906, 1358 + dy, 'Network Data'), (1144, 1358 + dy, 'Pause Mode')):
        if is_blue(mean_px(im, x, y, 3)):
            run('wclick', WIN, x, y)
            time.sleep(0.4)
            print('unchecked %s' % name)
    im = shot('rs_dialog3.png')
    for (x, y, name) in ((906, 1358 + dy, 'Network Data'), (1144, 1358 + dy, 'Pause Mode')):
        if is_blue(mean_px(im, x, y, 3)):
            print('%s still checked, abort' % name)
            sys.exit(3)
    # Load
    run('wclick', WIN, 1225, 1428 + dy)
    print('scenario load clicked')
    time.sleep(7)
    # 남아 있는 대화상자 닫기 (Cancel)
    run('wclick', WIN, 1072, 1426 + dy)
    time.sleep(0.8)
    # F4 네트워크 설정
    run('key', 'F4')
    time.sleep(1.5)
    im = shot('rs_net.png')
    dot = mean_px(im, 978, 876 + dy, 2)
    connected = dot[1] > 90 and dot[1] > dot[0] + 50
    print('ego network status dot %s -> %s' % (dot, 'connected' if connected else 'disconnected'))
    if not connected:
        run('wclick', WIN, 1490, 1406 + dy)
        time.sleep(2)
        im = shot('rs_net2.png')
        dot = mean_px(im, 978, 876 + dy, 2)
        connected = dot[1] > 90 and dot[1] > dot[0] + 50
        print('after Connect: %s -> %s' % (dot, 'connected' if connected else 'STILL disconnected'))
    # 창 닫기
    run('wclick', WIN, 1608, 740 + dy)
    time.sleep(0.8)
    shot('rs_done.png')
    sys.exit(0 if connected else 4)


if __name__ == '__main__':
    main()
