#!/usr/bin/env python3
"""MORAI 주행 영상 녹화: (시나리오, 방법) 목록을 차례로 불러와 MORAI 창을 녹화하면서 한 번씩 주행한다.

    python3 record_videos.py --out ~/bags/research/video_1106 \
        H_s30_d40:rx_fix H_s30_d40:baseline S3_rear_decel:rx_fix ...

한 회: load_scenario.py → MORAI 항상 위·알림 끄기·포인터를 창 밖으로 → ffmpeg x11grab(원본 mkv, 시작 시각 기록)
→ run_episode.sh(스택 + rosbag + episode_monitor) → 녹화 종료·원상복구 → morai_metrics.py.
결과: <out>/<tag>_raw.mkv, <tag>.video_t0(녹화 시작 epoch), <tag>.bag, <out>/vid_runs.jsonl.
자막 입히기는 make_video.py.

사람이 마우스를 쓰는 중이면(5 s 동안 포인터 5 px 넘게 움직임) 시작하지 않고 기다린다. 프로세스는 PID 로만 내린다.
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from run_matrix import METHODS, run

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
X = ["python3", str(HERE / "xctl.py")]
ENV = dict(os.environ, DISPLAY=":1")


def pointer_busy(secs=5.0):
    x11 = ctypes.cdll.LoadLibrary("libX11.so.6")
    x11.XOpenDisplay.restype = ctypes.c_void_p
    d = x11.XOpenDisplay(b":1")
    x11.XDefaultRootWindow.restype = ctypes.c_ulong
    x11.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
    root = x11.XDefaultRootWindow(d)
    x11.XQueryPointer.argtypes = [ctypes.c_void_p, ctypes.c_ulong] + [ctypes.POINTER(ctypes.c_ulong)] * 2 + \
        [ctypes.POINTER(ctypes.c_int)] * 4 + [ctypes.POINTER(ctypes.c_uint)]

    def pos():
        r, c = ctypes.c_ulong(), ctypes.c_ulong()
        a, b, e, f = ctypes.c_int(), ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
        m = ctypes.c_uint()
        x11.XQueryPointer(d, root, ctypes.byref(r), ctypes.byref(c), ctypes.byref(a), ctypes.byref(b),
                          ctypes.byref(e), ctypes.byref(f), ctypes.byref(m))
        return a.value, b.value

    p0, mx = pos(), 0
    for _ in range(int(secs / 0.25)):
        time.sleep(0.25)
        p = pos()
        mx = max(mx, abs(p[0] - p0[0]) + abs(p[1] - p0[1]))
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay(d)
    return mx > 5


def sim_window():
    out = subprocess.run(["xwininfo", "-root", "-tree"], capture_output=True, text=True, env=ENV).stdout
    for line in out.splitlines():
        if '"Simulator"' in line:
            return line.split()[0]
    return None


def xctl(*a):
    return subprocess.run(X + [str(v) for v in a], capture_output=True, text=True, env=ENV).stdout.strip()


def record_one(sc, m, out, prefix):
    tag = f"{prefix}{sc}_{m}"
    scen = f"research_{sc}"
    rc, so, se = run(["python3", str(HERE / "load_scenario.py"), scen], 180)
    if rc != 0:
        rc, so, se = run(["python3", str(HERE / "load_scenario.py"), scen], 180)
    if rc != 0:
        return {"tag": tag, "scenario": scen, "method": m, "error": "load_scenario failed", "out": (so + se)[-500:]}
    win = sim_window()
    banners = subprocess.run(["gsettings", "get", "org.gnome.desktop.notifications", "show-banners"],
                             capture_output=True, text=True).stdout.strip() or "true"
    subprocess.run(["gsettings", "set", "org.gnome.desktop.notifications", "show-banners", "false"])
    xctl("above", win, "on")
    xctl("activate", win)
    wx, wy, ww, wh = map(int, xctl("geom", win).split())
    ww, wh = ww // 2 * 2, wh // 2 * 2
    # MORAI 는 포인터가 창 위에 있으면 커서를 화면 안에 그린다 → 창 밖으로
    xctl("wmove", win, ww + 300 if wx + ww + 300 < 3800 else -300, wh // 2)
    raw = out / f"{tag}_raw.mkv"
    flog = open(out / f"{tag}.ffmpeg.log", "w")
    fp = subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-f", "x11grab", "-draw_mouse", "0", "-framerate", "30",
                           "-video_size", f"{ww}x{wh}", "-i", f":1+{wx},{wy}", "-c:v", "h264_nvenc", "-preset", "slow",
                           "-rc", "vbr", "-cq", "23", "-b:v", "0", "-pix_fmt", "yuv420p", str(raw)],
                          stdout=flog, stderr=subprocess.STDOUT, env=ENV)
    (out / f"{tag}.video_t0").write_text(f"{time.time():.3f}\n")
    t0 = time.time()
    method, keep, extra = METHODS[m]
    try:
        rc, so, se = run(["bash", str(HERE / "run_episode.sh"), tag, method, keep] + extra, 420)
    finally:
        fp.send_signal(2)
        try:
            fp.wait(20)
        except subprocess.TimeoutExpired:
            fp.terminate()
        xctl("above", win, "off")
        subprocess.run(["gsettings", "set", "org.gnome.desktop.notifications", "show-banners", banners])
    mon = None
    try:
        mon = json.loads((out / f"{tag}.monitor.json").read_text().strip().splitlines()[-1])
    except Exception:
        pass
    met = None
    bag = out / f"{tag}.bag"
    if bag.exists():
        r2, so2, se2 = run(["python3", str(REPO / "research" / "analysis" / "morai_metrics.py"), str(bag)], 600)
        try:
            met = json.loads(so2.strip().splitlines()[-1])
        except Exception:
            met = {"error": se2[-500:]}
    return {"tag": tag, "scenario": scen, "method": m, "monitor": mon, "metrics": met,
            "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="<시나리오(research_ 뺀 이름)>:<방법(run_matrix.METHODS)>")
    ap.add_argument("--out", default=str(Path.home() / "bags/research/video"))
    ap.add_argument("--prefix", default="vid_")
    a = ap.parse_args()
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["OUT"] = str(out)                    # run_episode.sh 의 bag·로그 위치
    os.environ["MORAI_SHOT_DIR"] = str(out / "shots")
    (out / "shots").mkdir(exist_ok=True)
    log = out / "vid_runs.jsonl"
    done = {json.loads(l)["tag"] for l in log.read_text().splitlines() if l.strip()} if log.exists() else set()
    for item in a.runs:
        sc, m = item.split(":")
        if m not in METHODS:
            sys.exit(f"unknown method {m}")
        tag = f"{a.prefix}{sc}_{m}"
        if tag in done:
            continue
        for wait in range(20):                      # 사람이 쓰는 중이면 최대 20 분 기다림
            if not pointer_busy():
                break
            print(f"USER_ACTIVE before {tag}, waiting 60 s", flush=True)
            time.sleep(60)
        else:
            print("USER_ACTIVE for 20 min, stop", flush=True)
            return 3
        print(f"=== {tag} {time.strftime('%H:%M:%S')}", flush=True)
        rec = record_one(sc, m, out, a.prefix)
        if "error" in rec or not (rec.get("metrics") or {}).get("bag"):
            print("RUN PROBLEM", json.dumps(rec)[:400], flush=True)
        log.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(json.dumps({k: rec.get(k) for k in ("tag", "monitor")}), flush=True)
    print("ALL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
