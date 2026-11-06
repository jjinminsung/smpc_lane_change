#!/usr/bin/env python3
"""record_videos.py 로 찍은 원본(<tag>_raw.mkv)에 자막을 입혀 보여 줄 영상(<tag>.mp4)을 만든다.

    python3 make_video.py --dir ~/bags/research/video_1106            # vid_runs.jsonl 의 모든 회
    python3 make_video.py --dir ~/bags/research/video_1106 vid_S3_rear_decel_rx_fix

자막(bag 에서): 방법·시나리오, 시간·속도·차선, 이벤트(차선 변경 시작·완료·복귀, 차선 끝 정지, MORAI 접촉), 결과.
시간 t 는 morai_metrics.py 와 같이 자차가 처음 움직인 순간(0.5 m/s)부터 센다.
출발 3 s 전부터 합류 4 s 뒤까지 자른다. 75 s 보다 길면 2 배속(자막에 표시).
"""
import argparse
import bisect
import json
import subprocess
from pathlib import Path

import rosbag

W_OUT = 1080
METHOD_LABEL = {
    "rx_fix": "새 구조 — 반응형 시나리오 트리 SMPC",
    "rx_single_fix": "새 구조 - 가지 없음 (비교군)",
    "baseline": "기존 SMPC (비교군)",
    "tree_nogate": "1차 구조 D (비교군)",
}
S_DESC = {
    "S1_normal": "S1 보통 교통: NPC 70 → 70 km/h",
    "S2_rear_accel": "S2 뒤차 가속: NPC 54 → 90 km/h",
    "S3_rear_decel": "S3 감속하는 교통: NPC 72 → 43 km/h",
    "S4_front_decel": "S4 느린 교통: NPC 72 → 29 km/h",
}


def scen_desc(sc):
    sc = sc.replace("research_", "")
    if sc in S_DESC:
        return S_DESC[sc]
    _, s, d = sc.split("_")                    # H_s30_d40
    s, d = int(s[1:]), int(d[1:])
    kind = "느린 교통" if d <= 60 else "빠른 교통"
    return f"H 격자 {kind}: NPC 생성 {s} → 목표 {d} km/h"


def ts_fmt(t):
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def esc(s):
    return s.replace("{", "(").replace("}", ")")


def read_bag(path):
    odom, status, targets, sup, coll = [], [], [], [], []
    b = rosbag.Bag(str(path))
    for top, m, t in b.read_messages(topics=["/odom", "/smpc/lane_change_status", "/smpc/targets",
                                             "/smpc/lane_change_supervisor_debug", "/CollisionData"]):
        ts = t.to_sec()
        if top == "/odom":
            odom.append((ts, (m.twist.twist.linear.x ** 2 + m.twist.twist.linear.y ** 2) ** 0.5))
        elif top == "/smpc/lane_change_status":
            status.append((ts, m.active, m.source_lane_id, m.target_lane_id))
        elif top == "/smpc/targets":
            targets.append((ts, m.current_lane_id, m.distance_to_lane_end))
        elif top == "/smpc/lane_change_supervisor_debug":
            sup.append((ts, m.data))
        else:
            others = [c.unique_id for c in m.collision_object if c.unique_id != 0]
            if others:
                coll.append((ts, others))
    b.close()
    return odom, status, targets, sup, coll


def events_of(odom, status, targets, sup, t_move, contacts):
    """(절대 시각, 문구) 목록. morai_metrics.py 와 같은 규칙으로 복귀를 가른다."""
    aborts = [ts for ts, d in sup if ("abort" in d and ("confirmed" in d or "emergency" in d or "tree_return" in d))
              or d.startswith("tree_return")]
    ev, prev = [], None
    for ts, act, src, tgt in status:
        if prev is not None and act and not prev[1]:
            ev.append((ts, f"차선 변경 시작  {src} → {tgt}", "start"))
        if prev is not None and not act and prev[1]:
            if any(abs(ts - a) < 0.6 for a in aborts):
                ev.append((ts, f"복귀  → 차선 {prev[2]}", "abort"))
            else:
                ev.append((ts, f"차선 변경 완료  {prev[2]} → {prev[3]}", "done" if prev[3] == 3 else "complete"))
        prev = (ts, act, src, tgt)
    tt = [x[0] for x in targets]
    stopped = False
    for ts, v in odom:
        j = max(0, min(bisect.bisect_left(tt, ts), len(tt) - 1)) if tt else None
        lane, dist = (targets[j][1], targets[j][2]) if j is not None else (3, 1e9)
        if v < 0.5 and lane < 3 and dist < 60.0 and not stopped:
            ev.append((ts, f"차선 {lane} 끝에서 정지 (남은 {dist:.0f} m)", "stop"))
            stopped = True
        elif v > 1.0:
            stopped = False
    # MORAI 접촉: /CollisionData 에는 NPC 끼리의 접촉도 들어온다. morai_metrics.py 가 자차와 GT 차체 간격 0.3 m 이내로
    # 걸러 낸 morai_contacts(t = 자차가 움직인 뒤 시간)만 쓴다.
    for c in contacts or []:
        ev.append((t_move + c["t"], f"MORAI 접촉 (NPC {c['uid']})", "contact"))
    return sorted(ev)


def make(d, rec):
    tag = rec["tag"]
    raw, bag = d / f"{tag}_raw.mkv", d / f"{tag}.bag"
    if not raw.exists() or not bag.exists():
        print("skip (no raw/bag)", tag)
        return None
    v0 = float((d / f"{tag}.video_t0").read_text())
    odom, status, targets, sup, coll = read_bag(bag)
    mv = [ts for ts, v in odom if v > 0.5]
    t_move = mv[0] if mv else odom[0][0]
    met = rec.get("metrics") or {}
    ev = events_of(odom, status, targets, sup, t_move, met.get("morai_contacts"))
    t_done = next((ts for ts, _, k in ev if k == "done"), None)
    dur_raw = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(raw)],
                                   capture_output=True, text=True).stdout.strip() or 0)
    start = max(0.0, t_move - v0 - 3.0)
    end = min(dur_raw, (t_done - v0 + 4.0) if t_done else (odom[-1][0] - v0 + 1.0))
    speed = 2.0 if end - start > 75.0 else 1.0
    out_len = (end - start) / speed

    def o(ts):          # 절대 시각 → 출력 영상 시각
        return (ts - v0 - start) / speed

    # 화면 비율: 원본 크기 → 출력 높이
    wh = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                         "-of", "csv=p=0", str(raw)], capture_output=True, text=True).stdout.strip().split(",")
    H_OUT = int(round(W_OUT * int(wh[1]) / int(wh[0]) / 2) * 2)
    sc = rec["scenario"]
    head = [METHOD_LABEL.get(rec["method"], rec["method"]), scen_desc(sc),
            "자차: 차선 0 → 3 합류 (차선 0·1·2 는 끝남) · GT 인지" + (" · 2 배속" if speed > 1 else "")]
    ass = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W_OUT}", f"PlayResY: {H_OUT}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Head,Noto Sans CJK KR,30,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,8,0,7,24,24,54,1",
        "Style: Sub,Noto Sans CJK KR,24,&H00F0F0F0,&H00FFFFFF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,6,0,7,24,24,98,1",
        "Style: Live,Noto Sans CJK KR,26,&H00FFFFFF,&H00FFFFFF,&H00000000,&H90000000,1,0,0,0,100,100,0,0,3,8,0,1,24,24,30,1",
        "Style: Ev,Noto Sans CJK KR,40,&H00FFFFFF,&H00FFFFFF,&H00000000,&HB0000000,1,0,0,0,100,100,0,0,3,12,0,8,24,24,190,1",
        "Style: End,Noto Sans CJK KR,38,&H00FFFFFF,&H00FFFFFF,&H00000000,&HC0000000,1,0,0,0,100,100,0,0,3,16,0,5,40,40,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    ass.append(f"Dialogue: 0,{ts_fmt(0)},{ts_fmt(out_len)},Head,,0,0,0,,{esc(head[0])}")
    ass.append(f"Dialogue: 0,{ts_fmt(0)},{ts_fmt(out_len)},Sub,,0,0,0,,{esc(head[1])}\\N{esc(head[2])}")
    # 시간·속도·차선 (0.5 s 마다, 원본 시간 기준)
    tt = [x[0] for x in targets]
    ot = [x[0] for x in odom]
    step = 0.5
    t = v0 + start
    while t < v0 + end:
        i = min(bisect.bisect_left(ot, t), len(ot) - 1)
        j = min(bisect.bisect_left(tt, t), len(tt) - 1) if tt else None
        lane = targets[j][1] if j is not None else "?"
        rel = t - t_move
        txt = f"t {rel:5.1f} s   {odom[i][1] * 3.6:4.0f} km/h   차선 {lane}" if rel >= 0 else f"출발 전   차선 {lane}"
        ass.append(f"Dialogue: 0,{ts_fmt(o(t))},{ts_fmt(o(t + step))},Live,,0,0,0,,{txt}")
        t += step
    # 이벤트 배너 (2.5 s, 겹치면 다음 이벤트 전까지)
    for k, (ts, txt, kind) in enumerate(ev):
        if not (v0 + start <= ts <= v0 + end):
            continue
        e = ts + 2.5 * speed
        if k + 1 < len(ev):
            e = min(e, max(ev[k + 1][0], ts + 0.8 * speed))
        if kind == "done":
            txt = f"합류 완료  {ts - t_move:.1f} s"
        ass.append(f"Dialogue: 1,{ts_fmt(o(ts))},{ts_fmt(o(e))},Ev,,0,0,0,,{esc(txt)}")
    # 결과
    if met.get("success"):
        res = (f"결과: 합류 성공 {met['t_merge']:.1f} s\\N복귀 {len(met.get('aborts', []))} 회 · "
               f"차선 끝 정지 {met.get('lane_end_stops', 0)} 회 · 접촉 {len(met.get('morai_contacts', []))} 건")
    else:
        reason = (rec.get("monitor") or {}).get("reason", "?")
        why = {"timeout": "180 s 안에 합류 못 함", "stuck": "차선 끝에서 60 s 넘게 멈춤", "overrun": "차선 끝을 넘음"}.get(reason, reason)
        res = f"결과: 실패 ({why})\\N복귀 {len(met.get('aborts', []))} 회 · 차선 끝 정지 {met.get('lane_end_stops', 0)} 회"
    ass.append(f"Dialogue: 2,{ts_fmt(max(0, out_len - 3.5))},{ts_fmt(out_len)},End,,0,0,0,,{res}")
    af = d / f"{tag}.ass"
    af.write_text("\n".join(ass) + "\n")
    out = d / f"{tag}.mp4"
    vf = f"setpts=(PTS-STARTPTS)/{speed},scale={W_OUT}:{H_OUT},ass={af}"
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(raw),
           "-vf", vf, "-r", "30", "-c:v", "h264_nvenc", "-preset", "slow", "-rc", "vbr", "-cq", "30", "-b:v", "0",
           "-maxrate", "3M", "-bufsize", "6M", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
    subprocess.run(cmd, check=True)
    info = {"tag": tag, "mp4": out.name, "len_s": round(out_len, 1), "speed": speed, "size_mb": round(out.stat().st_size / 1e6, 1),
            "events": [(round(ts - t_move, 1), txt) for ts, txt, _ in ev], "result": res.replace("\\N", " / ")}
    print(json.dumps(info, ensure_ascii=False), flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*")
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir).expanduser()
    recs = {}
    for line in (d / "vid_runs.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["tag"]] = r
    tags = a.tags or list(recs)
    infos = [make(d, recs[t]) for t in tags if t in recs]
    old = {}
    if (d / "videos.json").exists():
        old = {x["tag"]: x for x in json.loads((d / "videos.json").read_text())}
    old.update({x["tag"]: x for x in infos if x})
    (d / "videos.json").write_text(json.dumps(list(old.values()), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
