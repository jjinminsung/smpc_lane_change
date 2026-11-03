#!/usr/bin/env python3
"""Replay target_selector_node alone with parameter overrides (re-verification 2).

Same recipe as gtcheck_replay.py (isolated master 11411 only, use_sim_time, --clock,
config/target_selector.yaml loaded into the node namespace, target_source=object_topic),
plus --set key=value overrides (e.g. the research_morai.launch UDP-path settings) and a
selectable input topic list.  Every process started here is stopped by PID.

Output: research/results/gtcheck_replay/<label><suffix>.bag (+ node log, cpu).
"""
import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gtcheck_replay import (CFG, LANES, NODE, OUT, PORT, cpu_seconds, port_open,  # noqa: E402
                            stop)

DEFAULT_TOPICS = ["/Object_topic", "/odom", "/gps_state", "/smpc/lane_change_status", "/Ego_topic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--set", nargs="*", default=[], help="param=value overrides (node private)")
    ap.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    os.makedirs(OUT, exist_ok=True)
    env = dict(os.environ)
    env["ROS_MASTER_URI"] = "http://localhost:%d" % PORT
    env.pop("ROS_IP", None)
    if port_open(PORT):
        sys.exit("port %d already in use - refusing to share a master" % PORT)
    core = subprocess.Popen(["roscore", "-p", str(PORT)], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("[replay2] roscore pid %d on %d" % (core.pid, PORT))
    try:
        for _ in range(100):
            if port_open(PORT):
                break
            time.sleep(0.1)
        time.sleep(1.0)
        run = lambda *c: subprocess.run(list(c), env=env, check=True)
        run("rosparam", "set", "/use_sim_time", "true")
        for bag in a.bags:
            base = os.path.basename(bag)[:-4]
            label = (base[len("lc_gt_"):] if base.startswith("lc_gt_") else base) + a.suffix
            out_bag = os.path.join(OUT, label + ".bag")
            if subprocess.run(["rosparam", "get", "/" + NODE], env=env, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL).returncode == 0:
                run("rosparam", "delete", "/" + NODE)
            run("rosparam", "load", CFG, "/" + NODE)
            run("rosparam", "set", "/%s/target_source" % NODE, "object_topic")
            run("rosparam", "set", "/%s/waypoint_directory" % NODE, LANES)
            for kv in a.set:
                k, v = kv.split("=", 1)
                run("rosparam", "set", "/%s/%s" % (NODE, k), v)
            log = open(os.path.join(OUT, label + "_node.log"), "w")
            node = rec = None
            try:
                node = subprocess.Popen(["rosrun", "smpc_lane_change", "target_selector_node",
                                         "__name:=" + NODE], env=env, stdout=log, stderr=log)
                rec = subprocess.Popen(["rosbag", "record", "-O", out_bag, "/smpc/targets",
                                        "__name:=gtcheck_record"], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[replay2] %s node pid %d record pid %d" % (label, node.pid, rec.pid))
                time.sleep(4.0)
                t0 = time.time()
                subprocess.run(["rosbag", "play", bag, "--clock", "-q", "--topics"] + a.topics,
                               env=env, check=True)
                wall = time.time() - t0
                time.sleep(2.0)
                cpu = cpu_seconds(node.pid)
                print("[replay2] %s played in %.1f s, node cpu %.1f s" % (label, wall, cpu))
                with open(os.path.join(OUT, label + "_cpu.txt"), "w") as fc:
                    fc.write("wall_play_s %.3f\nnode_cpu_s %.3f\nrate 1.000\n" % (wall, cpu))
            finally:
                stop(rec, "record")
                stop(node, "node")
                log.close()
    finally:
        stop(core, "roscore")
        print("[replay2] roscore stopped")


if __name__ == "__main__":
    main()
