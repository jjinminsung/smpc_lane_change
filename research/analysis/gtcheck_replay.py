#!/usr/bin/env python3
"""Replay target_selector_node alone in object_topic mode (GT check A).

Isolated ROS master on port 11411 only (never 11311, the MORAI side).  For each
bag: load config/target_selector.yaml into the node namespace (exactly what
launch/smpc_integration.launch does), override ~target_source=object_topic and
~waypoint_directory, play the node inputs with --clock (use_sim_time) and
record the new /smpc/targets.  Every process started here is stopped by PID.

Output: research/results/gtcheck_replay/<label>.bag (+ node log).
"""
import argparse
import os
import signal
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(REPO, "research", "results", "gtcheck_replay")
CFG = os.path.join(REPO, "config", "target_selector.yaml")
LANES = os.path.expanduser("~/catkin_ws/src/waypoint_system/data/default")
PORT = 11411
NODE = "smpc_target_selector"
PLAY_TOPICS = ["/Object_topic", "/odom", "/gps_state", "/smpc/lane_change_status", "/Ego_topic"]


def port_open(port):
    s = socket.socket()
    s.settimeout(0.3)
    try:
        return s.connect_ex(("127.0.0.1", port)) == 0
    finally:
        s.close()


def stop(proc, name, sig=signal.SIGINT, timeout=15.0):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[replay] %s (pid %d) did not exit, SIGTERM" % (name, proc.pid), file=sys.stderr)
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()


def cpu_seconds(pid):
    """utime+stime of a process (and its exec'd node) in seconds."""
    try:
        with open("/proc/%d/stat" % pid) as f:
            parts = f.read().rsplit(")", 1)[1].split()
        return (int(parts[11]) + int(parts[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bags", nargs="+")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--suffix", default="")
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
    print("[replay] roscore pid %d on %d" % (core.pid, PORT))
    try:
        for _ in range(100):
            if port_open(PORT):
                break
            time.sleep(0.1)
        time.sleep(1.0)
        run = lambda *c: subprocess.run(list(c), env=env, check=True)
        run("rosparam", "set", "/use_sim_time", "true")
        for bag in a.bags:
            label = os.path.basename(bag)[len("lc_gt_"):-4] + a.suffix
            out_bag = os.path.join(OUT, label + ".bag")
            log = open(os.path.join(OUT, label + "_node.log"), "w")
            run("rosparam", "delete", "/" + NODE) if subprocess.run(
                ["rosparam", "get", "/" + NODE], env=env, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL).returncode == 0 else None
            run("rosparam", "load", CFG, "/" + NODE)
            run("rosparam", "set", "/%s/target_source" % NODE, "object_topic")
            run("rosparam", "set", "/%s/waypoint_directory" % NODE, LANES)
            run("rosparam", "set", "/%s/tracked_objects_topic" % NODE, "/tracked_objects")
            node = rec = None
            try:
                node = subprocess.Popen(["rosrun", "smpc_lane_change", "target_selector_node",
                                         "__name:=" + NODE], env=env, stdout=log, stderr=log)
                rec = subprocess.Popen(["rosbag", "record", "-O", out_bag, "/smpc/targets",
                                        "__name:=gtcheck_record"], env=env,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print("[replay] %s node pid %d record pid %d" % (label, node.pid, rec.pid))
                time.sleep(4.0)
                t0 = time.time()
                subprocess.run(["rosbag", "play", bag, "--clock", "-q", "-r", str(a.rate),
                                "--topics"] + PLAY_TOPICS, env=env, check=True)
                wall = time.time() - t0
                time.sleep(2.0)
                cpu = cpu_seconds(node.pid)
                print("[replay] %s played in %.1f s, node cpu %.1f s" % (label, wall, cpu))
                with open(os.path.join(OUT, label + "_cpu.txt"), "w") as fc:
                    fc.write("wall_play_s %.3f\nnode_cpu_s %.3f\nrate %.3f\n" % (wall, cpu, a.rate))
            finally:
                stop(rec, "record")
                stop(node, "node")
                log.close()
    finally:
        stop(core, "roscore")
        print("[replay] roscore stopped")


if __name__ == "__main__":
    main()
