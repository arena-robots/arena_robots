#!/usr/bin/env python3
"""Run and record the lever door opening sequence across 3 start poses."""

import json
import subprocess
import sys
import time
from pathlib import Path

POSES = {
    1: {"name": "Center Approach", "scenario": "pose_1"},
    2: {"name": "Left Approach", "scenario": "pose_2"},
    3: {"name": "Right Approach", "scenario": "pose_3"},
}

DATA_DIR = Path("/opt/arena_ws/data")


def run_pose(pose_id: int) -> dict:
    info = POSES[pose_id]
    print(f"\n==========================================")
    print(f" Running Pose {pose_id}: {info['name']}")
    print(f"==========================================")

    metrics_file = DATA_DIR / f"metrics_pose_{pose_id}.json"
    if metrics_file.exists():
        metrics_file.unlink()

    # Launch sim + opener
    cmd_opener = [
        "ros2", "run", "arena_robots", "lever_door_opener",
        "--ros-args", "-p", f"metrics_file:={metrics_file}",
    ]
    opener_proc = subprocess.Popen(cmd_opener)
    time.sleep(2.0)

    # Trigger opening
    subprocess.run(["ros2", "service", "call", "/lever_door_opener/open", "std_srvs/srv/Trigger", "{}"], check=True)

    # Wait for metrics file
    start_t = time.time()
    result = None
    while time.time() - start_t < 60.0:
        if metrics_file.exists():
            try:
                result = json.loads(metrics_file.read_text())
                break
            except json.JSONDecodeError:
                pass
        time.sleep(1.0)

    opener_proc.terminate()
    opener_proc.wait()

    if not result:
        result = {"verdict": "TIMEOUT", "peak_handle_force_n": 0.0, "contact_speed_mps": 0.0}
    return result


def main() -> None:
    pose_arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    selected_poses = [int(pose_arg)] if pose_arg.isdigit() else [1, 2, 3]

    results = {}
    for pid in selected_poses:
        results[pid] = run_pose(pid)

    print("\n" + "=" * 65)
    print(" SUMMARY FOR PR (CW46 / CW47 LEVER DOOR BENCHMARK)")
    print("=" * 65)
    print(f"{'Pose':<8} | {'Peak Force (N)':<15} | {'Contact Speed (m/s)':<20} | {'Verdict':<8}")
    print("-" * 65)
    for pid, res in results.items():
        f = res.get("peak_handle_force_n", 0.0)
        s = res.get("contact_speed_mps", 0.0)
        v = res.get("verdict", "FAIL")
        print(f"Pose {pid:<3} | {f:<6.2f} (< 22.0 N)  | {s:<6.4f} (< 0.05 m/s)   | {v}")
    print("=" * 65)
    print("Videos saved under: /opt/arena_ws/data/recordings/")


if __name__ == "__main__":
    main()

