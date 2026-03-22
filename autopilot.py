"""
Autonomous supervisor for Astar Island.

Responsibilities:
1. Keep the watcher alive at all times
2. Stop training during active rounds so prediction/submission has the GPU
3. Restart training between rounds
4. Restart stale processes if they stop updating their logs
5. Publish a small status file for easy monitoring
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from api_client import AstarClient
from watcher import TOKEN

BASE_DIR = Path("/workspace")
WATCHER_SCRIPT = BASE_DIR / "watcher.py"
TRAIN_SCRIPT = BASE_DIR / "train_v6.py"
ACTIVE_TRAIN_SCRIPT = BASE_DIR / "train_active.py"
LATE_SUBMIT_SCRIPT = BASE_DIR / "late_submit_worker.py"
WATCHER_LOG = BASE_DIR / "watcher.log"
TRAIN_LOG = BASE_DIR / "train_v6.log"
ACTIVE_TRAIN_LOG = BASE_DIR / "train_active.log"
AUTOPILOT_LOG = BASE_DIR / "autopilot.log"
STATUS_FILE = BASE_DIR / "autopilot_status.json"

POLL_SECS = 30
WATCHER_STALE_SECS = 180
TRAIN_STALE_SECS = 1800
LATE_SUBMIT_BUFFER_SECS = 15 * 60


def log(msg: str):
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with AUTOPILOT_LOG.open("a") as f:
        f.write(line + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_time(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def iter_matching_pids(script_path: Path):
    needle = str(script_path)
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore").replace("\x00", " ")
        except Exception:
            continue
        if needle in cmd and int(pid) != os.getpid():
            yield int(pid)


def newest_pid(script_path: Path) -> Optional[int]:
    pids = sorted(iter_matching_pids(script_path))
    return pids[-1] if pids else None


def kill_script(script_path: Path):
    for pid in list(iter_matching_pids(script_path)):
        try:
            os.kill(pid, signal.SIGTERM)
            log(f"Stopped {script_path.name} pid={pid}")
        except ProcessLookupError:
            pass


def start_script(script_path: Path, log_path: Path) -> int:
    proc = subprocess.Popen(
        ["python3", "-u", str(script_path)],
        stdout=log_path.open("w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log(f"Started {script_path.name} pid={proc.pid}")
    return proc.pid


def ensure_round_worker(round_id: str, round_number: int, closes_at: str) -> int:
    needle = f"{LATE_SUBMIT_SCRIPT} {round_id}"
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode("utf-8", "ignore").replace("\x00", " ")
        except Exception:
            continue
        if needle in cmd:
            return int(pid)

    proc = subprocess.Popen(
        ["python3", "-u", str(LATE_SUBMIT_SCRIPT), round_id, str(round_number), closes_at],
        stdout=(BASE_DIR / "late_submit_worker.console.log").open("w"),
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    log(f"Started late_submit_worker.py for round #{round_number} pid={proc.pid}")
    return proc.pid


def ensure_running(script_path: Path, log_path: Path, stale_after: int) -> int:
    pid = newest_pid(script_path)
    if pid is None:
        return start_script(script_path, log_path)

    if log_path.exists():
        age = time.time() - log_path.stat().st_mtime
        if age > stale_after:
            log(f"{script_path.name} looks stale ({age:.0f}s since log update), restarting")
            kill_script(script_path)
            return start_script(script_path, log_path)

    return pid


def main():
    client = AstarClient(TOKEN)
    log("Astar autopilot started")

    while True:
        active_round = None
        try:
            active_round = client.get_active_round()
        except Exception as e:
            log(f"Active round check failed: {e}")

        watcher_pid = ensure_running(WATCHER_SCRIPT, WATCHER_LOG, WATCHER_STALE_SECS)

        train_pid = newest_pid(TRAIN_SCRIPT)
        active_train_pid = newest_pid(ACTIVE_TRAIN_SCRIPT)
        late_submit_pid = None
        if active_round:
            if train_pid is not None:
                log(f"Active round #{active_round['round_number']} detected; pausing training")
                kill_script(TRAIN_SCRIPT)
                train_pid = None
            secs_left = parse_time(active_round["closes_at"]).timestamp() - time.time()
            late_submit_pid = ensure_round_worker(
                active_round["id"], active_round["round_number"], active_round["closes_at"])
            if secs_left > (LATE_SUBMIT_BUFFER_SECS + 15 * 60):
                active_train_pid = ensure_running(ACTIVE_TRAIN_SCRIPT, ACTIVE_TRAIN_LOG, TRAIN_STALE_SECS)
            elif active_train_pid is not None:
                log("Close to final overwrite window; stopping active-round training")
                kill_script(ACTIVE_TRAIN_SCRIPT)
                active_train_pid = None
        else:
            if active_train_pid is not None:
                kill_script(ACTIVE_TRAIN_SCRIPT)
                active_train_pid = None
            train_pid = ensure_running(TRAIN_SCRIPT, TRAIN_LOG, TRAIN_STALE_SECS)

        status = {
            "timestamp_utc": utc_now(),
            "active_round": active_round,
            "watcher_pid": watcher_pid,
            "trainer_pid": train_pid,
            "active_trainer_pid": active_train_pid,
            "late_submit_pid": late_submit_pid,
            "watcher_log": str(WATCHER_LOG),
            "trainer_log": str(TRAIN_LOG),
            "active_trainer_log": str(ACTIVE_TRAIN_LOG),
            "autopilot_log": str(AUTOPILOT_LOG),
        }
        STATUS_FILE.write_text(json.dumps(status, indent=2))
        time.sleep(POLL_SECS)


if __name__ == "__main__":
    main()
