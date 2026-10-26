"""The temporary Prefect server ends with the command that started it.

Prefect starts the server as a child process and stops it only when Python exits
normally. The command line stops it also when the program is ended by a termination
request or a hang-up. A kill that cannot be caught is the one end that leaves it.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")) or sys.platform == "win32",
    reason="the quantitative extra is required, on a system with signals",
)

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = ROOT / "examples/kinetics-from-measurements/campaign.yaml"


def servers_of(parent: int) -> list[int]:
    listing = subprocess.run(
        ["ps", "-axww", "-o", "pid=,ppid=,command="], capture_output=True, text=True
    ).stdout
    found = []
    for line in listing.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and "prefect.server.api.server:create_app" in parts[2]:
            if int(parts[1]) == parent:
                found.append(int(parts[0]))
    return found


def start(tmp_path):
    env = {**os.environ, "PREFECT_HOME": str(tmp_path / "prefect")}
    env.pop("PREFECT_API_URL", None)
    return subprocess.Popen(
        [sys.executable, "-m", "alambiq", "run", str(CAMPAIGN),
         "--run-dir", str(tmp_path / "run")],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=ROOT,
    )  # fmt: skip


def wait_for(condition, seconds):
    end = time.time() + seconds
    while time.time() < end:
        if condition():
            return True
        time.sleep(0.2)
    return False


@pytest.mark.parametrize("name", ["SIGTERM", "SIGHUP"])
def test_the_server_ends_when_the_command_is_ended_by_a_signal(tmp_path, name):
    process = start(tmp_path)
    try:
        assert wait_for(lambda: servers_of(process.pid), 120), "no server started"
        child = servers_of(process.pid)
        process.send_signal(getattr(signal, name))
        process.wait(timeout=60)
        assert wait_for(lambda: not any(_alive(p) for p in child), 20), child
    finally:
        if process.poll() is None:
            process.kill()


def test_the_server_ends_when_the_command_finishes(tmp_path):
    process = start(tmp_path)
    seen: list[int] = []
    while process.poll() is None:
        seen += [p for p in servers_of(process.pid) if p not in seen]
        time.sleep(0.2)
    assert process.returncode == 0
    assert seen
    assert wait_for(lambda: not any(_alive(p) for p in seen), 20), seen


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    listing = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout.strip()
    return bool(listing) and not listing.startswith("Z")
