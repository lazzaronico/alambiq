"""Signal tests with disposable native workers run through a capability of the core.

Run by `test_native_signal.py` as its own process, which it then interrupts. The mode
says whether the worker is launched directly or from a campaign step, and whether its
output is captured or logged.
"""

import signal
import sys
from dataclasses import dataclass
from pathlib import Path

from alambiq._native_process import native_process_scope, run_captured, run_logged
from alambiq.core.registry import capability

# A job started in the background inherits SIGINT as ignored, which would leave nothing
# for the run to relay; this process is the subject of the test, so it sets its own.
signal.signal(signal.SIGINT, signal.default_int_handler)

folder = Path(sys.argv[1])
mode = sys.argv[2]
child = "import os,time,json;print(json.dumps({'child':os.getpid(),'group':os.getpgrp()}),flush=True);time.sleep(30)"
parent = f"import subprocess,sys,os,json;print(json.dumps({{'parent':os.getpid(),'group':os.getpgrp()}}),flush=True);subprocess.Popen([sys.executable,'-u','-c',{child!r}]).wait()"
pids = str(folder / "pids.log")
capture_child = f"import os,time,json;open({pids!r},'a').write(json.dumps({{'child':os.getpid()}})+'\\n');time.sleep(30)"
capture_parent = f"import os,sys,subprocess,json;open({pids!r},'w').write(json.dumps({{'parent':os.getpid()}})+'\\n');subprocess.Popen([sys.executable,'-c',{capture_child!r}]).wait()"


def wait_native() -> int:
    if "captured" in mode:
        return run_captured(
            [sys.executable, "-c", capture_parent], timeout=30
        ).returncode
    with (folder / "pids.log").open("w") as log:
        return run_logged(
            [sys.executable, "-u", "-c", parent], log, timeout=30
        ).returncode


@dataclass(frozen=True, slots=True)
class Request:
    """Nothing to ask for.

    nothing:
        Unused.
    """

    nothing: int = 0


@dataclass(frozen=True, slots=True)
class Result:
    """The worker's exit status."""

    returncode: int


@capability(
    "test.wait_native",
    version="1.0",
    family="investigate",
    summary="Wait for a disposable native worker",
    render=lambda result: "",
)
def wait_capability(request: Request, ctx) -> Result:
    """Launch the worker and wait for it."""
    return Result(wait_native())


if mode.startswith("direct"):
    with native_process_scope():
        wait_native()
else:
    from alambiq.api import run

    run(
        {
            "id": "signal",
            "steps": [{"id": "wait", "capability": "test.wait_native", "params": {}}],
        },
        run_dir=folder / "run",
    )
