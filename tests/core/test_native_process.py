"""Real POSIX child lifetime tests; no scientific executable or model required."""

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from alambiq._native_process import native_process_scope, run_logged


@unittest.skipUnless(os.name == "posix", "Alambiq native profiles target Mac/Linux")
class NativeProcessTests(unittest.TestCase):
    def test_long_allowance_returns_when_the_child_completes(self):
        for timeout in (120 * 86400, sys.maxsize):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "native.log"
                with path.open("w") as log:
                    result = run_logged(
                        [sys.executable, "-c", "print('completed')"],
                        log,
                        timeout=timeout,
                    )
                self.assertEqual(result.returncode, 0)
                self.assertEqual(path.read_text(), "completed\n")

    def test_closed_run_context_cannot_launch_late_tasks_and_restores_handlers(self):
        from contextvars import copy_context

        before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        with native_process_scope():
            copied = copy_context()
        self.assertEqual(before, {sig: signal.getsignal(sig) for sig in before})
        with patch("alambiq._native_process.subprocess.Popen") as popen:
            with self.assertRaisesRegex(RuntimeError, "scope closed"):
                copied.run(run_logged, ["test"], object(), timeout=10)
            popen.assert_not_called()

    def test_success_nonzero_output_and_launch_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "native.log"
            with path.open("w") as log:
                result = run_logged(
                    [
                        sys.executable,
                        "-c",
                        "import os,sys;print(os.getcwd());print(os.environ['TEST_NATIVE_VALUE']);print('stderr',file=sys.stderr);sys.exit(7)",
                    ],
                    log,
                    cwd=folder,
                    env={**os.environ, "TEST_NATIVE_VALUE": "retained"},
                    timeout=5,
                )
            self.assertEqual(result.returncode, 7)
            self.assertIn(folder, path.read_text())
            self.assertIn("retained", path.read_text())
            self.assertIn("stderr", path.read_text())
            with path.open("a") as log, self.assertRaises(FileNotFoundError):
                run_logged([str(Path(folder) / "absent")], log, timeout=5)

    def test_timeout_stops_parent_and_child_not_unrelated_process(self):
        child = "import os,time,json;print(json.dumps({'child':os.getpid(),'group':os.getpgrp()}),flush=True);time.sleep(30)"
        parent = f"import subprocess,sys,os,time,json;print(json.dumps({{'parent':os.getpid(),'group':os.getpgrp()}}),flush=True);p=subprocess.Popen([sys.executable,'-u','-c',{child!r}]);p.wait()"
        outsider = subprocess.Popen(
            [sys.executable, "-c", "import time;time.sleep(30)"]
        )
        pids = []
        try:
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "native.log"
                with (
                    path.open("w") as log,
                    self.assertRaises(subprocess.TimeoutExpired),
                ):
                    run_logged([sys.executable, "-u", "-c", parent], log, timeout=1)
                records = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(records), 2, records)
                pids = [records[0]["parent"], records[1]["child"]]
                self.assertEqual(records[0]["group"], pids[0])
                self.assertEqual(records[1]["group"], pids[0])
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    states = [
                        subprocess.run(
                            ["ps", "-o", "stat=", "-p", str(pid)],
                            capture_output=True,
                            text=True,
                        ).stdout.strip()
                        for pid in pids
                    ]
                    if all(not state or state.startswith("Z") for state in states):
                        break
                    time.sleep(0.02)
                self.assertTrue(
                    all(not state or state.startswith("Z") for state in states), states
                )
                self.assertIsNone(outsider.poll())
        finally:
            # Only test-created processes are eligible for this fallback cleanup.
            for pid in pids:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            outsider.kill()
            outsider.wait()

    def test_direct_interrupt_cleans_group_reaps_and_preserves_exception(self):
        with (
            patch("alambiq._native_process.subprocess.Popen") as popen,
            patch("alambiq._native_process.os.killpg") as kill,
        ):
            process = popen.return_value
            process.pid = 12345
            process.wait.side_effect = [KeyboardInterrupt(), -signal.SIGKILL]
            with self.assertRaises(KeyboardInterrupt):
                run_logged(["test-program"], object(), timeout=10)
            self.assertTrue(popen.call_args.kwargs["start_new_session"])
            kill.assert_called_once_with(12345, signal.SIGKILL)
            self.assertEqual(process.wait.call_count, 2)

    def test_group_exit_race_preserves_timeout(self):
        with (
            patch("alambiq._native_process.subprocess.Popen") as popen,
            patch(
                "alambiq._native_process.os.killpg",
                side_effect=ProcessLookupError,
            ),
        ):
            process = popen.return_value
            process.wait.side_effect = [subprocess.TimeoutExpired("test", 10), 0]
            with self.assertRaises(subprocess.TimeoutExpired):
                run_logged(["test"], object(), timeout=10)
            self.assertEqual(process.wait.call_count, 2)
