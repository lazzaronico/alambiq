"""Command completion and nested failures retire the owned process group."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from alambiq._native_process import run_captured, run_logged


@unittest.skipUnless(os.name == "posix", "Mac/Linux native process groups")
class NativeDescendantTests(unittest.TestCase):
    def assert_child_stopped(self, marker):
        pid = int(marker.read_text())
        try:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True,
                text=True,
            ).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), state)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_logged_and_captured_leader_exit_retires_surviving_child(self):
        for captured in (False, True):
            for code in (0, 7):
                with self.subTest(captured=captured, code=code):
                    with tempfile.TemporaryDirectory() as directory:
                        marker = Path(directory) / "child.pid"
                        program = (
                            "import pathlib,subprocess,sys;"
                            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);"
                            f"pathlib.Path({str(marker)!r}).write_text(str(p.pid));"
                            f"print('leader complete',flush=True);sys.exit({code})"
                        )
                        command = [sys.executable, "-u", "-c", program]
                        if captured:
                            result = run_captured(command, timeout=5)
                            self.assertEqual(result.stdout, "leader complete\n")
                        else:
                            log = Path(directory) / "command.log"
                            with log.open("w") as stream:
                                result = run_logged(command, stream, timeout=5)
                            self.assertEqual(log.read_text(), "leader complete\n")
                        self.assertEqual(result.returncode, code)
                        self.assert_child_stopped(marker)
