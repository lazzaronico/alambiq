"""Captured command lifetime without scientific tools or models."""

import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alambiq import _native_process as native


@unittest.skipUnless(os.name == "posix", "Mac/Linux native profiles")
class NativeCaptureTests(unittest.TestCase):
    def test_separate_streams_nonzero_and_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            result = native.run_captured(
                [
                    sys.executable,
                    "-c",
                    "import os,sys;print(os.getcwd());print(os.environ['CAPTURE_TEST']);print('error',file=sys.stderr);sys.exit(3)",
                ],
                timeout=5,
                cwd=folder,
                env={**os.environ, "CAPTURE_TEST": "kept"},
            )
            self.assertEqual(result.returncode, 3)
            self.assertIn(str(Path(folder).resolve()), result.stdout)
            self.assertIn("kept", result.stdout)
            self.assertEqual(result.stderr, "error\n")

    def test_large_output_does_not_block_on_pipe_buffers(self):
        result = native.run_captured(
            [
                sys.executable,
                "-c",
                "import sys;sys.stdout.write('x'*200000);sys.stderr.write('y'*200000)",
            ],
            timeout=5,
        )
        self.assertEqual(result.stdout, "x" * 200000)
        self.assertEqual(result.stderr, "y" * 200000)

    def test_timeout_retains_bytes_and_stops_child(self):
        child = "import time;time.sleep(60)"
        command = [
            sys.executable,
            "-u",
            "-c",
            f"import subprocess,sys,time;p=subprocess.Popen([sys.executable,'-c',{child!r}]);print(p.pid,flush=True);print('partial error',file=sys.stderr,flush=True);time.sleep(60)",
        ]
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            native.run_captured(command, timeout=1)
        exc = caught.exception
        self.assertIsInstance(exc.stdout, bytes)
        self.assertEqual(exc.stderr, b"partial error\n")
        pid = int(exc.stdout.strip())
        try:
            state = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
            ).stdout.strip()
            self.assertTrue(not state or state.startswith("Z"), state)
        finally:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_cleanup_permission_failure_preserves_complete_streams(self):
        def deny_cleanup(process):
            self.assertEqual(process.returncode, 0)
            raise PermissionError(1, "owned group cleanup denied")

        with patch("alambiq._native_process._stop", side_effect=deny_cleanup):
            with self.assertRaises(PermissionError) as caught:
                native.run_captured(
                    [
                        sys.executable,
                        "-c",
                        "import sys;print('complete result');print('native diagnostic',file=sys.stderr)",
                    ],
                    timeout=5,
                )
        self.assertEqual(caught.exception.output, b"complete result\n")
        self.assertEqual(caught.exception.stderr, b"native diagnostic\n")

    def test_launch_failure_is_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                native.run_captured([str(Path(folder) / "absent")], timeout=1)
