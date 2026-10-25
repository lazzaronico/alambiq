import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


@unittest.skipUnless(os.name == "posix", "Mac/Linux signal semantics")
class NativeSignalTests(unittest.TestCase):
    def test_prefect_captured_interrupt_stops_owned_group(self):
        self.check_signals("prefect-captured")

    def test_direct_captured_interrupt_stops_owned_group(self):
        self.check_signals("direct-captured")

    def test_campaign_interrupt_stops_owned_group(self):
        self.check_signals("prefect")

    def test_direct_logged_interrupt_stops_owned_group(self):
        self.check_signals("direct")

    def check_signals(self, mode):
        for sig in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=sig), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                pids = []
                with (folder / "driver.log").open("w") as log:
                    driver = subprocess.Popen(
                        [
                            sys.executable,
                            str(Path(__file__).with_name("native_signal_fixture.py")),
                            tmp,
                            mode,
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env={
                            **os.environ,
                            "ALAMBIQ_PREFECT_HOME": str(folder / "prefect"),
                        },
                    )
                    try:
                        deadline = time.monotonic() + 30
                        while time.monotonic() < deadline and driver.poll() is None:
                            path = folder / "pids.log"
                            lines = (
                                path.read_text().splitlines() if path.exists() else []
                            )
                            if len(lines) == 2:
                                records = [json.loads(line) for line in lines]
                                pids = [records[0]["parent"], records[1]["child"]]
                                break
                            time.sleep(0.05)
                        self.assertEqual(
                            len(pids), 2, (folder / "driver.log").read_text()
                        )
                        driver.send_signal(sig)
                        try:
                            status = driver.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            self.fail(
                                "Interrupted Prefect flow waited for native timeout instead of terminating its child group"
                            )
                        self.assertNotEqual(status, 0)
                        states = [
                            subprocess.run(
                                ["ps", "-o", "stat=", "-p", str(pid)],
                                capture_output=True,
                                text=True,
                            ).stdout.strip()
                            for pid in pids
                        ]
                        self.assertTrue(
                            all(not state or state.startswith("Z") for state in states),
                            states,
                        )
                    finally:
                        for pid in pids:
                            try:
                                os.kill(pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        if driver.poll() is None:
                            driver.kill()
                        driver.wait()
