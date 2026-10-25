"""Owned native commands on supported POSIX hosts."""

import locale
import os
import signal
import subprocess
import tempfile
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

_active_run = ContextVar("alambiq_native_processes", default=None)


@dataclass
class _OwnedProcesses:
    processes: set = field(default_factory=set)
    lock: object = field(default_factory=threading.RLock)
    closed: bool = False


def _stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


@contextmanager
def native_process_scope():
    """Forward main-thread interruption to this run's live native commands.

    Prefect copies the calling context into task threads. This short-lived set
    holds the Popen handles of the commands this run started.
    """
    owner = _OwnedProcesses()
    token = _active_run.set(owner)
    previous = {}

    def relay(signum, frame):
        with owner.lock:
            owner.closed = True
            for process in owner.processes:
                _stop(process)
        handler = previous[signum]
        if callable(handler):
            handler(signum, frame)
        else:
            signal.signal(signum, handler)
            signal.raise_signal(signum)

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handler = signal.getsignal(signum)
                if handler != signal.SIG_IGN:
                    previous[signum] = handler
                    signal.signal(signum, relay)
        yield
    finally:
        with owner.lock:
            owner.closed = True
            for process in owner.processes:
                _stop(process)
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        _active_run.reset(token)


def run_logged(command, log, *, timeout, cwd=None, env=None, stderr=subprocess.STDOUT):
    """Retire owned descendants on completion, timeout or direct interruption.

    Output goes to the caller's file. Each command runs in its own session and process
    group, and retirement signals that group; a program that detaches into another
    session leaves the group. The surrounding run scope relays main-thread signals.
    """
    owner = _active_run.get() or _OwnedProcesses()
    if (
        _active_run.get() is None
        and threading.current_thread() is threading.main_thread()
    ):
        # Explicit resource builders also call this outside a Prefect campaign.
        with native_process_scope():
            return run_logged(
                command, log, timeout=timeout, cwd=cwd, env=env, stderr=stderr
            )
    with owner.lock:
        if owner.closed:
            raise RuntimeError("Native command cannot start after its run scope closed")
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=stderr,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        owner.processes.add(process)
    try:
        returncode = process.wait(timeout=timeout)
    except BaseException:
        _stop(process)
        process.wait()
        raise
    else:
        # A tool can exit while children remain, including after an inner timeout.
        # Retire this command's group before releasing ownership or reading output.
        _stop(process)
    finally:
        with owner.lock:
            owner.processes.discard(process)
    return subprocess.CompletedProcess(command, returncode)


def run_captured(command, *, timeout, cwd=None, env=None):
    """Separate text streams with the same owned-group lifetime as logged tools.

    Temporary files avoid pipe-buffer deadlocks, including a child inheriting
    its parent's descriptors. Timeout output stays bytes, like subprocess.run;
    successful streams use the host text encoding and universal newlines.
    """
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        try:
            result = run_logged(
                command, stdout, stderr=stderr, timeout=timeout, cwd=cwd, env=env
            )
        except subprocess.TimeoutExpired as exc:
            stdout.seek(0)
            stderr.seek(0)
            exc.output = stdout.read()
            exc.stderr = stderr.read()
            raise
        except OSError as exc:
            # A completed command can still fail owned-group cleanup. Preserve
            # its streams for the caller's failure receipt without hiding it.
            stdout.seek(0)
            stderr.seek(0)
            exc.output = stdout.read()
            exc.stderr = stderr.read()
            raise
        stdout.seek(0)
        stderr.seek(0)

        def text(stream):
            return (
                stream.read()
                .decode(locale.getpreferredencoding(False))
                .replace("\r\n", "\n")
                .replace("\r", "\n")
            )

        return subprocess.CompletedProcess(
            command, result.returncode, text(stdout), text(stderr)
        )
