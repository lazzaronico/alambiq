"""Hints for failures an engine can recognize from a tool's own message.

Imports the standard library only, like every module in this directory.
"""

from __future__ import annotations

NOEXEC_MARKERS = (
    "with error 13",
    "failed to map segment",
    "cannot execute",
    "Permission denied",
)


def exec_hint(text: str) -> str:
    """What to do when a tool cannot run a file it wrote to the temporary directory.

    MMseqs2 writes and runs helper scripts there, and Java maps a library it unpacks there;
    a container's temporary file system is mounted without the right to execute by default.
    """
    if any(marker in text for marker in NOEXEC_MARKERS):
        return (
            "\nThe tool could not run a file it wrote to the temporary directory. In a "
            "container, mount that directory with execute permission: "
            "`--tmpfs /tmp:rw,exec,nosuid,nodev,mode=1777,size=4g` "
            "(see environments/docker/README.md)."
        )
    return ""
