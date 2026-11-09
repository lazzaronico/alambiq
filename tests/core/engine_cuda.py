"""Whether the Python of an engine environment sees a CUDA device."""

from __future__ import annotations

import subprocess


def engine_has_cuda(python: str) -> bool:
    """True when `torch.cuda.is_available()` holds under the given interpreter."""
    done = subprocess.run(
        [
            python,
            "-I",
            "-c",
            "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)",
        ],
        capture_output=True,
    )
    return done.returncode == 0
