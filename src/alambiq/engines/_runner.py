"""Run one engine function on a JSON request.

This file is executed by path inside whichever environment holds the engine,
and it is never imported:

    python -I _runner.py module function request.json result.json

It reads the request, calls `function` from `module.py` beside it, and writes
the returned value as JSON. Because the module is loaded by file, the
environment needs nothing from Alambiq installed. It also imports the standard
library only, like every module in this directory, so what crosses the boundary
is plain data.

An engine module may import its siblings in this directory by name.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: _runner.py module function request.json result.json",
            file=sys.stderr,
        )
        return 2
    module_name, function_name, request_path, result_path = argv

    if not module_name.isidentifier() or module_name.startswith("_"):
        print(f"{module_name!r} is not an engine module name", file=sys.stderr)
        return 2
    source = HERE / f"{module_name}.py"
    if not source.is_file():
        print(f"no engine module {module_name!r} in {HERE}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location(f"engine_{module_name}", source)
    if spec is None or spec.loader is None:
        print(f"cannot load engine module {module_name!r}", file=sys.stderr)
        return 2
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    function = getattr(module, function_name, None)
    if not callable(function):
        print(f"{module_name} has no function {function_name!r}", file=sys.stderr)
        return 2

    request = json.loads(Path(request_path).read_text())
    try:
        result = function(request)
    except (RuntimeError, ValueError) as error:
        # An engine that refuses (a tool missing, input it cannot use) says why in a
        # sentence. That sentence is written beside the result so the caller can report
        # it without the engine's traceback; any other exception keeps its traceback.
        Path(result_path + ".error").write_text(
            json.dumps({"type": type(error).__name__, "message": str(error)})
        )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    Path(result_path).write_text(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
