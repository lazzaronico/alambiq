"""The command line.

alambiq run campaign.yaml --report report.html
alambiq validate campaign.yaml
alambiq history RUN_DIR
alambiq replay RUN_DIR STEP --out replay-dir
alambiq capability list
alambiq capability show assay.fit
alambiq env check openmm
alambiq resource list
alambiq resource show uniref50-2026_02
alambiq resource verify esmc-300m
alambiq resource provision thermompnn --yes
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import alambiq
from alambiq import api
from alambiq.core import envs, resources


def _run(args: argparse.Namespace) -> int:
    outcome = api.run(args.campaign, run_dir=args.run_dir, fresh=args.fresh)
    for record in outcome.steps.values():
        state = "reused" if record.reused else record.status
        print(f"{record.step_id:24s} {state:10s} {record.seconds:6.2f}s")
        for note in record.notes:
            print(f"  note: {note}")
        if record.error:
            print(f"  error: {record.error}", file=sys.stderr)
    if args.report:
        print(f"report: {outcome.report(args.report)}")
    print(f"artifacts: {outcome.run_dir}")
    return 0 if outcome.ok else 1


def _validate(args: argparse.Namespace) -> int:
    problems = api.validate(args.campaign)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print("campaign is valid")
    return 1 if problems else 0


def _history(args: argparse.Namespace) -> int:
    rows = api.history(args.run_dir)
    if not rows:
        print(f"{args.run_dir} holds no recorded runs", file=sys.stderr)
        return 1
    for row in rows:
        print(
            f"{row['started']}  {row['campaign']}  "
            f"{'complete' if row['ok'] else 'incomplete'}  "
            f"{row['computed']} computed, {row['reused']} reused, {row['failed']} failed"
            + (f", {row['skipped']} not run" if row["skipped"] else "")
            + f"  {row['seconds']:.1f}s"
        )
    return 0


def _replay(args: argparse.Namespace) -> int:
    try:
        target = api.replay(args.run_dir, args.step, args.out)
    except (FileNotFoundError, KeyError, ValueError) as error:
        print(error.args[0] if isinstance(error, KeyError) else error, file=sys.stderr)
        return 1
    print(f"wrote {target}; run it with: alambiq run {target}")
    return 0


def _capability(args: argparse.Namespace) -> int:
    if args.action == "list":
        rows = api.capability_list()
        width = max((len(r["id"]) for r in rows), default=0)
        for row in rows:
            tail = f"  [{row['environment']}]" if row["environment"] else ""
            print(f"{row['id']:<{width}}  {row['family']:<12} {row['summary']}{tail}")
        return 0
    described = api.describe(args.id)
    if args.json:
        print(json.dumps(described, indent=2))
        return 0
    print(f"{described['id']}  ({described['family']}, version {described['version']})")
    print(described["summary"])
    if described["environment"]:
        print(f"environment: {described['environment']}")
    if described["resources"]:
        print(f"resources: {', '.join(described['resources'])}")
    if described["presets"]:
        print(f"presets: {', '.join(described['presets'])}")
    print("\nrequest:")
    for field in described["request"]:
        print(f"  {field['name']:<20} {field['type']:<22} {field['default']}")
    print("\nresult:")
    for field in described["result"]:
        print(f"  {field['name']:<20} {field['type']}")
    if described["documentation"]:
        print("\n" + described["documentation"])
    return 0


def _env(args: argparse.Namespace) -> int:
    names = [args.name] if args.name else sorted(envs.ENVIRONMENTS)
    failures = 0
    for name in names:
        report = envs.spec(name).check()
        mark = "ready" if report["ready"] else "not ready"
        print(f"{name:10s} {mark}")
        if not report["ready"]:
            failures += 1
            print(
                f"  {report.get('reason') or 'missing: ' + ', '.join(report['missing'])}"
            )
            print(f"  recipe: {report['recipe']}")
        else:
            for tool, version in (report.get("versions") or {}).items():
                print(f"  {tool} {version}")
    return 1 if failures else 0


def _megabytes(count: int) -> str:
    return f"{count / 1e6:,.1f} MB" if count < 1e9 else f"{count / 1e9:,.2f} GB"


def _resource(args: argparse.Namespace) -> int:
    if args.action == "list":
        for name in resources.names():
            record = resources.spec(name)
            size = sum(f["bytes"] for f in record["files"])
            state = "installed" if resources.location(name).is_dir() else "absent"
            print(
                f"{name:24s} {record['version'][:14]:14s} {record['license']:28s} "
                f"{_megabytes(size):>10s}  {state}"
            )
        return 0
    if not args.name:
        print(f"resource {args.action} needs a resource name", file=sys.stderr)
        return 2
    try:
        record = resources.spec(args.name)
    except KeyError as error:
        print(error.args[0], file=sys.stderr)
        return 2
    size = sum(f["bytes"] for f in record["files"])
    if args.action == "show":
        place = resources.location(args.name)
        print(f"{args.name}  ({record['logical_id']}, version {record['version']})")
        print(f"provider: {record.get('provider', 'unstated')}")
        print(f"license:  {record['license']}")
        print(f"size:     {_megabytes(size)} in {len(record['files'])} files")
        print(f"location: {place} ({'installed' if place.is_dir() else 'absent'})")
        print(f"identity: {resources.identity(args.name)}")
        if record.get("description"):
            print(record["description"])
        if record.get("build"):
            print(f"obtained by: {record['build']}")
        return 0
    try:
        if args.action == "verify":
            print(f"verified: {resources.verify(args.name)}")
            return 0
        if args.action == "provision" and resources.location(args.name).is_dir():
            print(f"already installed and verified: {resources.verify(args.name)}")
            return 0
        fetch = record.get("fetch")
        downloaded = [f for f in record["files"] if f.get("url")]
        if not fetch and len(downloaded) != len(record["files"]):
            print(
                f"{args.name} is built locally and is not downloaded: "
                f"{record.get('build', 'see its record')}",
                file=sys.stderr,
            )
            return 1
        if not args.yes:
            built = record.get("source_files", [])
            hosts = sorted(
                {u.split("/")[2] for u in record.get("sources", [])}
                | {f["url"].split("/")[2] for f in downloaded}
                | {s["url"].split("/")[2] for s in built}
            )
            if built:
                amount = (
                    f"download {_megabytes(sum(s['bytes'] for s in built))} from "
                    f"{', '.join(hosts)} and build {_megabytes(size)}"
                )
            else:
                amount = f"download {_megabytes(size)} from {', '.join(hosts)}"
            print(
                f"Would {amount} ({record['license']}) into "
                f"{resources.location(args.name)}. Run again with --yes to proceed."
            )
            return 2
        print(f"provisioned: {resources.provision(args.name)}")
        return 0
    except resources.ResourceMissing as error:
        print(error, file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alambiq", description=__doc__)
    parser.add_argument("--version", action="version", version=alambiq.__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a campaign")
    run.add_argument("campaign", type=Path)
    run.add_argument("--run-dir", type=Path, default=None)
    run.add_argument("--report", type=Path, default=None)
    run.add_argument(
        "--fresh", action="store_true", help="recompute every step and ignore the step cache"
    )
    run.set_defaults(handler=_run)

    validate = sub.add_parser("validate", help="check a campaign without running it")
    validate.add_argument("campaign", type=Path)
    validate.set_defaults(handler=_validate)

    hist = sub.add_parser("history", help="list the runs recorded in a run directory")
    hist.add_argument("run_dir", type=Path)
    hist.set_defaults(handler=_history)

    replay = sub.add_parser(
        "replay",
        help="write one step of a recorded run as a campaign that runs it again",
    )
    replay.add_argument("run_dir", type=Path)
    replay.add_argument("step")
    replay.add_argument("--out", type=Path, required=True, help="directory to write the campaign and its input files to")
    replay.set_defaults(handler=_replay)

    cap = sub.add_parser("capability", help="list capabilities or show one")
    cap.add_argument("action", choices=["list", "show"])
    cap.add_argument("id", nargs="?")
    cap.add_argument("--json", action="store_true")
    cap.set_defaults(handler=_capability)

    env = sub.add_parser("env", help="check engine environments")
    env.add_argument("action", choices=["check", "list"])
    env.add_argument("name", nargs="?")
    env.set_defaults(handler=_env)

    resource = sub.add_parser(
        "resource", help="list, check or install pinned models and databases"
    )
    resource.add_argument("action", choices=["list", "show", "verify", "provision"])
    resource.add_argument("name", nargs="?")
    resource.add_argument(
        "--yes", action="store_true", help="confirm a download (provision only)"
    )
    resource.set_defaults(handler=_resource)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "capability" and args.action == "show" and not args.id:
        print("capability show needs a capability id", file=sys.stderr)
        return 2
    try:
        return int(args.handler(args))
    except (KeyError, ValueError, OSError) as error:
        # A campaign that cannot be read or planned (an unknown capability, a step that
        # binds to one that is not there, a cycle, a missing file) is a mistake in what
        # the researcher wrote; it is reported as one, without a traceback. Set
        # ALAMBIQ_DEBUG to see the traceback instead.
        if os.environ.get("ALAMBIQ_DEBUG"):
            raise
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"error: {message}", file=sys.stderr)
        return 2


def _stop_temporary_server() -> None:
    """Stop the temporary Prefect server that this process started, if it started one.

    Prefect starts the server as a child process and stops it only when Python exits
    normally, so a program ended by a signal, or one whose server did not finish starting,
    would leave it running.
    """
    module = sys.modules.get("prefect.server.api.server")
    servers = getattr(getattr(module, "SubprocessASGIServer", None), "_instances", {})
    for server in list(servers.values()):
        try:
            server.stop()
        except Exception:  # noqa: BLE001 - leaving must not hide the real outcome
            pass


def entry(argv: list[str] | None = None) -> int:
    """Run `main` as a program: the temporary server ends with the program.

    A hang-up (a closed terminal or a dropped SSH connection) or a termination request
    ends the program as an exit, which stops the server; an interrupt already does. Only
    a kill that cannot be caught leaves the server running.
    """

    def leave(number: int, frame: object) -> None:
        raise SystemExit(128 + number)

    for name in ("SIGTERM", "SIGHUP"):
        number = getattr(signal, name, None)
        if number is not None:
            signal.signal(number, leave)
    try:
        return main(argv)
    finally:
        _stop_temporary_server()


if __name__ == "__main__":
    raise SystemExit(entry())
