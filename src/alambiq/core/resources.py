"""Pinned model and database resources.

A resource is a JSON record in `alambiq/resources/` naming each file it
consists of, with its size and SHA-256. The record is what a result depends on
when it says "computed with ESM C 300M" or "searched UniRef50". A changed
revision or file digest makes the resource a different resource, and a stored
result that used the earlier resource is not reused.

The identity here is derived from the record alone. `location` finds the files on
disk and `verify` checks them against the record, which a capability does at the
moment it reads them. Verifying hashes every file once; a receipt of each file's
size and modification time then lets later runs skip the hash until the file
changes. A database of 129 GB is therefore hashed once, not on every run.
Resources are provisioned by an explicit call before a run.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from importlib.resources import files
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024
RECEIPT_NOTE = (
    "Sizes and digests recorded after a full check. A file whose size or "
    "modification time differs from the receipt is hashed again."
)


class ResourceMissing(RuntimeError):
    """A pinned resource is not on this machine, or is not what the record says."""


def spec(name: str) -> dict[str, Any]:
    """The pinned record for a resource."""
    record = files("alambiq.resources").joinpath(f"{name}.json")
    if not record.is_file():
        raise KeyError(f"No resource {name!r} is pinned in alambiq/resources")
    return json.loads(record.read_text())


def names() -> list[str]:
    """The name of every pinned resource."""
    return sorted(
        item.name.removesuffix(".json")
        for item in files("alambiq.resources").iterdir()
        if item.name.endswith(".json") and item.is_file()
    )


def identity(name: str) -> str:
    """A digest of what the resource is: its version and every file's digest."""
    record = spec(name)
    material: dict[str, Any] = {
        "version": record.get("version"),
        "files": sorted((f["path"], f["sha256"]) for f in record["files"]),
    }
    if record.get("trees"):
        material["trees"] = sorted((t["path"], t["sha256"]) for t in record["trees"])
    return hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def content_address(name: str) -> str:
    """The directory name a resource lives under.

    Normally a digest of its files' records. A record may state its own address
    when the resource was installed under one before it was pinned here.
    """
    stated = spec(name).get("content_address")
    if stated:
        return stated
    records = [
        {k: f[k] for k in ("path", "bytes", "sha256")} for f in spec(name)["files"]
    ]
    trees = spec(name).get("trees", [])
    return hashlib.sha256(_canonical({"files": records, "trees": trees})).hexdigest()


def home() -> Path:
    """Where resources are kept: `ALAMBIQ_RESOURCE_HOME`, else `~/.alambiq/resources`."""
    configured = os.environ.get("ALAMBIQ_RESOURCE_HOME")
    base = Path(configured) if configured else Path.home() / ".alambiq" / "resources"
    return base.expanduser().resolve()


def location(name: str) -> Path:
    """Where a resource is expected, whether or not it is there.

    `ALAMBIQ_RESOURCE_<NAME>` (the name upper-cased, punctuation as
    underscores) overrides the default content-addressed place, for a resource
    that lives on another disk.
    """
    variable = "ALAMBIQ_RESOURCE_" + "".join(
        c if c.isalnum() else "_" for c in name.upper()
    )
    configured = os.environ.get(variable)
    if configured:
        return Path(configured).expanduser().resolve()
    return home() / "sha256" / content_address(name)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _receipt_path(name: str) -> Path:
    """Where the receipt of a full check is kept: beside the resources unless
    `ALAMBIQ_RESOURCE_RECEIPTS` names another directory (for a read-only store)."""
    configured = os.environ.get("ALAMBIQ_RESOURCE_RECEIPTS")
    base = Path(configured).expanduser().resolve() if configured else home() / "verified"
    return base / f"{content_address(name)}.json"


def _verify_tree(
    name: str, root: Path, tree: dict[str, Any], seen: dict[str, list[Any]]
) -> set[str]:
    """Check a directory of many files against one recorded digest of the whole.

    The digest is that of the sorted list of every file's relative path, size and
    SHA-256, so it covers names as well as contents. Each file is hashed once;
    after that its size and modification time stand in for the hash.
    """
    base = root / tree["path"]
    if base.is_symlink() or not base.is_dir():
        raise ResourceMissing(f"Resource {name}: {tree['path']} is not a directory")
    records, listed = [], set()
    for path in sorted(base.rglob("*")):
        if path.is_symlink():
            raise ResourceMissing(f"Resource {name}: {path} is a link, and links are not accepted")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ResourceMissing(f"Resource {name}: {path} is not a regular file")
        relative = str(path.relative_to(base))
        key = f"{tree['path']}/{relative}"
        stat = path.stat()
        known = seen.get(key)
        if known and known[:2] == [stat.st_size, stat.st_mtime_ns]:
            digest = known[2]
        else:
            digest = _hash(path)
            seen[key] = [stat.st_size, stat.st_mtime_ns, digest]
        records.append({"path": relative, "bytes": stat.st_size, "sha256": digest})
        listed.add(str(path.relative_to(root)))
    observed = {
        "file_count": len(records),
        "bytes": sum(r["bytes"] for r in records),
        "sha256": hashlib.sha256(_canonical(records)).hexdigest(),
    }
    expected = {k: tree[k] for k in observed}
    if observed != expected:
        raise ResourceMissing(
            f"Resource {name}: the files under {tree['path']} do not match the record "
            f"(found {observed['file_count']} files, {observed['bytes']} bytes; "
            f"the record states {expected['file_count']} files, {expected['bytes']} bytes)"
        )
    return listed


def verify(name: str) -> Path:
    """Check a resource's files against its record and return where it is.

    Every file must exist as a regular file, not a link, with the recorded size
    and SHA-256, and nothing else may sit beside them. Raises `ResourceMissing`
    saying what is wrong and how to provision it.
    """
    record = spec(name)
    root = location(name)
    if not root.is_dir():
        raise ResourceMissing(
            f"Resource {name} is not at {root}. Provision it with "
            f"`alambiq resource provision {name}`, or set "
            f"ALAMBIQ_RESOURCE_{''.join(c if c.isalnum() else '_' for c in name.upper())} "
            "to where it is."
        )
    receipt_file = _receipt_path(name)
    try:
        receipt = json.loads(receipt_file.read_text())
    except (OSError, json.JSONDecodeError):
        receipt = {}
    receipt = receipt if receipt.get("root") == str(root) else {"root": str(root)}
    seen: dict[str, list[int]] = receipt.setdefault("files", {})

    expected = {f["path"] for f in record["files"]}
    for f in record["files"]:
        path = root / f["path"]
        if path.is_symlink() or not path.is_file():
            raise ResourceMissing(
                f"Resource {name}: {f['path']} is missing or not a regular file"
            )
        stat = path.stat()
        if stat.st_size != f["bytes"]:
            raise ResourceMissing(
                f"Resource {name}: {f['path']} is {stat.st_size} bytes; the record states "
                f"{f['bytes']} bytes"
            )
        if seen.get(f["path"]) == [stat.st_size, stat.st_mtime_ns]:
            continue
        if _hash(path) != f["sha256"]:
            raise ResourceMissing(
                f"Resource {name}: {f['path']} does not match its recorded SHA-256"
            )
        seen[f["path"]] = [stat.st_size, stat.st_mtime_ns]
    tree_files: set[str] = set()
    for tree in record.get("trees", []):
        tree_files |= _verify_tree(name, root, tree, seen)
    present = {
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() or p.is_symlink()
    } - {"resource-lock.json"}
    extra = sorted(present - expected - tree_files)
    if extra:
        raise ResourceMissing(
            f"Resource {name} contains files that its record does not list: "
            f"{', '.join(extra[:5])}"
        )
    receipt["note"] = RECEIPT_NOTE
    # The receipt only saves hashing the files again. A store mounted read-only has no
    # place to keep it, and the check it records has passed, so failing to save it is not
    # a failure to verify; every later run then hashes the files again.
    try:
        receipt_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt_file.with_suffix(f".{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(receipt))
        temporary.replace(receipt_file)
    except OSError:
        pass
    return root


def provision(name: str) -> Path:
    """Download a resource whose record lists a URL for every file, then verify it.

    Called only by a person. Files are fetched into a staging directory beside
    the final place, checked against the record, and moved into place only when
    all of them match, so a failed download leaves nothing half-installed.
    """
    from urllib.request import Request, urlopen

    record = spec(name)
    if record.get("fetch"):
        return _provision_by_fetch(name, record)
    missing_url = [f["path"] for f in record["files"] if not f.get("url")]
    if missing_url or record.get("trees"):
        raise ResourceMissing(
            f"Resource {name} is built locally, not downloaded: {record.get('build', 'see its record')}"
        )
    target = location(name)
    if target.is_dir():
        return verify(name)
    staging = target.parent / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        for f in record["files"]:
            destination = staging / f["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            request = Request(
                f["url"], headers={"User-Agent": "Alambiq resource provisioning"}
            )
            with (
                urlopen(request, timeout=60) as response,
                destination.open("wb") as out,
            ):
                if not response.geturl().startswith("https://"):
                    raise ResourceMissing(
                        f"{f['url']} was redirected to an address that is not HTTPS"
                    )
                count = 0
                while chunk := response.read(CHUNK):
                    out.write(chunk)
                    count += len(chunk)
                    if count > f["bytes"]:
                        raise ResourceMissing(
                            f"{f['path']} is larger than the recorded {f['bytes']} bytes"
                        )
            if (
                destination.stat().st_size != f["bytes"]
                or _hash(destination) != f["sha256"]
            ):
                raise ResourceMissing(
                    f"{f['path']} downloaded from {f['url']} does not match its record "
                    "(size or SHA-256)"
                )
        (staging / "resource-lock.json").write_bytes(
            _canonical(
                {
                    "logical_id": record["logical_id"],
                    "version": record["version"],
                    "content_sha256": content_address(name),
                    "files": [
                        {k: f[k] for k in ("path", "bytes", "sha256")}
                        for f in record["files"]
                    ],
                }
            )
        )
        staging.rename(target)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise
    return verify(name)


def _provision_by_fetch(name: str, record: dict[str, Any]) -> Path:
    """Fetch a resource with the downloader of the tool that uses it, then verify it.

    Some models are obtained by the tool's own routine, which unpacks archives and names
    files its own way. The record says which environment and engine entry to run
    (`fetch.environment`, `fetch.entry`). The files are fetched into a staging directory
    beside the final place, moved into place, and verified against the record; a directory
    that does not verify is removed, so nothing unverified is left where a run would find it.
    """
    import shutil

    from alambiq.core import envs

    target = location(name)
    if target.is_dir():
        return verify(name)
    fetch = record["fetch"]
    staging = target.parent / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    try:
        envs.run_in_environment(
            fetch["environment"],
            fetch["entry"],
            {"target": str(staging)},
            timeout=6 * 3600,
        )
        (staging / "resource-lock.json").write_bytes(
            _canonical(
                {
                    "logical_id": record["logical_id"],
                    "version": record["version"],
                    "content_sha256": content_address(name),
                    "files": [
                        {k: f[k] for k in ("path", "bytes", "sha256")}
                        for f in record["files"]
                    ],
                }
            )
        )
        staging.rename(target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        return verify(name)
    except ResourceMissing:
        shutil.rmtree(target, ignore_errors=True)
        raise
