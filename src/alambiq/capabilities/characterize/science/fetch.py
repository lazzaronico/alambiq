"""Download of one structure file from a public archive, with the checks that identify what arrived."""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULTS = {
    "rcsb": "https://files.rcsb.org/download",
    "alphafold": "https://alphafold.ebi.ac.uk/api/prediction",
}
PDB_ID = re.compile(r"[0-9][A-Za-z0-9]{3}")
UNIPROT = re.compile(
    r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}"
)
LOOPBACK = {"localhost", "127.0.0.1", "::1"}
EXTENSIONS = {"mmcif": "cif", "pdb": "pdb"}


def check_request(
    *,
    source: str,
    identifier: str,
    fmt: str,
    timeout: int,
    max_megabytes: int,
    base_url: str | None,
) -> None:
    if source not in DEFAULTS:
        raise ValueError(
            f"source is {source!r}; one of {', '.join(DEFAULTS)} is required"
        )
    if fmt not in EXTENSIONS:
        raise ValueError(f"format is {fmt!r}; mmcif or pdb is required")
    if source == "rcsb" and not PDB_ID.fullmatch(identifier):
        raise ValueError(
            f"identifier is {identifier!r}; a PDB entry identifier of 4 characters, a digit then "
            "3 letters or digits, is required for source rcsb"
        )
    if source == "alphafold" and not UNIPROT.fullmatch(identifier.upper()):
        raise ValueError(
            f"identifier is {identifier!r}; a UniProt accession such as P69905 is required for "
            "source alphafold"
        )
    for name, value, low, high in (
        ("timeout_seconds", timeout, 1, 600),
        ("max_megabytes", max_megabytes, 1, 1000),
    ):
        if not low <= value <= high:
            raise ValueError(
                f"{name} is {value}; an integer from {low} to {high} is required"
            )
    if base_url is not None and not allowed_url(base_url):
        raise ValueError(
            f"base_url is {base_url!r}; an https URL, or an http URL on the loopback address, is required"
        )


def allowed_url(url: str) -> bool:
    """An https URL, or an http URL on the loopback address."""
    parts = urllib.parse.urlsplit(url)
    if not parts.hostname:
        return False
    return parts.scheme == "https" or (
        parts.scheme == "http" and parts.hostname in LOOPBACK
    )


def request_url(source: str, identifier: str, fmt: str, base_url: str | None) -> str:
    base = (base_url or DEFAULTS[source]).rstrip("/")
    if source == "rcsb":
        return f"{base}/{identifier.upper()}.{EXTENSIONS[fmt]}"
    return f"{base}/{identifier.upper()}"


def get(url: str, *, timeout: int, max_bytes: int) -> tuple[bytes, str | None]:
    """The body of a URL and its Last-Modified header; the body is at most `max_bytes`."""
    request = urllib.request.Request(
        url, headers={"User-Agent": "alambiq-structure-fetch"}
    )
    opener = urllib.request.build_opener(
        urllib.request.HTTPSHandler, urllib.request.HTTPHandler
    )
    try:
        with opener.open(request, timeout=timeout) as response:
            body = response.read(max_bytes + 1)
            modified = response.headers.get("Last-Modified")
    except urllib.error.HTTPError as exc:
        raise ValueError(
            f"{url} answered with HTTP status {exc.code} {exc.reason}"
        ) from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        reason = getattr(exc, "reason", exc)
        raise ValueError(
            f"{url} could not be read within {timeout} s: {reason}"
        ) from exc
    if len(body) > max_bytes:
        raise ValueError(
            f"{url} holds more than {max_bytes // 1_000_000} MB; raise max_megabytes to read it"
        )
    return body, modified


def alphafold_file_url(listing: bytes, identifier: str, fmt: str, url: str) -> str:
    """The file URL that the prediction API names for the accession."""
    try:
        entries = json.loads(listing)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{url} did not return JSON") from exc
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"AlphaFold DB holds no prediction for {identifier}")
    entry = next(
        (
            e
            for e in entries
            if isinstance(e, dict) and e.get("uniprotAccession") == identifier.upper()
        ),
        entries[0],
    )
    field = "cifUrl" if fmt == "mmcif" else "pdbUrl"
    value = entry.get(field) if isinstance(entry, dict) else None
    if not isinstance(value, str) or not value:
        raise ValueError(f"The prediction listing of {identifier} has no {field} field")
    return value


def check_content(body: bytes, fmt: str, url: str) -> str:
    """The text of the file, after a check that it is the format requested."""
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{url} did not return text") from exc
    if fmt == "mmcif":
        first = next(
            (ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")),
            "",
        )
        if not first.startswith("data_") or "_atom_site" not in text:
            raise ValueError(f"{url} does not hold mmCIF with atom records")
    elif not any(ln.startswith(("ATOM  ", "HETATM")) for ln in text.splitlines()):
        raise ValueError(f"{url} does not hold PDB atom records")
    return text


def fetch(
    *,
    source: str,
    identifier: str,
    fmt: str,
    timeout: int,
    max_megabytes: int,
    base_url: str | None,
) -> dict[str, Any]:
    max_bytes = max_megabytes * 1_000_000
    url = request_url(source, identifier, fmt, base_url)
    modified = None
    if source == "alphafold":
        listing, _ = get(url, timeout=timeout, max_bytes=max_bytes)
        url = alphafold_file_url(listing, identifier, fmt, url)
        if not allowed_url(url):
            raise ValueError(
                f"The prediction listing of {identifier} names {url!r}; an https URL is required"
            )
    body, modified = get(url, timeout=timeout, max_bytes=max_bytes)
    text = check_content(body, fmt, url)
    return {
        "url": url,
        "text": text,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "last_modified": modified,
        "retrieved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
