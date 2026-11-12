"""Download one structure file from the RCSB Protein Data Bank or the AlphaFold Protein Structure Database.

Returns the file with the URL it came from, the time of retrieval, its size and its SHA-256.
The file is a deposited structure (RCSB, by PDB entry identifier; Berman et al., 2000) or a
predicted model (AlphaFold DB, by UniProt accession; Jumper et al., 2021; Bertoni et al.,
2026), in mmCIF or PDB format.

The method:

- For `rcsb`, the capability reads `<base>/<ID>.cif` or `<base>/<ID>.pdb` from
  `https://files.rcsb.org/download`.
- For `alphafold`, the capability reads the prediction listing of the accession from
  `https://alphafold.ebi.ac.uk/api/prediction/<accession>` and downloads the file named in its
  `cifUrl` or `pdbUrl` field. The result records the file URL, which carries the model version.
- The body must be text in the requested format: mmCIF with atom records, or PDB with ATOM or
  HETATM records. A page of another kind is refused.
- The file is stored with its SHA-256. A step with the same request in the same run directory
  reuses the stored file and does not download it again.

Limits.

- The capability connects to the network only when `allow_network` is true; the request is
  refused before any connection otherwise.
- The files are those of the archive at the time of retrieval. An archive can revise an entry
  or a model; the SHA-256 and the retrieval time identify the file used.
- A URL must use https, or http on the loopback address for a mirror on the same machine.
- A response of more than `max_megabytes` is refused.
- Terms of use and citation requirements belong to the archive that serves the file.

References.

- Berman HM, Westbrook J, Feng Z, Gilliland G, Bhat TN, Weissig H, Shindyalov IN, Bourne PE.
  The Protein Data Bank. Nucleic Acids Res 28, 235-242 (2000). https://doi.org/10.1093/nar/28.1.235
- Jumper J, et al. Highly accurate protein structure prediction with AlphaFold. Nature 596,
  583-589 (2021). https://doi.org/10.1038/s41586-021-03819-2
- Bertoni D, et al. AlphaFold Protein Structure Database 2025: a redesigned interface and
  updated structural coverage. Nucleic Acids Res 54, D358-D362 (2026).
  https://doi.org/10.1093/nar/gkaf1226
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from alambiq.capabilities.characterize.structure_fetch_report import render
from alambiq.core.context import StepContext
from alambiq.core.registry import capability


@dataclass(frozen=True, slots=True)
class FetchRequest:
    """The archive, the entry, the format, and the permission to use the network.

    source:
        `rcsb` (deposited structures, by PDB entry identifier) or `alphafold` (predicted models,
        by UniProt accession).
    identifier:
        A PDB entry identifier of 4 characters, such as `1L2Y`, for `rcsb`; a UniProt accession,
        such as `P69905`, for `alphafold`. Letters are read in either case.
    format:
        `mmcif` or `pdb`; default `mmcif`.
    allow_network:
        Whether the step may connect to the archive; default false. The request is refused
        without a connection when it is false.
    timeout_seconds:
        The limit on each connection, 1 to 600; default 60.
    max_megabytes:
        The largest response accepted, 1 to 1,000 MB; default 100.
    base_url:
        A mirror that serves the same paths, replacing the archive's address: an https URL, or an
        http URL on the loopback address. Default: the archive's own address.
    """

    source: Literal["rcsb", "alphafold"]
    identifier: str
    format: Literal["mmcif", "pdb"] = "mmcif"
    allow_network: bool = False
    timeout_seconds: int = 60
    max_megabytes: int = 100
    base_url: str | None = None


@dataclass(frozen=True, slots=True)
class FetchResult:
    """The stored file and where it came from."""

    structure: str
    """The stored mmCIF or PDB file."""
    source: str
    identifier: str
    format: str
    url: str
    """The URL of the file; for AlphaFold DB, the file URL named by the prediction listing."""
    retrieved_utc: str
    bytes: int
    sha256: str
    last_modified: str | None
    """The Last-Modified header of the response; None when the server sent none."""


@capability(
    "structure.fetch",
    version="1.0",
    family="characterize",
    summary="Download a structure file from the RCSB PDB or the AlphaFold database, with its provenance",
    render=render,
)
def fetch(request: FetchRequest, ctx: StepContext) -> FetchResult:
    """Check the request, download the file, and keep it with its URL, time and SHA-256."""
    from alambiq.capabilities.characterize.science.fetch import (
        check_request,
        request_url,
    )
    from alambiq.capabilities.characterize.science.fetch import fetch as download

    check_request(
        source=request.source,
        identifier=request.identifier,
        fmt=request.format,
        timeout=request.timeout_seconds,
        max_megabytes=request.max_megabytes,
        base_url=request.base_url,
    )
    if not request.allow_network:
        raise ValueError(
            f"structure.fetch downloads from "
            f"{request_url(request.source, request.identifier, request.format, request.base_url)}; "
            "set allow_network: true to permit the connection"
        )
    found = download(
        source=request.source,
        identifier=request.identifier,
        fmt=request.format,
        timeout=request.timeout_seconds,
        max_megabytes=request.max_megabytes,
        base_url=request.base_url,
    )
    media = "chemical/x-mmcif" if request.format == "mmcif" else "chemical/x-pdb"
    stored = ctx.keep_text(found["text"], media)
    return FetchResult(
        structure=str(stored.path),
        source=request.source,
        identifier=request.identifier.upper(),
        format=request.format,
        url=found["url"],
        retrieved_utc=found["retrieved_utc"],
        bytes=found["bytes"],
        sha256=found["sha256"],
        last_modified=found["last_modified"],
    )
