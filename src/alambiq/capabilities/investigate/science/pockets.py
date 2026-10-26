"""Reading P2Rank's tables and joining its pockets to a protein record.

Pure functions over text. P2Rank names a residue by its author chain and author
residue number with any insertion code; the join to a `protein.inspect` record
builds those keys exactly from the recorded author numbering.
"""

from __future__ import annotations

import csv
import io
import math
from collections import defaultdict
from typing import Any


def _float(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"P2Rank output field {key!r} is missing or not a number"
        ) from error
    if not math.isfinite(value):
        raise ValueError(
            f"P2Rank output field {key!r} is not a finite number"
        )
    return value


def _integer(row: dict[str, str], key: str) -> int:
    value = _float(row, key)
    if value != int(value):
        raise ValueError(
            f"P2Rank output field {key!r} is not an integer"
        )
    return int(value)


def parse_pockets(text: str, maximum: int) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text, newline=""), skipinitialspace=True)
    required = {
        "name",
        "rank",
        "score",
        "probability",
        "sas_points",
        "surf_atoms",
        "center_x",
        "center_y",
        "center_z",
        "residue_ids",
    }
    missing = required - {key.strip() for key in reader.fieldnames or []}
    if missing:
        raise ValueError(
            f"The header of the P2Rank predictions table lacks the columns {sorted(missing)}"
        )
    rows = [{str(k).strip(): str(v).strip() for k, v in row.items()} for row in reader]
    pockets = []
    for row in rows[:maximum]:
        probability = _float(row, "probability")
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"P2Rank reports probability {probability} for pocket {row.get('name', '')}; "
                "a value from 0 to 1 is required"
            )
        residues = [item for item in row.get("residue_ids", "").split() if item]
        if not residues:
            raise ValueError(
                f"P2Rank pocket {row.get('name', '')} lists no residues; "
                "at least 1 residue is required"
            )
        if _integer(row, "sas_points") < 0 or _integer(row, "surf_atoms") < 0:
            raise ValueError(
                f"P2Rank reports a negative point or atom count for pocket "
                f"{row.get('name', '')}; nonnegative counts are required"
            )
        pockets.append(
            {
                "name": row.get("name", ""),
                "rank": _integer(row, "rank"),
                "score": _float(row, "score"),
                "probability": probability,
                "sas_point_count": _integer(row, "sas_points"),
                "surface_atom_count": _integer(row, "surf_atoms"),
                "center_x_angstrom": _float(row, "center_x"),
                "center_y_angstrom": _float(row, "center_y"),
                "center_z_angstrom": _float(row, "center_z"),
                "residue_ids": residues,
            }
        )
    if [p["rank"] for p in pockets] != list(range(1, len(pockets) + 1)):
        raise ValueError(
            f"The P2Rank pocket ranks are {[p['rank'] for p in pockets]}; "
            "ranks must be consecutive integers starting at 1"
        )
    if any(not p["name"] for p in pockets):
        raise ValueError(
            "A P2Rank pocket has an empty name"
        )
    return pockets


def validate_residue_table(text: str) -> int:
    """Check P2Rank's per-residue table and return its row count."""
    reader = csv.DictReader(io.StringIO(text, newline=""), skipinitialspace=True)
    required = {
        "chain",
        "residue_label",
        "residue_name",
        "score",
        "zscore",
        "probability",
        "pocket",
    }
    missing = required - {key.strip() for key in reader.fieldnames or []}
    if missing:
        raise ValueError(
            f"The header of the P2Rank residues table lacks the columns {sorted(missing)}"
        )
    count = 0
    for item in reader:
        if None in item or any(value is None for value in item.values()):
            raise ValueError(
                "A row of the P2Rank residues table has a different number of fields "
                "than the header; one value per column is required"
            )
        row = {key.strip(): value.strip() for key, value in item.items()}
        for key in ("score", "zscore", "probability"):
            _float(row, key)
        if not 0 <= _float(row, "probability") <= 1 or _integer(row, "pocket") < 0:
            raise ValueError(
                "A row of the P2Rank residues table has a probability outside 0 to 1 or a "
                "negative pocket rank"
            )
        if not row["residue_label"] or not row["residue_name"]:
            raise ValueError(
                "A row of the P2Rank residues table has an empty residue_label or "
                "residue_name"
            )
        count += 1
    return count


def structure_from_dossier(dossier: dict[str, Any]) -> str:
    """The exact structure text of a `protein.inspect` record, checked against its mapping."""
    import hashlib

    from alambiq.capabilities.characterize.science.dossier import inspect_structure
    from alambiq.capabilities.shared.sequences import parse_fasta

    if not isinstance(dossier, dict) or dossier.get("schema_version") != "0.2":
        raise ValueError(
            "The record given in protein is not a protein.inspect record with "
            "schema_version 0.2 and a structure"
        )
    try:
        _, sequence = parse_fasta(f">query\n{dossier['sequence']}")
        if hashlib.sha256(sequence.encode()).hexdigest() != dossier["sequence_sha256"]:
            raise ValueError(
                "sequence_sha256 of the protein.inspect record differs from the SHA-256 "
                "of its sequence; the two must be equal"
            )
        text = dossier["source_texts"]["structure"]
        if (
            hashlib.sha256(text.encode()).hexdigest()
            != dossier["sources"]["structure"]["sha256"]
        ):
            raise ValueError(
                "The structure text of the protein.inspect record differs from its "
                "recorded SHA-256 digest; the two must be equal"
            )
        import gemmi

        atom = (
            gemmi.cif.read_string(text).sole_block().get_mmcif_category("_atom_site.")
        )
        if set(atom.get("pdbx_PDB_model_num", ["1"])) != {"1"}:
            raise ValueError(
                "The structure source contains a model other than model 1; mapping "
                "pockets requires a single deposited model numbered 1"
            )
        chains, atoms = inspect_structure(text, sequence, 1)
        if chains != dossier["chains"] or atoms != dossier["atoms"]:
            raise ValueError(
                "The coordinates or residue mappings of the protein.inspect record "
                "differ from its source; they must equal those parsed from the "
                "structure source"
            )
        return text
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            "The protein.inspect record lacks a structure, a mapping or another required "
            "field; a complete record with an exact structure source and mapping is required"
        ) from error


def map_pockets(
    dossier: dict[str, Any], pockets: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach each pocket's residues to query positions through exact author keys."""
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chain in dossier["chains"]:
        for residue in chain["residues"]:
            key = f"{residue['auth_asym_id']}_{residue['auth_seq_id']}{residue['insertion_code']}"
            by_key[key].append({"label_asym_id": chain["label_asym_id"], **residue})
    mapped = []
    for pocket in pockets:
        residues, unknown, ambiguous = [], [], []
        for key in pocket["residue_ids"]:
            matches = by_key.get(key, [])
            if len(matches) == 1:
                residues.append({"p2rank_residue_id": key, **matches[0]})
            elif matches:
                ambiguous.append(key)
            else:
                unknown.append(key)
        mapped.append(
            {
                **pocket,
                "residues": residues,
                "query_positions": sorted(
                    {
                        r["query_position"]
                        for r in residues
                        if r["query_position"] is not None
                    }
                ),
                "unmapped_residue_ids": unknown,
                "ambiguous_residue_ids": ambiguous,
            }
        )
    return mapped
