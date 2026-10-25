"""Offline protein reference, sequence and structure inspection.

Builds one dossier in the query sequence's own coordinate system from a
sequence, an optional UniProt reference record and an optional mmCIF
structure. Every source keeps its own numbering: reference features are
projected onto the query only where the query occurs exactly once in the
reference, a structure chain is mapped only where its polymer sequence contains
the query exactly once or is contained by it exactly once, and a residue is
mapped only when its monomer agrees with the sequence at that position. Anything
else is kept and marked withheld, with the reason.

The functions take text, and each source is a mapping with a
`text` entry and optional `source_url`, `license` and `receipt` notes.
"""

from __future__ import annotations

import hashlib
import json
import math
import re

from alambiq.capabilities.shared.sequences import parse_fasta
from alambiq.capabilities.shared.sources import source_text

ALPHABET = re.compile(r"[ACDEFGHIKLMNPQRSTVWYBXZJUO]+")
MAX_BYTES = 20_000_000
MAX_RESIDUES = 10_000
MAX_ATOMS = 100_000


def unique_offset(container: str, fragment: str) -> int | None:
    if not fragment:
        return None
    index = container.find(fragment)
    return index if index >= 0 and container.find(fragment, index + 1) < 0 else None


def project_features(reference: dict, query: str) -> list[dict]:
    reference_sequence = reference.get("sequence", {}).get("value", "")
    offset = unique_offset(reference_sequence, query)
    records = []
    for feature in reference.get("features", []):
        location = feature.get("location", {})
        start, end = location.get("start", {}), location.get("end", {})
        a, b = start.get("value"), end.get("value")
        exact = (
            not location.get("sequence")
            and start.get("modifier") == end.get("modifier") == "EXACT"
            and type(a) is int
            and type(b) is int
            and 1 <= a <= b <= len(reference_sequence)
        )
        mapped = exact and offset is not None and offset < a <= b <= offset + len(query)
        records.append(
            {
                **feature,
                "reference_start": a,
                "reference_end": b,
                "query_start": a - offset if mapped else None,
                "query_end": b - offset if mapped else None,
                "mapping": "exact" if mapped else "withheld",
                "reason": "unique-exact-subsequence"
                if mapped
                else "sequence-mismatch-ambiguous-or-feature-outside-exact-query-span",
            }
        )
    return records


def inspect_structure(text: str, query: str, model_number: int) -> tuple[list, list]:
    try:
        import gemmi
    except ModuleNotFoundError as exc:
        if exc.name != "gemmi":
            raise
        raise ModuleNotFoundError(
            "Structure inspection requires the optional dossier extra (Gemmi 0.7.5). "
            "From the Alambiq repository root, run python -m pip install '.[dossier]' "
            "in the same Python environment used to run Alambiq, then rerun the campaign.",
            name="gemmi",
        ) from exc

    if gemmi.__version__ != "0.7.5":
        raise ValueError(
            f"gemmi {gemmi.__version__} is installed; gemmi 0.7.5 is required "
            "(extra `dossier`)"
        )
    block = gemmi.cif.read_string(text).sole_block()
    poly = block.get_mmcif_category("_entity_poly.")
    asym = block.get_mmcif_category("_struct_asym.")
    atom = block.get_mmcif_category("_atom_site.")
    if not poly or not asym or not atom:
        raise ValueError(
            "structure: at least one of the mmCIF categories _entity_poly, "
            "_struct_asym and _atom_site is missing or empty; all three are "
            "required, and PDB format is not read"
        )
    entities = {
        eid: "".join(seq.split())
        for eid, kind, seq in zip(
            poly["entity_id"],
            poly["type"],
            poly["pdbx_seq_one_letter_code_can"],
            strict=True,
        )
        if kind == "polypeptide(L)" and isinstance(seq, str)
    }
    # Some ModelCIF writers declare canonical X while supplying exact monomers.
    # Resolve only complete, unambiguous standard-residue sequences, never atoms
    # (which can omit residues) or the researcher's query as a substitute source.
    resolved = {}
    monomers = block.get_mmcif_category("_entity_poly_seq.")
    if {"entity_id", "num", "mon_id"} <= monomers.keys():
        for eid, canonical in entities.items():
            if "X" not in canonical:
                continue
            rows = [
                (int(number), monomer, hetero)
                for entity, number, monomer, hetero in zip(
                    monomers["entity_id"],
                    monomers["num"],
                    monomers["mon_id"],
                    monomers.get("hetero", [False] * len(monomers["num"])),
                    strict=True,
                )
                if entity == eid
            ]
            rows.sort(key=lambda row: row[0])
            letters = "".join(
                gemmi.find_tabulated_residue(r[1]).one_letter_code for r in rows
            )
            if (
                [r[0] for r in rows] == list(range(1, len(canonical) + 1))
                and all(r[2] in (False, None, "n", "no") for r in rows)
                and re.fullmatch(r"[ACDEFGHIKLMNPQRSTVWY]+", letters)
                and all(
                    a == "X" or a == b for a, b in zip(canonical, letters, strict=True)
                )
            ):
                resolved[eid] = letters
    chains = []
    by_chain = {}
    for chain, eid in zip(asym["id"], asym["entity_id"], strict=True):
        if eid not in entities:
            continue
        seq = resolved.get(eid, entities[eid])
        offset = (
            unique_offset(query, seq)
            if ALPHABET.fullmatch(seq) and "X" not in seq
            else None
        )
        contained_query = None
        if (
            offset is None
            and len(query) < len(seq)
            and ALPHABET.fullmatch(seq)
            and "X" not in seq
        ):
            contained_query = unique_offset(seq, query)
            if contained_query is not None:
                offset = -contained_query
        item = {
            "label_asym_id": chain,
            "entity_id": eid,
            "sequence": seq,
            "query_offset": offset,
            "mapping": "exact" if offset is not None else "withheld",
            "reason": "unique-exact-query-within-polymer"
            if contained_query is not None
            else "unique-exact-subsequence"
            if offset is not None
            else "sequence-mismatch-or-ambiguous",
            "model_number": model_number,
            "residues": [],
            **(
                {
                    "sequence_source": "entity_poly_seq-standard-monomers",
                    "declared_canonical_sequence": entities[eid],
                }
                if eid in resolved
                else {}
            ),
        }
        by_chain[chain] = item
        chains.append(item)
    count = len(atom["id"])
    if count > MAX_ATOMS:
        raise ValueError(
            f"structure: the file has {count:,} _atom_site rows; at most "
            f"{MAX_ATOMS:,} are accepted"
        )
    columns = list(atom)
    rows = [
        dict(zip(columns, values, strict=True))
        for values in zip(*(atom[c] for c in columns), strict=True)
    ]
    if str(model_number) not in {r.get("pdbx_PDB_model_num", "1") for r in rows}:
        raise ValueError(
            f"model_number: the requested model number is absent from the structure "
            f"(requested {model_number})"
        )
    atoms, residues = [], {}
    for row in rows:
        if str(row.get("pdbx_PDB_model_num", "1")) != str(model_number):
            continue
        # Apply the explicit display policy while retaining the source coordinates.
        if row.get("label_alt_id") not in (None, False, "", "A"):
            continue
        chain, seqid = row["label_asym_id"], row.get("label_seq_id")
        item = by_chain.get(chain)
        label = int(seqid) if seqid not in (None, False, "") else None
        qpos = None
        if item is not None and label is not None:
            if not 1 <= label <= len(item["sequence"]):
                raise ValueError(
                    "Atom label_seq_id falls outside the declared polymer sequence"
                )
            monomer = gemmi.find_tabulated_residue(
                row["label_comp_id"]
            ).one_letter_code.upper()
            if (
                item["query_offset"] is not None
                and monomer == item["sequence"][label - 1]
                and 1 <= label + item["query_offset"] <= len(query)
            ):
                qpos = label + item["query_offset"]
            key = (chain, label, row.get("auth_seq_id"), row.get("pdbx_PDB_ins_code"))
            residues[key] = {
                "label_seq_id": label,
                "auth_seq_id": row.get("auth_seq_id"),
                "auth_asym_id": row.get("auth_asym_id"),
                "insertion_code": row.get("pdbx_PDB_ins_code") or "",
                "query_position": qpos,
                "residue_name": row["label_comp_id"],
                "mapping": "exact" if qpos is not None else "withheld",
                "reason": "exact-residue"
                if qpos is not None
                else "outside-query-span"
                if item["query_offset"] is not None
                and not 1 <= label + item["query_offset"] <= len(query)
                else "unmapped-chain-or-monomer-mismatch",
            }
        xyz = [float(row[f"Cartn_{axis}"]) for axis in "xyz"]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError("Non-finite atom coordinates")
        atoms.append(
            {
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "elem": row["type_symbol"],
                "atom": row["label_atom_id"],
                "resn": row["label_comp_id"],
                "chain": chain,
                "resi": label
                if label is not None
                else int(row.get("auth_seq_id") or 0),
                "hetflag": item is None,
                "properties": {
                    "query_position": qpos,
                    "label_seq_id": label,
                    "source_atom_id": row["id"],
                    "auth_chain": row.get("auth_asym_id"),
                    "auth_residue": row.get("auth_seq_id"),
                    "insertion_code": row.get("pdbx_PDB_ins_code") or "",
                },
            }
        )
    for (chain, *_), residue in residues.items():
        by_chain[chain]["residues"].append(residue)
    if not atoms:
        raise ValueError(
            "No display atoms under the declared model/alternate-location policy"
        )
    return chains, atoms


def build_dossier(inputs: dict, model_number: int = 1) -> dict:
    texts = {name: source_text(spec) for name, spec in inputs.items()}
    reference = json.loads(texts["uniprot"]) if "uniprot" in texts else {}
    if reference:
        parse_fasta(
            f">{reference['primaryAccession']}\n{reference['sequence']['value']}"
        )
    if "fasta" in texts:
        identifier, sequence = parse_fasta(texts["fasta"])
    else:
        identifier, sequence = parse_fasta(
            f">{reference['primaryAccession']}\n{reference['sequence']['value']}"
        )
    features = project_features(reference, sequence)
    chains, atoms = (
        inspect_structure(texts["structure"], sequence, model_number)
        if "structure" in texts
        else ([], [])
    )
    sources = {
        name: {
            "sha256": hashlib.sha256(texts[name].encode("utf-8")).hexdigest(),
            "source_url": spec.get(
                "source_url",
                "user-supplied:local-file"
                if "path" in spec
                else "user-supplied:inline",
            ),
            "license": spec.get("license", "unspecified"),
            **({"receipt": spec["receipt"]} if "receipt" in spec else {}),
        }
        for name, spec in inputs.items()
    }
    return {
        "schema_version": "0.2",
        "identifier": identifier,
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "reference": reference,
        "features": list(features),
        "chains": list(chains),
        "atoms": list(atoms),
        "sources": sources,
        "source_texts": texts,
    }
