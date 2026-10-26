"""Explicit mapped-alpha-carbon rigid fits over verified dossier source bytes."""

import hashlib
import json
import math

from alambiq.capabilities.characterize.science.alignment import inspect_alignment
from alambiq.capabilities.characterize.science.dossier import inspect_structure
from alambiq.capabilities.shared.sequences import parse_fasta
from alambiq.capabilities.shared.sources import read_input


def _source(dossier, chain_id):
    try:
        if dossier["schema_version"] != "0.2":
            raise ValueError(
                "schema_version of the protein.inspect record is not 0.2; a current record is required"
            )
        sequence = parse_fasta(f">query\n{dossier['sequence']}")[1]
        if hashlib.sha256(sequence.encode()).hexdigest() != dossier["sequence_sha256"]:
            raise ValueError(
                "sequence_sha256 of the protein.inspect record differs from the SHA-256 of its sequence; the two must be equal"
            )
        text = read_input(
            {
                "text": dossier["source_texts"]["structure"],
                "sha256": dossier["sources"]["structure"]["sha256"],
            }
        )
        models = {c["model_number"] for c in dossier["chains"]}
        if len(models) != 1 or any(type(m) is not int or m < 1 for m in models):
            raise ValueError(
                "The chains of the protein.inspect record name no model number or more than one; exactly one positive integer model number is required"
            )
        model = next(iter(models))
        chains, atoms = inspect_structure(text, sequence, model)
        if chains != dossier["chains"] or atoms != dossier["atoms"]:
            raise ValueError(
                "The chains or atoms of the protein.inspect record differ from those "
                "parsed from its structure source; they must be identical"
            )
        selected = [c for c in chains if c["label_asym_id"] == chain_id]
        if len(selected) != 1 or selected[0]["mapping"] != "exact":
            raise ValueError(
                f"The label_asym_id {chain_id} is absent, duplicated or not mapped exactly to the query; "
                "one chain with mapping exact is required on each side"
            )
        atoms = [a for a in atoms if a["chain"] == chain_id]
        ca = {}
        for atom in atoms:
            position = atom["properties"]["query_position"]
            if atom["atom"] == "CA" and atom["elem"] == "C" and position is not None:
                if position in ca:
                    raise ValueError(
                        f"Chain {chain_id} has more than one alpha carbon at query "
                        f"position {position}; at most one is allowed"
                    )
                ca[position] = atom
        return {
            "identifier": dossier["identifier"],
            "sequence": sequence,
            "sequence_sha256": dossier["sequence_sha256"],
            "label_asym_id": chain_id,
            "model_number": model,
            "source": dossier["sources"]["structure"],
            "source_text": text,
            "atoms": atoms,
        }, ca
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "A protein.inspect record lacks a required field or has a field of the wrong "
            "type; complete records with structure sources and exact mappings are required"
        ) from exc


def _noncollinear(points):
    # Translation-invariant relative area check, only to reject an undefined fit.
    origin = points[0]
    vectors = [p - origin for p in points[1:]]
    longest = max(vectors, key=lambda v: v.length())
    scale = longest.length()
    return scale > 0 and any(
        longest.cross(v).length() > scale * scale * 1e-10 for v in vectors
    )


def _fitted_cif(mobile, transform):
    import gemmi

    original = gemmi.cif.read_string(mobile["source_text"]).sole_block()
    doc = gemmi.cif.Document()
    block = doc.add_new_block("alambiq_fitted_chain")
    asym = original.get_mmcif_category("_struct_asym.")
    entity = asym["entity_id"][asym["id"].index(mobile["label_asym_id"])]
    for category, column, selected in (
        ("_entity.", "id", entity),
        ("_entity_poly.", "entity_id", entity),
        ("_entity_poly_seq.", "entity_id", entity),
        ("_struct_asym.", "id", mobile["label_asym_id"]),
    ):
        data = original.get_mmcif_category(category)
        if data:
            indices = [i for i, v in enumerate(data[column]) if v == selected]
            block.set_mmcif_category(
                category, {k: [v[i] for i in indices] for k, v in data.items()}
            )
    data = original.get_mmcif_category("_atom_site.")
    ids = {a["properties"]["source_atom_id"] for a in mobile["atoms"]}
    indices = [
        i
        for i, value in enumerate(data["id"])
        if value in ids
        and data["label_asym_id"][i] == mobile["label_asym_id"]
        and data.get("pdbx_PDB_model_num", ["1"] * len(data["id"]))[i]
        == str(mobile["model_number"])
    ]
    if len(indices) != len(ids) or len(ids) != len(mobile["atoms"]):
        raise ValueError(
            "The atom ids of the fitted chain do not each match one atom_site row of the "
            "structure source; one row per atom is required"
        )
    selected = {k: [v[i] for i in indices] for k, v in data.items()}
    for i in range(len(indices)):
        point = transform.apply(
            gemmi.Position(*(float(selected[f"Cartn_{k}"][i]) for k in "xyz"))
        )
        for key, value in zip("xyz", point.tolist(), strict=True):
            selected[f"Cartn_{key}"][i] = f"{value:.10f}"
    block.set_mmcif_category("_atom_site.", selected)
    return doc.as_string()


def _alignment_pairs(spec, format, selected, reference, mobile):
    if not isinstance(selected, dict) or set(selected) != {"reference", "mobile"}:
        raise ValueError(
            "The alignment rows must be a mapping with exactly the keys reference and "
            "mobile, each naming one row identifier"
        )
    if any(not isinstance(v, str) or not v for v in selected.values()):
        raise ValueError(
            "reference_row or mobile_row is missing or empty; both must be nonempty strings"
        )
    text = read_input(spec)
    inspected = inspect_alignment(
        text,
        format,
        full_sequence_context=spec.get("receipt", {}).get("full_sequence_context"),
    )
    rows = {r["id"]: r for r in inspected["rows"]}
    positions = {}
    for side, source in (("reference", reference), ("mobile", mobile)):
        if selected[side] not in rows:
            raise ValueError(
                f"{side}_row names {selected[side]}, which is not a row of the alignment; "
                "an identifier of an alignment row is required"
            )
        row = rows[selected[side]]
        if row.get("full_sequence", row["sequence"]) != source["sequence"]:
            raise ValueError(
                f"The {side} alignment row has a sequence that differs from the query "
                f"sequence of the {side} protein.inspect record; the two must be identical"
            )
        positions[side] = row.get("full_sequence_positions", row["positions"])
    columns, pairs = [], []
    for column, (r, m) in enumerate(
        zip(positions["reference"], positions["mobile"], strict=True), 1
    ):
        if r is not None and m is not None:
            columns.append(column)
            pairs.append({"reference": r, "mobile": m})
    return pairs, {
        "format": format,
        "rows": dict(selected),
        "paired_columns": columns,
        "unpaired_column_count": len(positions["reference"]) - len(columns),
        "source": {
            **{k: v for k, v in spec.items() if k != "path"},
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    }


def compare_structures(
    reference,
    mobile,
    *,
    reference_chain,
    mobile_chain,
    residue_pairs=None,
    alignment=None,
    alignment_format=None,
    alignment_rows=None,
):
    import gemmi

    ref, ref_ca = _source(reference, reference_chain)
    mob, mob_ca = _source(mobile, mobile_chain)
    alignment_record = None
    if alignment is not None:
        if residue_pairs is not None:
            raise ValueError(
                "Both alignment and residue_pairs are set; at most one of them is allowed"
            )
        residue_pairs, alignment_record = _alignment_pairs(
            alignment, alignment_format, alignment_rows, ref, mob
        )
        correspondence = "selected-alignment-columns"
    elif alignment_format is not None or alignment_rows is not None:
        raise ValueError(
            "alignment_format or a row identifier is set without alignment; "
            "these fields require alignment"
        )
    elif residue_pairs is None:
        if ref["sequence"] != mob["sequence"]:
            raise ValueError(
                "The query sequences of the two records differ and neither residue_pairs "
                "nor alignment is set; one of them is required"
            )
        residue_pairs = [
            {"reference": i, "mobile": i} for i in range(1, len(ref["sequence"]) + 1)
        ]
        correspondence = "identical-query-positions"
    else:
        correspondence = "explicit-query-positions"
    if (
        not isinstance(residue_pairs, (list, tuple))
        or not 1 <= len(residue_pairs) <= 10000
    ):
        raise ValueError("residue_pairs must be a list of 1 to 10,000 pairs")
    seen = {"reference": set(), "mobile": set()}
    pairs, ref_points, mob_points = [], [], []
    for pair in residue_pairs:
        if not isinstance(pair, dict) or set(pair) != set(seen):
            raise ValueError(
                "A pair in residue_pairs has keys other than reference and mobile; "
                "each pair requires exactly these two keys"
            )
        for side, data in (("reference", ref), ("mobile", mob)):
            value = pair[side]
            if (
                type(value) is not int
                or not 1 <= value <= len(data["sequence"])
                or value in seen[side]
            ):
                raise ValueError(
                    "A pair in residue_pairs repeats a position or has a position that is "
                    "not an integer within the query length; positions must be unique, "
                    "in-range integers on each side"
                )
            seen[side].add(value)
        r, m = ref_ca.get(pair["reference"]), mob_ca.get(pair["mobile"])
        status = (
            "fitted"
            if r and m
            else "missing-both-ca"
            if not r and not m
            else "missing-reference-ca"
            if not r
            else "missing-mobile-ca"
        )
        pairs.append(
            {
                **pair,
                "reference_residue": ref["sequence"][pair["reference"] - 1],
                "mobile_residue": mob["sequence"][pair["mobile"] - 1],
                "status": status,
                "reference_atom": r,
                "mobile_atom": m,
                "distance_before_angstrom": None,
                "distance_after_angstrom": None,
            }
        )
        if r and m:
            ref_points.append(gemmi.Position(*(r[k] for k in "xyz")))
            mob_points.append(gemmi.Position(*(m[k] for k in "xyz")))
    if len(ref_points) < 3:
        raise ValueError(
            f"{len(ref_points)} pairs have an alpha carbon in both structures; "
            "at least 3 are required"
        )
    if not _noncollinear(ref_points) or not _noncollinear(mob_points):
        raise ValueError(
            "The alpha carbons of the fitted pairs are collinear or coincident in one "
            "structure; noncollinear positions are required in both"
        )
    fit = gemmi.superpose_positions(ref_points, mob_points)
    valid = [p for p in pairs if p["status"] == "fitted"]
    for pair, r, m in zip(valid, ref_points, mob_points, strict=True):
        pair["distance_before_angstrom"] = (r - m).length()
        pair["distance_after_angstrom"] = (
            r - gemmi.Position(fit.transform.apply(m))
        ).length()
    fitted_atoms = []
    for atom in mob["atoms"]:
        coords = fit.transform.apply(gemmi.Position(*(atom[k] for k in "xyz")))
        fitted_atoms.append({**atom, **dict(zip("xyz", coords.tolist(), strict=True))})
    fitted = _fitted_cif(mob, fit.transform)
    for side in (ref, mob):
        side["fit_query_coverage"] = len(valid) / len(side["sequence"])
    data = {
        "schema_version": "0.1",
        "method": "mapped-ca-superposition-v1",
        "gemmi_version": gemmi.__version__,
        "reference": ref,
        "mobile": mob,
        "correspondence": correspondence,
        "pairs": pairs,
        "requested_pair_count": len(pairs),
        "fit_pair_count": len(valid),
        "rmsd_before_angstrom": math.sqrt(
            sum(p["distance_before_angstrom"] ** 2 for p in valid) / len(valid)
        ),
        "rmsd_after_angstrom": math.sqrt(
            sum(p["distance_after_angstrom"] ** 2 for p in valid) / len(valid)
        ),
        "transform": {
            "convention": "reference_frame_position = rotation @ mobile_source_position + translation_angstrom",
            "rotation": fit.transform.mat.tolist(),
            "translation_angstrom": fit.transform.vec.tolist(),
        },
        "fit_policy": "Uniform alpha-carbon weights; every pair with an observed alpha carbon in both structures enters the fit, and no pair is rejected by its distance. Pairs with a missing alpha carbon are listed with their status. One selected deposited model and label chain per source, blank/A alternates. The fitted structure contains the selected mobile chain only; all its atoms are transformed.",
        "fitted_mobile_atoms": fitted_atoms,
        "fitted_structure_sha256": hashlib.sha256(fitted.encode()).hexdigest(),
        "fitted_structure_text": fitted,
    }
    if alignment_record is not None:
        data["alignment_correspondence"] = alignment_record
    json.dumps(data, allow_nan=False)
    return data, fitted
