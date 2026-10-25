"""Inspect the local pLDDT that a ModelCIF declares explicitly."""

import math


def _rows(block, category, required):
    data = block.get_mmcif_category(category)
    if data and not set(required) <= data.keys():
        raise ValueError(f"Incomplete {category} confidence category")
    return (
        [
            dict(zip(data, values, strict=True))
            for values in zip(*data.values(), strict=True)
        ]
        if data
        else []
    )


def inspect_confidence(dossier, model_number):
    text = dossier["source_texts"].get("structure")
    if text is None:
        return None
    import gemmi

    block = gemmi.cif.read_string(text).sole_block()
    definitions = _rows(block, "_ma_qa_metric.", ("id", "name", "type", "mode"))
    chosen = [r for r in definitions if r["type"] == "pLDDT" and r["mode"] == "local"]
    if not chosen:
        return None
    if len(chosen) != 1 or len({r["id"] for r in definitions}) != len(definitions):
        raise ValueError("Ambiguous local pLDDT metric definitions")
    metric = chosen[0]
    chains = {c["label_asym_id"]: c for c in dossier["chains"]}
    asym = block.get_mmcif_category("_struct_asym.")
    identities = set(asym["id"])
    seen, records = set(), []
    rows = _rows(
        block,
        "_ma_qa_metric_local.",
        (
            "model_id",
            "label_asym_id",
            "label_seq_id",
            "label_comp_id",
            "metric_id",
            "metric_value",
        ),
    )
    for row in rows:
        if row["model_id"] != str(model_number) or row["metric_id"] != metric["id"]:
            continue
        chain_id, position = row["label_asym_id"], int(row["label_seq_id"])
        key = (chain_id, position)
        if chain_id not in identities or position < 1 or key in seen:
            raise ValueError("Invalid or duplicate local pLDDT residue identity")
        seen.add(key)
        value = row["metric_value"]
        value = None if value in (None, False, "") else float(value)
        if value is not None and (not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError(
                "ModelCIF pLDDT must be finite on its declared 0–100 scale"
            )
        chain = chains.get(chain_id)
        qpos, observed = None, False
        if chain:
            if position > len(chain["sequence"]):
                raise ValueError("Local pLDDT position outside polymer sequence")
            monomer = gemmi.find_tabulated_residue(
                row["label_comp_id"]
            ).one_letter_code.upper()
            if (
                chain["sequence"][position - 1] != "X"
                and monomer != chain["sequence"][position - 1]
            ):
                raise ValueError("Local pLDDT monomer disagrees with polymer sequence")
            matches = [r for r in chain["residues"] if r["label_seq_id"] == position]
            observed = bool(matches)
            positions = {r["query_position"] for r in matches}
            if len(positions) == 1:
                qpos = next(iter(positions))
            elif (
                not observed
                and chain["query_offset"] is not None
                and monomer == chain["sequence"][position - 1]
                and 1 <= position + chain["query_offset"] <= len(dossier["sequence"])
            ):
                qpos = position + chain["query_offset"]
        records.append(
            {
                "label_asym_id": chain_id,
                "label_seq_id": position,
                "residue_name": row["label_comp_id"],
                "query_position": qpos,
                "value": value,
                "polymer": chain is not None,
                "observed": observed,
                "status": "missing-value" if value is None else "present",
            }
        )
    return {
        "schema_version": "0.1",
        "source_structure_sha256": dossier["sources"]["structure"]["sha256"],
        "query_sequence_sha256": dossier["sequence_sha256"],
        "model_number": model_number,
        "source_category": "_ma_qa_metric_local",
        "metric": metric,
        "scale": [0, 100],
        "residues": records,
    }
