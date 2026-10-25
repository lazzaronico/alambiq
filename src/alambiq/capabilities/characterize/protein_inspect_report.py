"""How a protein inspection appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, number, table

CODES = {
    "H": "α-helix",
    "G": "3₁₀ helix",
    "I": "π-helix",
    "E": "strand",
    "B": "isolated bridge",
    "T": "turn",
    "S": "bend",
    " ": "coil",
}


def render(result) -> str:
    parts = [
        f'<p class="sub">{esc(result.identifier)} · {result.sequence_length} residues · '
        f"sequence {esc(result.sequence_sha256[:16])}</p>"
    ]
    if result.reference_features:
        parts.append(
            f"<p>{result.mapped_features} of {result.reference_features} reference "
            "features map exactly onto the query.</p>"
        )
    if result.chains:
        parts.append(
            table(
                ["Chain", "Mapping", "Reason", "Offset (residues)"],
                [
                    [
                        c["chain"],
                        c["mapping"],
                        c["reason"].replace("-", " "),
                        c["query_offset"],
                    ]
                    for c in result.chain_mappings
                ],
                numeric={3},
            )
        )
        parts.append(
            f'<p class="sub">{result.mapped_residues} of {result.observed_residues} '
            "observed polymer residues have a query position.</p>"
        )
    if result.confidence is not None:
        c = result.confidence
        parts.append(
            "<p>Declared per-residue pLDDT: "
            f"mean {number(c.mean)}, from {number(c.lowest)} to {number(c.highest)} over "
            f"{c.residues - c.missing} residues"
            + (f", {c.missing} without a value" if c.missing else "")
            + ".</p>"
        )
    if result.secondary_structure is not None:
        parts.append(
            table(
                ["Secondary structure", "Residues"],
                [
                    [f"{CODES.get(code, code)} ({code.strip() or 'coil'})", count]
                    for code, count in sorted(
                        result.secondary_structure.items(), key=lambda kv: -kv[1]
                    )
                ],
                numeric={1},
            )
        )
        parts.append(
            f'<p class="sub">Mean surface area {number(result.mean_surface_area)} '
            "Å² per observed residue.</p>"
        )
    return "".join(parts)
