"""How called substitutions appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def _differences(call) -> str:
    parts = [call.notation] if call.notation else []
    parts += [
        f"insertion {i.residues} after {i.after_position}" for i in call.insertions
    ]
    parts += [f"deletion {d.start}-{d.end}" for d in call.deletions]
    ends = [
        ("variant start", call.unaligned_variant_start),
        ("variant end", call.unaligned_variant_end),
        ("reference start", call.unaligned_reference_start),
        ("reference end", call.unaligned_reference_end),
    ]
    parts += [f"{n} unaligned at {label}" for label, n in ends if n]
    if call.reference is None:
        return ""
    return ", ".join(parts) if parts else "none"


def render(result) -> str:
    rows = [
        [
            c.id,
            c.length,
            c.reference or "",
            c.reference_name,
            c.identity if c.identity is not None else "",
            _differences(c),
        ]
        for c in result.variants
    ]
    return "".join(
        [
            f"<p>{result.matched} of {len(result.variants)} variants have a closest reference "
            f"in the {esc(result.source)}, aligned with "
            f"{esc(result.engine_versions.get('mmseqs', 'MMseqs2'))}.</p>",
            table(
                [
                    "Variant",
                    "Length (residues)",
                    "Closest reference",
                    "Name",
                    "Identity",
                    "Differences",
                ],
                rows,
                numeric={1, 4},
            ),
        ]
    )
