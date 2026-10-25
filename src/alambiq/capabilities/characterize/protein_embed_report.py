"""How embeddings appear in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    versions = ", ".join(f"{k} {v}" for k, v in result.engine_versions.items())
    return "".join(
        [
            f"<p>{len(result.proteins)} proteins embedded with {esc(result.model)} "
            f"(revision <code>{esc(result.revision[:12])}</code>, {esc(result.license)}). "
            f"Each vector has {result.dimension} values; layer {esc(result.layer)}, "
            f"pooling {esc(result.pooling)}, device {esc(result.device)}.</p>",
            table(
                ["Protein", "Length (residues)", "Sequence SHA-256"],
                [[p.id, p.length, p.sequence_sha256[:16]] for p in result.proteins],
                numeric={1},
            ),
            f'<p class="sub">{esc(versions)}</p>',
        ]
    )
