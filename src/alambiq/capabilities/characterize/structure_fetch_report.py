"""How a structure download appears in the dossier."""

from __future__ import annotations

from alambiq.report.page import esc, table


def render(result) -> str:
    return (
        f"<p>{esc(result.source)} {esc(result.identifier)} ({esc(result.format)}), "
        f"{result.bytes:,} bytes, retrieved {esc(result.retrieved_utc)}.</p>"
        + table(
            ["Field", "Value"],
            [
                ["URL", result.url],
                ["SHA-256", result.sha256],
                ["Last modified", result.last_modified or ""],
            ],
        )
    )
