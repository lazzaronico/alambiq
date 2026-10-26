"""The report: one self-contained HTML file per run.

The page contains the navigation and, for each step, the notes and the result. A
capability supplies a `render` function. It receives the typed result and returns the
HTML of the result.

A report opens from a file and needs no server and no network.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alambiq.core import registry
from alambiq.report import viewers

if TYPE_CHECKING:  # pragma: no cover
    from alambiq.core.execution import RunResult

STYLE = """
:root { color-scheme: light dark; --muted:GrayText; --accent:LinkText;
        --line:color-mix(in srgb, CanvasText 18%, Canvas); }
* { box-sizing:border-box; }
body { margin:0; font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
.wrap { display:grid; grid-template-columns:minmax(200px,240px) 1fr; }
nav { border-right:1px solid var(--line); padding:24px 16px; position:sticky;
      top:0; align-self:start; }
nav h1 { font-size:15px; margin:0 0 12px; }
nav a { display:block; padding:5px 0; color:var(--muted); text-decoration:none;
        font-size:14px; }
nav a:hover { color:var(--accent); }
main { padding:28px 32px 64px; max-width:900px; }
section { border-bottom:1px solid var(--line); padding-bottom:28px;
          margin-bottom:28px; }
h2 { font-size:20px; margin:0 0 4px; }
.sub { color:var(--muted); font-size:13px; margin:0 0 16px; }
table { border-collapse:collapse; width:100%; font-size:14px; margin:12px 0; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { color:var(--muted); font-weight:600; }
td.num { font-variant-numeric:tabular-nums; }
.note { background:color-mix(in srgb, #e69f00 16%, Canvas); border-left:3px solid #e69f00;
         padding:10px 14px; margin:12px 0; font-size:14px; }
.failed { background:color-mix(in srgb, #d55e00 16%, Canvas); border-left:3px solid #d55e00;
          padding:10px 14px; font-size:14px; }
details { margin:12px 0; font-size:14px; }
summary { cursor:pointer; color:var(--accent); }
pre { background:color-mix(in srgb, CanvasText 6%, Canvas); padding:12px; overflow-x:auto;
      font-size:12.5px; border-radius:4px; }
.badge { font-size:12px; color:var(--muted); border:1px solid var(--line);
         border-radius:10px; padding:1px 8px; margin-left:6px; }
svg.bf-chart { display:block; margin:12px 0; }
@media (max-width:760px) {
  .wrap { grid-template-columns:1fr; }
  nav { position:static; border-right:0; border-bottom:1px solid var(--line); }
  main { padding:20px 16px 48px; }
}
"""


def read_record(path: str | Path | None) -> dict[str, Any] | None:
    """A step's stored record, or None when it cannot be read.

    Reports draw charts from the full record and not from the summary that the step
    returns. When the record has been moved or removed, the report omits the chart.
    """
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def number(value: Any, places: int = 6) -> str:
    """A number formatted for a table cell."""
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return esc(value)
    if isinstance(value, int):
        return f"{value:,}"
    if value and (abs(value) < 1e-4 or abs(value) >= 1e7):
        return f"{value:.{places}e}"
    return f"{value:,.{places}g}"


def table(
    headers: list[str], rows: list[list[Any]], numeric: set[int] | None = None
) -> str:
    numeric = numeric or set()
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f'<td class="num">{number(c)}</td>'
            if i in numeric
            else f"<td>{esc(c)}</td>"
            for i, c in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _step_section(step: Any) -> str:
    cap = None
    try:
        cap = registry.get(step.capability)
    except KeyError:
        pass

    parts = [f'<section id="{esc(step.step_id)}">']
    badge = (
        '<span class="badge">reused</span>'
        if step.reused
        else f'<span class="badge">{step.seconds:.2f}s</span>'
    )
    parts.append(f"<h2>{esc(step.step_id)}{badge}</h2>")
    summary = cap.summary if cap else step.capability
    parts.append(f'<p class="sub">{esc(step.capability)} · {esc(summary)}</p>')

    if step.status == "failed":
        parts.append(f'<div class="failed">{esc(step.error)}</div>')

    for note in step.notes:
        parts.append(f'<div class="note">{esc(note)}</div>')

    if cap is not None and cap.render is not None and step.result is not None:
        parts.append(cap.render(step.result))
    elif step.outputs:
        parts.append(
            "<details><summary>Result</summary><pre>"
            + esc(json.dumps(step.outputs, indent=2, default=str)[:20000])
            + "</pre></details>"
        )

    parts.append("</section>")
    return "".join(parts)


def _history_rows(run: RunResult) -> list[dict[str, Any]]:
    from alambiq.core import records

    return records.history(run.run_dir, run.campaign)


def _history_section(run: RunResult) -> str:
    """Earlier runs of this campaign in the same run directory."""
    rows = _history_rows(run)
    if len(rows) < 2:
        return ""
    body = [
        [
            r["started"],
            "complete" if r["ok"] else "incomplete",
            r["computed"],
            r["reused"],
            r["failed"],
            r["seconds"],
        ]
        for r in reversed(rows)
    ]
    return (
        '<section id="history"><h2>History</h2>'
        f'<p class="sub">{len(rows)} runs of {esc(run.campaign)} are recorded in this '
        "run directory, newest first.</p>"
        + table(
            ["Started (UTC)", "Outcome", "Computed", "Reused", "Failed", "Seconds"],
            body,
            numeric={2, 3, 4, 5},
        )
        + "</section>"
    )


def render_page(run: RunResult) -> str:
    order = list(run.steps.values())
    links = "".join(f'<a href="#{esc(s.step_id)}">{esc(s.step_id)}</a>' for s in order)
    sections = "".join(_step_section(s) for s in order) + _history_section(run)
    scripts = viewers.viewer_scripts() if viewers.VIEWER_MARKER in sections else ""
    state = "complete" if run.ok else "incomplete"
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(run.campaign)}</title><style>{STYLE}{viewers.VIEWER_CSS}</style></head><body>"
        f'<div class="wrap"><nav><h1>{esc(run.campaign)}</h1>'
        f'<p class="sub">{len(order)} steps · {state}</p>{links}</nav>'
        f"<main>{sections}</main></div>{scripts}</body></html>"
    )


def write_report(run: RunResult, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_page(run))
    return target
