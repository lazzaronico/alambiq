"""Interactive views for the report: molecular structure and sequence alignment.

The structure viewer is 3Dmol.js, bundled in `assets/` so a report needs no network. It is
added to a page only when the page contains a structure; a report without a structure
contains no script.
The alignment view is plain HTML and CSS.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
MAX_STRUCTURE_BYTES = 4_000_000
MAX_ALIGNMENT_ROWS = 100
MAX_ALIGNMENT_COLUMNS = 3000

VIEWER_MARKER = 'class="bf-structure"'

VIEWER_CSS = """
.bf-structure { width:100%; border:1px solid var(--line); border-radius:4px;
                position:relative; margin:12px 0; }
.bf-structure[data-ready="error"] { display:flex; align-items:center;
  justify-content:center; color:var(--muted); font-size:13px; }
.aln { font:12.5px/1.35 ui-monospace,SFMono-Regular,Menlo,monospace; margin:12px 0;
       overflow-x:auto; }
.aln-block { margin-bottom:10px; white-space:pre; }
.aln-row { display:flex; gap:10px; }
.aln-name { width:14ch; flex:none; overflow:hidden; text-overflow:ellipsis;
            color:var(--muted); }
.aln-seq { white-space:pre; }
.aln .m { background:color-mix(in srgb, #0072b2 22%, Canvas); }
.aln .x { background:color-mix(in srgb, #d55e00 22%, Canvas); }
.aln .g { color:GrayText; }
.aln-pos { color:var(--muted); }
"""

# Reads each embedded structure and draws it; a failed drawing leaves a message.
VIEWER_SCRIPT = """
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".bf-structure").forEach(function (el) {
    try {
      var data = el.nextElementSibling.textContent.replace(/<\\\\\\/script/g, "</script");
      var viewer = $3Dmol.createViewer(el, {backgroundColor: "white"});
      viewer.addModel(data, el.dataset.format);
      JSON.parse(el.dataset.styles).forEach(function (s) {
        viewer.setStyle(s.sel || {}, s.style);
      });
      viewer.zoomTo();
      viewer.render();
      el.dataset.ready = "true";
    } catch (error) {
      el.dataset.ready = "error";
      el.textContent = "The structure cannot be drawn: " + error.message;
    }
  });
});
"""

DEFAULT_STYLES = (
    {"sel": {}, "style": {"cartoon": {"color": "spectrum"}}},
    {"sel": {"hetflag": True}, "style": {"stick": {"colorscheme": "greenCarbon"}}},
)


def read_asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def viewer_scripts() -> str:
    """The bundled 3Dmol and the glue that draws each embedded structure."""
    library = read_asset("3Dmol-min.js").replace("</script", "<\\/script")
    return f"<script>{library}</script><script>{VIEWER_SCRIPT}</script>"


def structure_viewer(
    text: str,
    fmt: str = "pdb",
    *,
    styles: Sequence[dict] = DEFAULT_STYLES,
    height: int = 420,
    caption: str = "",
) -> str:
    """A rotatable structure. `fmt` is a 3Dmol format name such as `pdb`, `cif`, `sdf`."""
    if len(text.encode("utf-8")) > MAX_STRUCTURE_BYTES:
        return (
            '<p class="sub">Structures up to 4,000,000 bytes are embedded in the report; '
            "this one is larger. Open its file in a molecular viewer.</p>"
        )
    data = text.replace("</script", "<\\/script")
    figure = (
        f'<div {VIEWER_MARKER} data-format="{escape(fmt, quote=True)}" '
        f'data-styles="{escape(json.dumps(list(styles)), quote=True)}" '
        f'style="height:{int(height)}px"></div>'
        f'<script type="text/plain" class="bf-structure-data">{data}</script>'
    )
    if caption:
        figure += f'<p class="sub">{escape(caption)}</p>'
    return figure


SOLVENT = {"HOH", "WAT", "NA", "CL", "Na+", "Cl-", "K", "K+"}


def without_solvent(text: str) -> str:
    """A PDB with its water and simple ions removed, so a solvated box shows the solute."""
    return "\n".join(
        line
        for line in text.splitlines()
        if not (
            line.startswith(("ATOM", "HETATM", "ANISOU"))
            and line[17:20].strip() in SOLVENT
        )
        and not line.startswith("CONECT")
    )


def structure_file(
    path: str | Path, fmt: str | None = None, *, solute_only: bool = False, **options
) -> str:
    """A viewer for a structure file, or nothing when it cannot be read."""
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""
    kind = fmt or _format(text)
    if solute_only and kind == "pdb":
        text = without_solvent(text)
    return structure_viewer(text, kind, **options)


def _format(text: str) -> str:
    head = text.lstrip()[:200]
    if head.startswith("data_") or "_atom_site." in text[:5000]:
        return "cif"
    return "pdb"


def alignment_view(
    rows: Sequence[tuple[str, str]], *, reference: int = 0, block: int = 60
) -> str:
    """Aligned sequences in blocks; residues are marked against the reference row.

    Residues that equal the reference residue are shaded blue and residues that differ
    are shaded orange; gaps are grey. Each row stores its full aligned string in
    `data-sequence`.
    """
    rows = list(rows)[:MAX_ALIGNMENT_ROWS]
    if not rows:
        return ""
    columns = min(max(len(s) for _, s in rows), MAX_ALIGNMENT_COLUMNS)
    if any(len(s) != len(rows[0][1]) for _, s in rows):
        raise ValueError(
            "Aligned sequences differ in length; all rows must have the same length"
        )
    ref = rows[reference][1]
    out = [
        f'<div class="aln" data-rows="{len(rows)}" data-columns="{columns}">',
    ]
    for start in range(0, columns, block):
        out.append(
            f'<div class="aln-block"><div class="aln-row aln-pos">'
            f'<span class="aln-name"></span><span class="aln-seq">{start + 1}</span></div>'
        )
        for index, (name, sequence) in enumerate(rows):
            runs: list[tuple[str, str]] = []
            for column in range(start, min(start + block, columns)):
                char = sequence[column]
                if char in "-.":
                    cls = "g"
                elif index == reference:
                    cls = "r"
                elif char.upper() == ref[column].upper():
                    cls = "m"
                else:
                    cls = "x"
                if runs and runs[-1][0] == cls:
                    runs[-1] = (cls, runs[-1][1] + char)
                else:
                    runs.append((cls, char))
            body = "".join(
                f'<span class="{cls}">{escape(text)}</span>' for cls, text in runs
            )
            out.append(
                f'<div class="aln-row" data-name="{escape(name, quote=True)}" '
                f'data-sequence="{escape(sequence[:columns], quote=True) if start == 0 else ""}">'
                f'<span class="aln-name" title="{escape(name, quote=True)}">'
                f'{escape(name)}</span><span class="aln-seq">{body}</span></div>'
            )
        out.append("</div>")
    out.append("</div>")
    return "".join(out)
