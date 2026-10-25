"""Every report an example writes is structurally sound.

The checks are mechanical, on the HTML of the reports the engine-free examples write:

* the document parses with balanced tags, a title, a language and a character set;
* it is self-contained: nothing is loaded from the network (no `src`, stylesheet, `@import`
  or `url()` that is not inline), and no script calls out;
* no Python or NumPy representation, `NaN`, `inf` or `None` shows in the text a reader sees;
* a table column of numbers states its unit in its header, or is a count, an index, a rank
  or a fraction in a header that says so.

* the report's outline (sections, headings, notes, table headers, collapsed sections) matches
  the snapshot kept in `tests/snapshots/reports/`, so a change to what a report says shows up
  in review. Set `ALAMBIQ_UPDATE_SNAPSHOTS=1` to rewrite the snapshots after a deliberate change.

These establish that the information is well formed. Whether a reader understands it is a
judgement the usability test makes.
"""

from __future__ import annotations

import os
import re
from html.parser import HTMLParser
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy") and find_spec("sklearn")),
    reason="the quantitative extra is required",
)

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
def _modules(*names):
    return lambda: all(find_spec(n) for n in names)


def _env(variable):
    return lambda: bool(os.environ.get(variable))


# Every example whose report can be written here, with what it needs beyond the core. The
# GB3 ensemble is left out: it takes tens of minutes on a CPU, and its report is checked in the
# Docker verification run.
REQUIREMENTS = {
    "kinetics-from-measurements": lambda: True,
    "representative-panel": lambda: True,
    "measured-variant-rounds": lambda: True,
    "operation-from-progress-curves": lambda: True,
    "cofactor-regeneration": lambda: True,
    "connected-investigation": lambda: True,
    "protein-first-look": _modules("gemmi", "mdtraj"),
    "variant-to-construct": _modules("pydna", "Bio", "dnachisel"),
    "transformation-and-routes": _modules("rdkit", "doranet", "equilibrator_api"),
    "relatives-and-context": _env("ALAMBIQ_ENV_MMSEQS"),
    "lanmodulin-comparison": lambda: bool(
        find_spec("gemmi") and os.environ.get("ALAMBIQ_ENV_MMSEQS")
    ),
    "transformation-to-enzymes": _env("ALAMBIQ_ENV_MMSEQS"),
    "pockets-and-complex": lambda: bool(
        find_spec("gemmi") and os.environ.get("ALAMBIQ_ENV_P2RANK")
    ),
}
ENGINE_FREE = list(REQUIREMENTS)
SNAPSHOTS = Path(__file__).resolve().parents[1] / "snapshots/reports"
VOID = {"meta", "link", "br", "hr", "img", "input", "path", "circle", "line", "rect"}
SVG_SELF_CLOSING = {"path", "circle", "line", "rect", "polyline", "polygon", "ellipse"}
REPRESENTATIONS = [
    r"\{'",
    r"\[\(",
    r"<[A-Za-z_.]+ object at 0x",
    r"PosixPath\(",
    r"Decimal\(",
    r"array\(",
    r"datetime\.",
    r"Traceback \(most recent",
]
UNIT_HINT = re.compile(
    r"\(.+\)|\[.+\]|\bfraction\b|\brank\b|\bcount\b|\bindex\b|\bposition\b|"
    r"\bn\b|\bnumber\b|\bP\b|\bR²\b|\bp-value\b|\bprobability\b|\bscore\b|"
    r"\bR-hat\b|%|\bratio\b|\bcoverage\b|\bidentity\b|\bversion\b|\bseed\b|"
    r"\brows\b|\bcolumns\b|\bframes\b|\bsteps\b|\bruns\b|\bcomputed\b|\breused\b|"
    r"\bfailed\b|\bdirection\b|\bweight\b|\bstart\b|\bend\b|\bcosine distance\b|"
    r"\bcorrelation\b|\bcondition\b|\bAIC\b|\bBIC\b|\bevidence\b|\bcandidates\b|"
    r"\bobservations\b|\bhomologs\b|\bresidues\b|\batoms\b|\bpoints\b|\bfeatures\b|"
    r"\bgaps\b|\bloci\b|\breferences\b|\bmatches\b|\bviolations\b|\bshare\b|\bbits\b|"
    r"\bE-value\b|\bln [KQ]|\bbond order\b|\bcoefficient\b|\boffset\b|\bcharge\b|/ error",
    re.IGNORECASE,
)


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.text: list[str] = []
        self.refs: list[tuple[str, str, str]] = []
        self.title = ""
        self.lang = None
        self.charset = False
        self.summaries: list[str] = []
        self.sections: list[str] = []
        self.tables: list[dict] = []
        self.outline: list[str] = []
        self._heading = None
        self._note = None
        self._in = {"title": False, "summary": False, "script": False, "style": False}
        self._table = None
        self._cell = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag not in VOID and tag not in SVG_SELF_CLOSING:
            self.stack.append(tag)
        if tag == "html":
            self.lang = attrs.get("lang")
        if tag == "meta" and "charset" in attrs:
            self.charset = True
        for name in ("src", "href", "action", "data", "poster"):
            value = attrs.get(name)
            if value and tag != "a":
                self.refs.append((tag, name, value))
        if tag == "link":
            self.refs.append((tag, "rel", attrs.get("rel", "")))
        if tag == "section":
            self.sections.append(attrs.get("id", ""))
            self.outline.append(f"section {attrs.get('id', '')}")
        if tag in {"h1", "h2", "h3"}:
            self._heading = [tag, ""]
        if tag == "div" and {"note", "failed"} & set(attrs.get("class", "").split()):
            self._note = [attrs["class"], ""]
        if tag in self._in:
            self._in[tag] = True
        if tag == "table":
            self._table = {"headers": [], "rows": []}
        if tag == "tr" and self._table is not None:
            self._table["rows"].append([])
        if tag in {"th", "td"} and self._table is not None:
            self._cell = {"tag": tag, "class": attrs.get("class", ""), "text": ""}

    def handle_endtag(self, tag):
        if tag in VOID or tag in SVG_SELF_CLOSING:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.stack[-1:] or 'nothing'}")
            if tag in self.stack:
                while self.stack and self.stack.pop() != tag:
                    pass
        else:
            self.stack.pop()
        if tag in self._in:
            self._in[tag] = False
        if tag in {"h1", "h2", "h3"} and self._heading is not None:
            # The step's timing badge (or "reused") is part of its heading; leave it out.
            heading = re.sub(r"(\d+\.\d+s|reused)$", "", " ".join(self._heading[1].split()))
            self.outline.append(f"{tag}: {heading}")
            self._heading = None
        if tag == "div" and self._note is not None:
            self.outline.append(f"{self._note[0]}: {' '.join(self._note[1].split())}")
            self._note = None
        if tag in {"th", "td"} and self._table is not None and self._cell is not None:
            if self._cell["tag"] == "th":
                self._table["headers"].append(self._cell["text"].strip())
            else:
                self._table["rows"][-1].append(self._cell)
            self._cell = None
        if tag == "table" and self._table is not None:
            self.outline.append("table: " + " | ".join(self._table["headers"]))
            self.tables.append(self._table)
            self._table = None

    def handle_data(self, data):
        if self._in["title"]:
            self.title += data
        if self._in["summary"]:
            self.summaries.append(data)
            self.outline.append(f"details: {data.strip()}")
        if self._cell is not None:
            self._cell["text"] += data
        if self._heading is not None:
            self._heading[1] += data
        if self._note is not None:
            self._note[1] += data
        if not (self._in["script"] or self._in["style"]):
            self.text.append(data)

    def handle_startendtag(self, tag, attrs):
        # <path .../> style tags: parse as a start tag without pushing it.
        self.handle_starttag(tag, attrs)
        if tag not in VOID and tag not in SVG_SELF_CLOSING and self.stack:
            self.stack.pop()


def parse(html: str) -> Page:
    page = Page()
    page.feed(html)
    page.close()
    return page


class Reports:
    """The report of each example, written on first use and kept for the module."""

    def __init__(self, factory):
        self.factory, self.pages = factory, {}

    def __getitem__(self, name):
        if not REQUIREMENTS[name]():
            pytest.skip(f"{name} needs an engine or a package that is not available")
        if name not in self.pages:
            from alambiq.api import run

            directory = self.factory.mktemp(name)
            outcome = run(EXAMPLES / name / "campaign.yaml", run_dir=directory / "run")
            assert outcome.ok, (
                name,
                [s.error for s in outcome.steps.values() if s.error],
            )
            self.pages[name] = outcome.report(directory / "report.html").read_text()
        return self.pages[name]


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    return Reports(tmp_path_factory)


@pytest.mark.parametrize("name", ENGINE_FREE)
def test_the_outline_matches_its_snapshot(reports, name):
    outline = "\n".join(parse(reports[name]).outline) + "\n"
    target = SNAPSHOTS / f"{name}.txt"
    if os.environ.get("ALAMBIQ_UPDATE_SNAPSHOTS"):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(outline)
    assert target.is_file(), f"no snapshot for {name}; set ALAMBIQ_UPDATE_SNAPSHOTS=1"
    assert outline == target.read_text(), (
        f"the outline of the {name} report changed; if that is deliberate, set "
        "ALAMBIQ_UPDATE_SNAPSHOTS=1 and review the difference"
    )


@pytest.mark.parametrize("name", ENGINE_FREE)
def test_document_is_well_formed(reports, name):
    page = parse(reports[name])
    assert page.errors == []
    assert page.stack == []
    assert page.title.strip()
    assert page.lang == "en"
    assert page.charset


@pytest.mark.parametrize("name", ENGINE_FREE)
def test_nothing_is_loaded_from_the_network(reports, name):
    html = reports[name]
    page = parse(html)
    loaded = [
        r
        for r in page.refs
        if not r[2].startswith(("#", "data:")) and r[1] in {"src", "href", "data"}
    ]
    assert loaded == []
    assert not [r for r in page.refs if r[0] == "link"]
    assert "@import" not in html
    assert not re.search(r"url\(\s*['\"]?(?:https?:)?//", html)
    assert not re.search(r"\b(fetch|XMLHttpRequest|WebSocket|sendBeacon)\s*\(", html)


@pytest.mark.parametrize("name", ENGINE_FREE)
def test_no_internal_representation_reaches_the_reader(reports, name):
    text = " ".join(parse(reports[name]).text)
    for pattern in REPRESENTATIONS:
        assert not re.search(pattern, text), (name, pattern)
    for word in ("NaN", "nan", "inf", "-inf", "None", "null", "undefined"):
        assert not re.search(rf"(?<![\w.-]){re.escape(word)}(?![\w.-])", text), (
            name,
            word,
        )


@pytest.mark.parametrize("name", ENGINE_FREE)
def test_numeric_columns_say_what_they_count(reports, name):
    unlabelled = []
    for table in parse(reports[name]).tables:
        for column, header in enumerate(table["headers"]):
            cells = [row[column] for row in table["rows"] if column < len(row)]
            if cells and all("num" in c["class"].split() for c in cells):
                # A table of "Quantity" and "Value" rows states each unit in its row label.
                if (
                    "Unit" not in table["headers"]
                    and not (header == "Value" and table["headers"][0] != "Coefficient")
                    and not UNIT_HINT.search(header)
                ):
                    unlabelled.append(header)
    assert unlabelled == []


# ------------------------------------------------------------ the lint catches what it names


def test_the_parser_reports_unbalanced_and_external_markup():
    page = parse(
        '<html lang="en"><head><title>t</title></head><body><div><p>text</div>'
        '<img src="https://example.org/x.png"><script src="//cdn.example/x.js"></script>'
        "</body></html>"
    )
    assert page.errors
    assert {r[2] for r in page.refs} == {
        "https://example.org/x.png",
        "//cdn.example/x.js",
    }


def test_the_parser_finds_a_numeric_column_with_no_unit():
    page = parse(
        "<table><thead><tr><th>Mass</th><th>Name</th></tr></thead><tbody>"
        '<tr><td class="num">1.5</td><td>a</td></tr></tbody></table>'
    )
    (table,) = page.tables
    cells = [row[0] for row in table["rows"] if row]
    assert all("num" in c["class"].split() for c in cells)
    assert not UNIT_HINT.search(table["headers"][0])
    assert UNIT_HINT.search("Mass (g)")
