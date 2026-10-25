"""The report's charts, alignment view and structure viewer show the data they were given.

A chart is checked by parsing it back: each marker and line carries its data, and the
root records how data maps to pixels, so the tests invert the geometry and compare the
result with the numbers that went in. A chart that merely carried the right attributes
but drew something else would fail the geometry check, and one that drew the right
picture from the wrong numbers would fail the data check.

The pages are then checked as a reader's browser meets them: no external request is made,
the structure viewer loads from the bundled library and draws, and a page with no
structure carries no script at all.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from html.parser import HTMLParser
from importlib.util import find_spec
from pathlib import Path

import pytest

from alambiq.report import charts, viewers
from alambiq.report.charts import Series, bar_chart, line_chart
from tests.core.test_engineering_capabilities import (  # noqa: F401
    assay,
    fit_step,
    panel,
)

NS = "{http://www.w3.org/2000/svg}"
ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests/fixtures"

needs_science = pytest.mark.skipif(
    not (find_spec("numpy") and find_spec("scipy")),
    reason="the quantitative extra is required",
)


def parse(svg: str) -> ET.Element:
    return ET.fromstring(
        svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    )


class Geometry:
    """Pixels back to data, from the calibration the chart records on its root."""

    def __init__(self, root: ET.Element):
        a = root.attrib
        self.left, self.right = float(a["data-px-left"]), float(a["data-px-right"])
        self.top, self.bottom = float(a["data-px-top"]), float(a["data-px-bottom"])
        self.x_low, self.x_high = float(a["data-x-low"]), float(a["data-x-high"])
        self.y_low, self.y_high = float(a["data-y-low"]), float(a["data-y-high"])
        self.x_log = a["data-x-scale"] == "log"
        self.y_log = a["data-y-scale"] == "log"

    def agrees(self, px: float, py: float, x: float, y: float) -> bool:
        """Whether the pixel is where (x, y) belongs, to a ten-thousandth of each axis."""
        return (
            abs(self.x(px) - x) <= (self.x_high - self.x_low) * 1e-4
            if not self.x_log
            else close(self.x(px), x, 2e-4)
        ) and (
            abs(self.y(py) - y) <= (self.y_high - self.y_low) * 1e-4
            if not self.y_log
            else close(self.y(py), y, 2e-4)
        )

    def x(self, px: float) -> float:
        fraction = (px - self.left) / (self.right - self.left)
        if self.x_log:
            lo, hi = math.log10(self.x_low), math.log10(self.x_high)
            return 10 ** (lo + fraction * (hi - lo))
        return self.x_low + fraction * (self.x_high - self.x_low)

    def y(self, py: float) -> float:
        fraction = (self.bottom - py) / (self.bottom - self.top)
        if self.y_log:
            lo, hi = math.log10(self.y_low), math.log10(self.y_high)
            return 10 ** (lo + fraction * (hi - lo))
        return self.y_low + fraction * (self.y_high - self.y_low)


def series_of(root: ET.Element, label: str) -> ET.Element:
    return next(g for g in root.iter(f"{NS}g") if g.get("data-series") == label)


def markers(root: ET.Element, label: str) -> list[tuple[float, float, float, float]]:
    """(data x, data y, pixel x, pixel y) for every marker of a series."""
    return [
        (
            float(c.get("data-x")),
            float(c.get("data-y")),
            float(c.get("cx")),
            float(c.get("cy")),
        )
        for c in series_of(root, label).iter(f"{NS}circle")
    ]


def line_points(root: ET.Element, label: str) -> list[tuple[float, float]]:
    line = next(series_of(root, label).iter(f"{NS}polyline"))
    return [
        tuple(map(float, pair.split(",")))
        for pair in line.get("data-points").split(";")
    ]


def line_pixels(root: ET.Element, label: str) -> list[tuple[float, float]]:
    line = next(series_of(root, label).iter(f"{NS}polyline"))
    return [tuple(map(float, pair.split(","))) for pair in line.get("points").split()]


def flat(pairs) -> list[float]:
    return [v for pair in pairs for v in pair]


def close(a: float, b: float, rel: float = 1e-5) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=1e-9)


class TestChartGeometry:
    X = [0.5, 1.0, 2.0, 4.0, 8.0]
    Y = [3.0, 1.0, 4.0, 1.5, 5.0]

    def test_every_marker_carries_its_data_and_is_drawn_where_the_data_says(self):
        root = parse(
            line_chart(
                [Series("s", self.X, self.Y, mode="points")], x_label="x", y_label="y"
            )
        )
        geo = Geometry(root)
        found = markers(root, "s")
        assert [(x, y) for x, y, _, _ in found] == sorted(
            zip(self.X, self.Y, strict=True)
        )
        for x, y, px, py in found:
            assert geo.agrees(px, py, x, y)

    def test_a_line_carries_its_points_and_its_pixels_invert_to_them(self):
        root = parse(
            line_chart([Series("s", self.X, self.Y)], x_label="x", y_label="y")
        )
        geo = Geometry(root)
        data = line_points(root, "s")
        assert data == sorted(zip(self.X, self.Y, strict=True))
        for (x, y), (px, py) in zip(data, line_pixels(root, "s"), strict=True):
            assert geo.agrees(px, py, x, y)

    def test_logarithmic_axes_place_points_by_their_logarithm(self):
        root = parse(
            line_chart(
                [Series("s", [1, 10, 100, 1000], [0.01, 0.1, 1.0, 10.0], mode="both")],
                x_label="x",
                y_label="y",
                x_log=True,
                y_log=True,
            )
        )
        geo = Geometry(root)
        assert geo.x_log and geo.y_log
        for x, y, px, py in markers(root, "s"):
            assert geo.agrees(px, py, x, y)
        pixels = line_pixels(root, "s")
        # Equal ratios are equal distances on a logarithmic axis.
        gaps = [b[0] - a[0] for a, b in zip(pixels, pixels[1:], strict=False)]
        assert max(gaps) - min(gaps) < 0.05

    def test_a_band_is_drawn_between_its_bounds(self):
        lower, upper = [2.0, 0.5, 3.0, 0.5, 4.0], [4.0, 1.5, 5.0, 2.5, 6.0]
        root = parse(
            line_chart(
                [Series("s", self.X, self.Y, band=(lower, upper))],
                x_label="x",
                y_label="y",
            )
        )
        geo = Geometry(root)
        polygon = next(series_of(root, "s").iter(f"{NS}polygon"))
        vertices = [
            tuple(map(float, v.split(","))) for v in polygon.get("points").split()
        ]
        upper_edge = [geo.y(py) for _, py in vertices[: len(self.X)]]
        lower_edge = [geo.y(py) for _, py in reversed(vertices[len(self.X) :])]
        assert all(close(a, b, 2e-4) for a, b in zip(upper_edge, upper, strict=True))
        assert all(close(a, b, 2e-4) for a, b in zip(lower_edge, lower, strict=True))

    def test_a_band_only_series_draws_no_line_or_marker(self):
        root = parse(
            line_chart(
                [Series("s", self.X, self.Y, mode="band", band=([0] * 5, [9] * 5))],
                x_label="x",
                y_label="y",
            )
        )
        group = series_of(root, "s")
        assert list(group.iter(f"{NS}polygon")) and not list(
            group.iter(f"{NS}polyline")
        )
        assert not list(group.iter(f"{NS}circle"))

    def test_values_that_are_not_numbers_are_left_out_not_drawn_as_zero(self):
        root = parse(
            line_chart(
                [
                    Series(
                        "s",
                        [1, 2, 3, 4],
                        [1.0, float("nan"), 3.0, float("inf")],
                        mode="points",
                    )
                ],
                x_label="x",
                y_label="y",
            )
        )
        assert [(x, y) for x, y, _, _ in markers(root, "s")] == [(1.0, 1.0), (3.0, 3.0)]

    def test_a_logarithmic_axis_leaves_out_values_it_cannot_show(self):
        root = parse(
            line_chart(
                [Series("s", [1, 2, 3], [0.0, 2.0, 4.0], mode="points")],
                x_label="x",
                y_label="y",
                y_log=True,
            )
        )
        assert [y for _, y, _, _ in markers(root, "s")] == [2.0, 4.0]

    def test_a_chart_with_nothing_to_draw_says_so(self):
        root = parse(
            line_chart([Series("s", [], [])], x_label="x", y_label="y", title="Nothing")
        )
        assert root.get("data-kind") == "empty"

    def test_a_long_line_is_thinned_but_keeps_its_ends(self):
        n = 5000
        x = list(range(n))
        root = parse(
            line_chart([Series("s", x, [i * 0.5 for i in x])], x_label="x", y_label="y")
        )
        points = line_points(root, "s")
        assert len(points) <= charts.MAX_LINE_POINTS
        assert points[0] == (0.0, 0.0) and points[-1] == (n - 1.0, (n - 1) * 0.5)

    def test_series_of_different_lengths_are_refused(self):
        with pytest.raises(ValueError, match="different lengths"):
            line_chart([Series("s", [1, 2], [1])], x_label="x", y_label="y")

    def test_a_label_cannot_inject_markup(self):
        svg = line_chart(
            [Series("<script>x</script>", [1, 2], [1, 2])],
            x_label="<b>x</b>",
            y_label="y",
        )
        assert "<script" not in svg and "<b>" not in svg
        parse(svg)


class TestTicks:
    def test_nice_ticks_are_round_and_inside_the_range(self):
        assert charts.nice_ticks(0, 1) == pytest.approx([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ticks = charts.nice_ticks(-3.2, 47.0)
        assert all(-3.2 <= t <= 47.0 for t in ticks)
        steps = {round(b - a, 9) for a, b in zip(ticks, ticks[1:], strict=False)}
        assert len(steps) == 1

    def test_logarithmic_ticks_are_decades_and_halves_within_the_range(self):
        assert charts.log_ticks(1, 1000) == [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]

    def test_tick_text_does_not_invent_precision(self):
        assert charts.tick_text(0) == "0"
        assert charts.tick_text(2.5) == "2.5"
        assert charts.tick_text(1e-5) == "1.00e-5"


class TestBarChart:
    def test_each_bar_is_as_long_as_its_value_and_marks_whether_it_was_chosen(self):
        items = [("a", 4.0, True), ("b", 2.0, False), ("c", -1.0, False)]
        root = parse(bar_chart(items, x_label="score"))
        low, high = float(root.get("data-x-low")), float(root.get("data-x-high"))
        left, right = float(root.get("data-px-left")), float(root.get("data-px-right"))
        scale = (right - left) / (high - low)
        bars = {r.get("data-label"): r for r in root.iter(f"{NS}rect")}
        for label, value, chosen in items:
            bar = bars[label]
            assert close(float(bar.get("width")), abs(value) * scale, 1e-4)
            assert float(bar.get("data-value")) == value
            assert bar.get("data-highlighted") == str(chosen).lower()
        zero = left + (0 - low) * scale
        assert float(bars["a"].get("x")) == pytest.approx(zero, abs=0.01)
        assert float(bars["c"].get("x")) + float(
            bars["c"].get("width")
        ) == pytest.approx(zero, abs=0.01)

    def test_nothing_to_show_says_so(self):
        assert parse(bar_chart([], x_label="x")).get("data-kind") == "empty"


class Rows(HTMLParser):
    """The alignment view's rows: their recorded sequence and the classes of their runs."""

    def __init__(self):
        super().__init__()
        self.rows: dict[str, str] = {}
        self.runs: dict[str, list[tuple[str, str]]] = {}
        self._name = None
        self._class = None
        self._in_seq = False

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "div" and "aln-row" in (a.get("class") or "") and "data-name" in a:
            self._name = a["data-name"]
            if a.get("data-sequence"):
                self.rows[self._name] = a["data-sequence"]
            self.runs.setdefault(self._name, [])
        elif tag == "span" and a.get("class") == "aln-seq" and self._name:
            self._in_seq = True
        elif tag == "span" and self._in_seq:
            self._class = a.get("class")

    def handle_data(self, data):
        if self._in_seq and self._class:
            self.runs[self._name].append((self._class, data))

    def handle_endtag(self, tag):
        if tag == "span":
            self._class = None
        if tag == "div":
            self._in_seq = False


class TestAlignmentView:
    ROWS = [("query", "ACDE-FGH"), ("close", "ACDEGFGH"), ("far", "ATD--FAH")]

    def test_each_row_is_recoverable_and_each_residue_is_marked_against_the_query(self):
        parser = Rows()
        parser.feed(viewers.alignment_view(self.ROWS))
        assert parser.rows == dict(self.ROWS)
        marked = {
            name: "".join((cls * len(text)) for cls, text in parser.runs[name])
            for name in parser.runs
        }
        assert marked["query"] == "rrrrgrrr"
        assert marked["close"] == "mmmmxmmm"
        assert marked["far"] == "mxmggmxm"
        assert {
            name: "".join(text for _, text in parser.runs[name]) for name in parser.runs
        } == dict(self.ROWS)

    def test_long_alignments_wrap_into_blocks_without_losing_a_column(self):
        long = [("q", "ACDEFGHIKL" * 15), ("s", "ACDEFGHIKM" * 15)]
        html = viewers.alignment_view(long, block=60)
        parser = Rows()
        parser.feed(html)
        assert html.count('class="aln-block"') == 3
        for name, sequence in long:
            assert "".join(text for _, text in parser.runs[name]) == sequence

    def test_rows_of_different_length_are_refused(self):
        with pytest.raises(ValueError, match="same length"):
            viewers.alignment_view([("a", "ACD"), ("b", "AC")])

    def test_only_the_first_hundred_rows_are_drawn(self):
        rows = [(f"r{i}", "ACD") for i in range(150)]
        assert (
            viewers.alignment_view(rows).count('data-name="')
            == viewers.MAX_ALIGNMENT_ROWS
        )


class TestStructureViewer:
    PDB = (
        "ATOM      1  N   ALA A   1      11.104   6.134  -6.504  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      11.639   6.071  -5.147  1.00  0.00           C\n"
        "HETATM    3  O   HOH A 101       1.000   2.000   3.000  1.00  0.00           O\n"
        "HETATM    4 NA    NA A 102       4.000   5.000   6.000  1.00  0.00          NA\n"
        "HETATM    5  C1  LIG L   1       7.000   8.000   9.000  1.00  0.00           C\n"
        "END\n"
    )

    def test_the_embedded_structure_is_the_structure_given(self):
        html = viewers.structure_viewer(self.PDB, "pdb")
        body = re.search(
            r'<script type="text/plain"[^>]*>(.*?)</script>', html, re.S
        ).group(1)
        assert body == self.PDB
        assert 'data-format="pdb"' in html and viewers.VIEWER_MARKER in html

    def test_text_that_would_close_the_script_cannot_escape_it(self):
        html = viewers.structure_viewer("REMARK </script><b>x</b>\n", "pdb")
        assert html.count("</script>") == 1
        assert "<\\/script" in html

    def test_water_and_simple_ions_are_removed_from_a_solvated_box(self):
        kept = viewers.without_solvent(self.PDB)
        assert "HOH" not in kept and " NA " not in kept
        assert "ALA" in kept and "LIG" in kept

    def test_a_structure_too_large_to_embed_is_declined_not_truncated(self):
        big = "REMARK " + "x" * viewers.MAX_STRUCTURE_BYTES + "\n"
        html = viewers.structure_viewer(big, "pdb")
        assert viewers.VIEWER_MARKER not in html and "this one is larger" in html

    def test_the_bundled_library_is_the_pinned_one(self):
        import hashlib

        digest = hashlib.sha256(
            (viewers.ASSETS / "3Dmol-min.js").read_bytes()
        ).hexdigest()
        assert (
            digest == "f7cc78921ae72e7623e89cdd111434f58c2efddd2ffda1cd212644b406fb8016"
        )
        assert "BSD-3-Clause" in (viewers.ASSETS / "LICENSE").read_text()


def run_campaign(steps: list[dict], tmp_path: Path, name: str = "run"):
    from alambiq.api import run

    outcome = run({"id": "visuals", "steps": steps}, run_dir=tmp_path / name)
    assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
    return outcome


def charts_in(html: str) -> list[ET.Element]:
    return [parse(m) for m in re.findall(r"<svg .*?</svg>", html, re.S)]


def chart_titled(html: str, title: str) -> ET.Element:
    return next(c for c in charts_in(html) if c.get("aria-label") == title)


@needs_science
class TestReportsDrawTheirData:
    def test_a_fit_report_plots_the_observations_the_curve_and_both_intervals(
        self, tmp_path
    ):
        from alambiq.api import run

        outcome = run(
            str(ROOT / "examples/kinetics-from-measurements/campaign.yaml"),
            run_dir=tmp_path / "run",
        )
        assert outcome.ok, [s.error for s in outcome.steps.values() if s.error]
        stored = json.loads(Path(outcome.outputs("fit")["record"]).read_text())
        root = chart_titled(
            outcome.report(tmp_path / "r.html").read_text(),
            "Measured values and the fitted curve",
        )
        geo = Geometry(root)
        observed = sorted((o["x"], o["y"]) for o in stored["observations"])
        found = markers(root, "Observations")
        assert flat((x, y) for x, y, _, _ in found) == pytest.approx(
            flat(observed), rel=1e-9
        )
        assert all(geo.agrees(px, py, x, y) for x, y, px, py in found)
        curve = line_points(root, "Fitted curve")
        assert flat(curve) == pytest.approx(
            flat((p["x"], p["y"]) for p in stored["predictions"]), rel=1e-9
        )
        for label, key in (
            ("95 % interval of the curve", "mean_interval_95"),
            ("95 % interval of a new measurement", "prediction_interval_95"),
        ):
            polygon = next(series_of(root, label).iter(f"{NS}polygon"))
            vertices = [
                tuple(map(float, v.split(","))) for v in polygon.get("points").split()
            ]
            n = len(stored["predictions"])
            ordered = sorted(stored["predictions"], key=lambda p: p["x"])
            span = geo.y_high - geo.y_low
            upper = [geo.y(py) for _, py in vertices[:n]]
            lower = [geo.y(py) for _, py in reversed(vertices[n:])]
            for got, row in zip(upper, ordered, strict=True):
                assert abs(got - row[key][1]) < span * 1e-4
            for got, row in zip(lower, ordered, strict=True):
                assert abs(got - row[key][0]) < span * 1e-4

    def test_a_simulation_report_plots_the_first_order_decay_it_computed(
        self, tmp_path
    ):
        outcome = run_campaign(
            [
                {
                    "id": "batch",
                    "capability": "operation.simulate",
                    "params": {
                        "definition": str(FIXTURES / "operation/first-order.json")
                    },
                }
            ],
            tmp_path,
        )
        root = chart_titled(
            outcome.report(tmp_path / "r.html").read_text(), "Concentrations over time"
        )
        substrate = dict(line_points(root, "S"))
        product = dict(line_points(root, "P"))
        assert set(substrate) == {0.0, 1.0, 5.0, 10.0}
        for t, s in substrate.items():
            assert close(s, 0.01 * math.exp(-0.2 * t), 1e-5)
            assert close(s + product[t], 0.01, 1e-9)

    def test_a_comparison_report_plots_the_measurements_beside_the_predictions(
        self, tmp_path
    ):
        outcome = run_campaign(
            [
                {
                    "id": "batch",
                    "capability": "operation.simulate",
                    "params": {
                        "definition": str(FIXTURES / "operation/first-order.json")
                    },
                },
                {
                    "id": "against",
                    "capability": "operation.compare",
                    "bind": {"operation": "batch.record"},
                    "params": {
                        "comparison": str(
                            FIXTURES / "operation/first-order-observations.json"
                        )
                    },
                },
            ],
            tmp_path,
        )
        stored = json.loads(Path(outcome.outputs("against")["record"]).read_text())
        page = outcome.report(tmp_path / "r.html").read_text()
        plotted = []
        for root in charts_in(page):
            if root.get("data-kind") != "line" or not list(root.iter(f"{NS}circle")):
                continue
            plotted += [(x, y) for x, y, _, _ in markers(root, "Measured")]
        expected = sorted((o["time_s"], o["value"]) for o in stored["observations"])
        assert flat(sorted(plotted)) == pytest.approx(flat(expected), rel=1e-9)
        assert len(plotted) == len(stored["observations"]) > 0

    def test_a_selection_report_plots_the_value_each_chosen_candidate_had(
        self,
        panel,  # noqa: F811
        assay,  # noqa: F811
        tmp_path,
    ):
        steps = [
            fit_step(panel, assay),
            {
                "id": "next",
                "capability": "candidate.select",
                "bind": {"model": "model.record"},
                "params": {
                    "panel": str(panel),
                    "count": 3,
                    "seed": 7,
                    "strategy": "highest-predicted",
                },
            },
        ]
        outcome = run_campaign(steps, tmp_path)
        stored = json.loads(Path(outcome.outputs("next")["record"]).read_text())
        page = outcome.report(tmp_path / "r.html").read_text()
        root = chart_titled(page, "Chosen candidates in the order they were chosen")
        bars = [
            (r.get("data-label"), float(r.get("data-value")))
            for r in root.iter(f"{NS}rect")
        ]
        assert [b[0] for b in bars] == [c["sample_id"] for c in stored["selected"]]
        assert [b[1] for b in bars] == pytest.approx(
            [c["selection_value"] for c in stored["selected"]], rel=1e-9
        )
        assert [v for _, v in bars] == sorted((v for _, v in bars), reverse=True)

    def test_an_embedding_report_plots_every_protein_at_its_layout_position(
        self, tmp_path
    ):
        from tests.core.test_protein_explore_embeddings import PANEL, record, write

        outcome = run_campaign(
            [
                {
                    "id": "explore",
                    "capability": "protein.explore_embeddings",
                    "params": {
                        "embeddings": write(tmp_path, "panel.json", record(PANEL))
                    },
                }
            ],
            tmp_path,
        )
        out = outcome.outputs("explore")
        root = chart_titled(
            outcome.report(tmp_path / "r.html").read_text(),
            "Two-dimensional layout of the panel",
        )
        plotted = []
        for label in ("Recommended", "Other proteins"):
            try:
                plotted += [(x, y) for x, y, _, _ in markers(root, label)]
            except StopIteration:
                continue
        assert flat(sorted(plotted)) == pytest.approx(
            flat(sorted((p["x"], p["y"]) for p in out["points"])), rel=1e-9
        )
        assert len(plotted) == len(PANEL)

    def test_an_alignment_report_shows_every_row_and_the_variability_along_it(
        self, tmp_path
    ):
        alignment = tmp_path / "msa.fasta"
        rows = {"query": "ACDEFGHIKL", "near": "ACDEFGHIKM", "far": "A-DEWGHIPL"}
        alignment.write_text("".join(f">{k}\n{v}\n" for k, v in rows.items()))
        outcome = run_campaign(
            [
                {
                    "id": "msa",
                    "capability": "sequence.inspect_msa",
                    "params": {"alignment": str(alignment), "format": "fasta"},
                }
            ],
            tmp_path,
        )
        page = outcome.report(tmp_path / "r.html").read_text()
        parser = Rows()
        parser.feed(page)
        assert parser.rows == rows
        root = chart_titled(page, "Variability along the alignment")
        entropy = line_points(root, "Entropy")
        assert [x for x, _ in entropy] == list(range(1, 11))
        # Column 1 is the same residue in every row; column 5 differs in one of three.
        assert entropy[0][1] == pytest.approx(0.0, abs=1e-9)
        assert entropy[4][1] > 0.5


FORBIDDEN_REFERENCE = re.compile(
    r"""(?:src|href|action|data|poster)\s*=\s*["']\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//|@import\s+["']?(?:https?:)?//""",
    re.I,
)


@dataclass(frozen=True, slots=True)
class Request:
    """Nothing to ask for.

    nothing:
        Unused.
    """

    nothing: int = 0


@dataclass(frozen=True, slots=True)
class Result:
    """Nothing."""

    nothing: int = 0


def structure_page(tmp_path: Path) -> Path:
    """A report with one structure section, made by a capability registered for the test."""
    from alambiq.core import registry
    from alambiq.core.registry import capability

    pdb = (FIXTURES / "structures/1L2Y.cif").read_text()

    with registry.isolated():

        @capability(
            "test.show_structure",
            version="1.0",
            family="investigate",
            summary="Show a structure",
            render=lambda result: viewers.structure_viewer(
                pdb, "cif", caption="Trp-cage"
            ),
        )
        def show(request: Request, ctx) -> Result:
            """Return nothing."""
            return Result()

        from alambiq.api import run

        outcome = run(
            {
                "id": "view",
                "steps": [
                    {"id": "s", "capability": "test.show_structure", "params": {}}
                ],
            },
            run_dir=tmp_path / "run",
        )
        assert outcome.ok
        return outcome.report(tmp_path / "structure.html")


class TestThePageIsSelfContained:
    def test_a_page_without_a_structure_carries_no_script_and_no_external_reference(
        self, tmp_path
    ):
        from alambiq.api import run

        outcome = run(
            str(ROOT / "examples/kinetics-from-measurements/campaign.yaml"),
            run_dir=tmp_path / "run",
        )
        if not outcome.ok:  # pragma: no cover - the quantitative extra is missing
            pytest.skip("the example needs the quantitative extra")
        page = outcome.report(tmp_path / "r.html").read_text()
        assert "<script" not in page and "3Dmol" not in page
        assert not FORBIDDEN_REFERENCE.search(page)
        assert "http" not in page

    def test_a_page_with_a_structure_carries_the_viewer_once_and_no_external_reference(
        self, tmp_path
    ):
        page = structure_page(tmp_path).read_text()
        assert page.count("3Dmol-min") <= 1 and "createViewer" in page
        assert page.count("$3Dmol.createViewer") == 1
        assert not FORBIDDEN_REFERENCE.search(
            re.sub(r"<script>.*?</script>", "", page, flags=re.S)
        )


@pytest.fixture
def browser():
    """Playwright's synchronous API, or a skip naming what is missing."""
    return pytest.importorskip(
        "playwright.sync_api", reason="Playwright is not installed"
    )


def launch(playwright):
    try:
        return playwright.chromium.launch()
    except Exception as error:  # noqa: BLE001
        pytest.skip(f"no headless browser is available: {error}")


class TestInABrowser:
    def test_a_structure_page_makes_no_request_and_draws_the_structure(
        self, tmp_path, browser
    ):
        page_file = structure_page(tmp_path)
        requested, errors = [], []
        with browser.sync_playwright() as p:
            chromium = launch(p)
            page = chromium.new_page()
            page.on("request", lambda r: requested.append(r.url))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.on(
                "console",
                lambda m: errors.append(m.text) if m.type == "error" else None,
            )
            page.goto(page_file.as_uri())
            page.wait_for_selector(".bf-structure[data-ready]", timeout=20000)
            state = page.get_attribute(".bf-structure", "data-ready")
            canvases = page.eval_on_selector_all(
                ".bf-structure canvas", "els => els.length"
            )
            atoms = page.evaluate(
                "() => Object.values($3Dmol.viewers || {}).length ? 1 : 0"
            )
            chromium.close()
        assert errors == []
        assert all(url.startswith(("file:", "data:", "blob:")) for url in requested), (
            requested
        )
        assert state == "true" and canvases == 1
        assert atoms in (0, 1)

    def test_a_report_with_charts_loads_without_errors_and_every_chart_is_visible(
        self, tmp_path, browser
    ):
        from alambiq.api import run

        outcome = run(
            str(ROOT / "examples/kinetics-from-measurements/campaign.yaml"),
            run_dir=tmp_path / "run",
        )
        if not outcome.ok:  # pragma: no cover
            pytest.skip("the example needs the quantitative extra")
        page_file = outcome.report(tmp_path / "r.html")
        requested, errors = [], []
        with browser.sync_playwright() as p:
            chromium = launch(p)
            page = chromium.new_page()
            page.on("request", lambda r: requested.append(r.url))
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(page_file.as_uri())
            sizes = page.eval_on_selector_all(
                "svg.bf-chart",
                "els => els.map(e => [e.getBoundingClientRect().width, e.getBoundingClientRect().height])",
            )
            circles = page.eval_on_selector_all(
                "svg.bf-chart circle.pt", "els => els.length"
            )
            chromium.close()
        assert errors == []
        assert all(url.startswith("file:") for url in requested), requested
        assert sizes and all(w > 100 and h > 100 for w, h in sizes)
        assert circles == 10


class TestTheOtherChartsDrawTheirData:
    """The functions that draw trajectories, profiles and fits, called with known numbers."""

    def test_a_trajectory_observable_is_plotted_against_its_times(self):
        from alambiq.capabilities.investigate.simulation_inspect_trajectory_report import (
            _series,
        )

        times = [0.0, 10.0, 20.0, 30.0, 40.0]
        rmsd = [0.0, 1.1, 1.6, 1.9, 2.0]
        root = parse(_series(times, rmsd, "RMSD", "Å"))
        geo = Geometry(root)
        points = line_points(root, "RMSD")
        assert flat(points) == pytest.approx(flat(zip(times, rmsd, strict=True)))
        for (x, y), (px, py) in zip(points, line_pixels(root, "RMSD"), strict=True):
            assert geo.agrees(px, py, x, y)

    def test_a_free_energy_profile_is_plotted_with_its_replica_error_as_a_band(self):
        from alambiq.capabilities.investigate.simulation_analyze_reaction_free_energy_report import (
            _profile,
        )

        centres = [-1.0, -0.5, 0.0, 0.5, 1.0]
        values = [10.0, 4.0, 0.0, 6.0, 12.0]
        errors = [0.5, 0.4, 0.3, 0.4, 0.6]
        root = parse(_profile(centres, values, errors))
        geo = Geometry(root)
        series = "Mean of replicas"
        assert flat(line_points(root, series)) == pytest.approx(
            flat(zip(centres, values, strict=True))
        )
        polygon = next(series_of(root, series).iter(f"{NS}polygon"))
        vertices = [
            tuple(map(float, v.split(","))) for v in polygon.get("points").split()
        ]
        span = geo.y_high - geo.y_low
        upper = [geo.y(py) for _, py in vertices[:5]]
        lower = [geo.y(py) for _, py in reversed(vertices[5:])]
        for got, v, e in zip(upper, values, errors, strict=True):
            assert abs(got - (v + e)) < span * 1e-4
        for got, v, e in zip(lower, values, errors, strict=True):
            assert abs(got - (v - e)) < span * 1e-4

    def test_a_profile_with_missing_points_leaves_them_out(self):
        from alambiq.capabilities.investigate.simulation_analyze_reaction_free_energy_report import (
            _profile,
        )

        root = parse(
            _profile([0, 1, 2, 3], [1.0, None, 3.0, 4.0], [0.1, None, 0.1, 0.1])
        )
        assert [x for x, _ in line_points(root, "Mean of replicas")] == [0.0, 2.0, 3.0]

    def test_a_torsion_surface_is_plotted_over_a_full_turn_with_gaps_left_open(self):
        from alambiq.capabilities.investigate.simulation_run_metadynamics_report import (
            _line,
        )

        angles = [-math.pi + 2 * math.pi * i / 8 for i in range(9)]
        energies = [3.0, 2.0, None, 0.5, 0.0, 1.0, 2.5, None, 3.0]
        root = parse(_line(angles, energies))
        assert float(root.get("data-x-low")) == pytest.approx(-math.pi)
        assert float(root.get("data-x-high")) == pytest.approx(math.pi)
        shown = line_points(root, "Free energy")
        expected = [
            (a, e) for a, e in zip(angles, energies, strict=True) if e is not None
        ]
        assert flat(shown) == pytest.approx(flat(expected))

    def test_an_operating_fit_plots_each_measurement_with_its_prediction(self):
        from alambiq.capabilities.operate.progress_chart import progress_charts

        rows = [
            {
                "experiment_id": "e1",
                "quantity": "concentration",
                "target": "S",
                "unit": "M",
                "time_s": t,
                "value": v,
                "predicted": p,
            }
            for t, v, p in (
                (0.0, 0.010, 0.010),
                (5.0, 0.0038, 0.0037),
                (10.0, 0.0014, 0.0014),
            )
        ] + [
            {
                "experiment_id": "e2",
                "quantity": "concentration",
                "target": "S",
                "unit": "M",
                "time_s": 0.0,
                "value": 0.02,
                "predicted": 0.02,
            },
        ]
        roots = charts_in(progress_charts(rows))
        assert len(roots) == 2
        first = next(r for r in roots if r.get("aria-label").startswith("e1"))
        measured = [(x, y) for x, y, _, _ in markers(first, "Measured")]
        assert flat(measured) == pytest.approx(
            flat([(0.0, 0.010), (5.0, 0.0038), (10.0, 0.0014)])
        )
        assert flat(line_points(first, "Predicted")) == pytest.approx(
            flat([(0.0, 0.010), (5.0, 0.0037), (10.0, 0.0014)])
        )

    def test_a_structure_comparison_plots_every_paired_distance(self):
        from types import SimpleNamespace

        from alambiq.capabilities.investigate.structure_compare_report import _profile

        pairs = [
            SimpleNamespace(distance_before_angstrom=b, distance_after_angstrom=a)
            for b, a in ((5.0, 1.0), (4.0, 0.5), (6.0, 2.5), (3.0, 0.2))
        ]
        root = parse(_profile(SimpleNamespace(pairs=pairs)))
        assert flat(line_points(root, "After the fit")) == pytest.approx(
            flat([(1, 1.0), (2, 0.5), (3, 2.5), (4, 0.2)])
        )
        assert flat(line_points(root, "Before the fit")) == pytest.approx(
            flat([(1, 5.0), (2, 4.0), (3, 6.0), (4, 3.0)])
        )
