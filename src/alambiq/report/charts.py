"""Charts for the report, drawn as inline SVG that contains no script and makes no network
request.

Every plotted value is also written into the SVG as data: each marker has `data-x` and
`data-y`, each line has its points, and the root records how data maps to pixels. A chart
can therefore be parsed back and its geometry compared with the numbers it was given,
which is how the tests check that a chart shows its data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from html import escape

PALETTE = (
    "#0072b2",
    "#d55e00",
    "#009e73",
    "#e69f00",
    "#56b4e9",
    "#cc79a7",
    "#f0e442",
    "#999999",
)
MAX_LINE_POINTS = 2000
MAX_MARKERS = 5000
MARGIN = {"left": 64, "right": 16, "top": 28, "bottom": 52}


@dataclass(frozen=True, slots=True)
class Series:
    """One set of points or one curve.

    mode: `line`, `points` or `both`. band: lower and upper bounds at each x, drawn as a
    shaded region. labels: a text for each point, shown when it is hovered.
    """

    label: str
    x: Sequence[float]
    y: Sequence[float]
    mode: str = "line"
    color: str | None = None
    dashed: bool = False
    band: tuple[Sequence[float], Sequence[float]] | None = None
    labels: Sequence[str] | None = None
    radius: float = 3.5


@dataclass(slots=True)
class _Axis:
    low: float
    high: float
    log: bool
    start: float
    end: float
    ticks: list[float] = field(default_factory=list)

    def at(self, value: float) -> float:
        if self.log:
            fraction = (math.log10(value) - math.log10(self.low)) / (
                math.log10(self.high) - math.log10(self.low)
            )
        else:
            fraction = (value - self.low) / (self.high - self.low)
        return self.start + fraction * (self.end - self.start)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def nice_ticks(low: float, high: float, count: int = 6) -> list[float]:
    """Round tick values spanning [low, high] with about `count` steps."""
    span = high - low
    if span <= 0:
        return [low]
    raw = span / max(count - 1, 1)
    power = 10 ** math.floor(math.log10(raw))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * power
        if span / step <= count:
            break
    first = math.ceil(low / step - 1e-9) * step
    ticks, value = [], first
    while value <= high + step * 1e-9:
        ticks.append(round(value, 12))
        value += step
    return ticks


def log_ticks(low: float, high: float) -> list[float]:
    lo, hi = math.floor(math.log10(low)), math.ceil(math.log10(high))
    ticks = [m * 10.0**e for e in range(lo, hi + 1) for m in (1, 2, 5)]
    inside = [t for t in ticks if low * (1 - 1e-9) <= t <= high * (1 + 1e-9)]
    if len(inside) < 3:
        return inside or [low, high]
    if hi - lo > 3:
        return [t for t in inside if str(f"{t:.0e}").startswith("1")]
    return inside


def tick_text(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e6:
        return f"{value:.2e}".replace("e-0", "e-").replace("e+0", "e+")
    return f"{value:.6g}"


def _axis(
    values: list[float], log: bool, start: float, end: float, fixed=None
) -> _Axis:
    if fixed is not None:
        low, high = fixed
    else:
        low, high = min(values), max(values)
    if log:
        if low <= 0:
            raise ValueError("A logarithmic axis requires positive values")
        if high <= low:
            low, high = low / 2, high * 2
        elif fixed is None:
            pad = (high / low) ** 0.04
            low, high = low / pad, high * pad
        axis = _Axis(low, high, True, start, end)
        axis.ticks = log_ticks(low, high)
        return axis
    if high <= low:
        pad = abs(low) * 0.1 or 1.0
        low, high = low - pad, high + pad
    elif fixed is None:
        pad = (high - low) * 0.04
        low, high = low - pad, high + pad
    axis = _Axis(low, high, False, start, end)
    axis.ticks = [t for t in nice_ticks(low, high) if low <= t <= high]
    return axis


def _thin(x: Sequence[float], y: Sequence[float], limit: int) -> list[int]:
    n = len(x)
    if n <= limit:
        return list(range(n))
    step = (n - 1) / (limit - 1)
    return sorted({round(i * step) for i in range(limit)})


def _number(value: float) -> str:
    return f"{value:.10g}"


def line_chart(
    series: Sequence[Series],
    *,
    x_label: str,
    y_label: str,
    title: str = "",
    x_log: bool = False,
    y_log: bool = False,
    width: int = 640,
    height: int = 380,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> str:
    """A chart of lines, points and shaded bands, as an `<svg>` string."""
    clean: list[tuple[Series, list[int]]] = []
    xs: list[float] = []
    ys: list[float] = []
    for item in series:
        if len(item.x) != len(item.y):
            raise ValueError(f"Series {item.label!r} has x and y of different lengths")
        keep = [
            i
            for i in range(len(item.x))
            if _finite(item.x[i])
            and _finite(item.y[i])
            and (not x_log or item.x[i] > 0)
            and (not y_log or item.y[i] > 0)
        ]
        if item.band is not None:
            lower, upper = item.band
            keep = [i for i in keep if _finite(lower[i]) and _finite(upper[i])]
            if y_log:
                keep = [i for i in keep if lower[i] > 0]
        clean.append((item, keep))
        xs += [item.x[i] for i in keep]
        ys += [item.y[i] for i in keep]
        if item.band is not None:
            ys += [item.band[0][i] for i in keep] + [item.band[1][i] for i in keep]
    if not xs:
        return empty_chart(title or "No data to plot", width, height)
    plot = {
        "left": MARGIN["left"],
        "right": width - MARGIN["right"],
        "top": MARGIN["top"],
        "bottom": height - MARGIN["bottom"],
    }
    ax = _axis(xs, x_log, plot["left"], plot["right"], x_range)
    ay = _axis(ys, y_log, plot["bottom"], plot["top"], y_range)
    out = [
        f'<svg class="bf-chart" role="img" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        f'data-kind="line" '
        f'aria-label="{escape(title or y_label + " against " + x_label, quote=True)}" '
        f'data-x-scale="{"log" if x_log else "linear"}" '
        f'data-y-scale="{"log" if y_log else "linear"}" '
        f'data-x-low="{_number(ax.low)}" data-x-high="{_number(ax.high)}" '
        f'data-y-low="{_number(ay.low)}" data-y-high="{_number(ay.high)}" '
        f'data-px-left="{plot["left"]}" data-px-right="{plot["right"]}" '
        f'data-px-top="{plot["top"]}" data-px-bottom="{plot["bottom"]}">'
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    out.append('<g class="grid" stroke="currentColor" stroke-opacity="0.15" stroke-width="1">')
    for t in ax.ticks:
        px = ax.at(t)
        out.append(
            f'<line x1="{px:.2f}" x2="{px:.2f}" y1="{plot["top"]}" y2="{plot["bottom"]}"/>'
        )
    for t in ay.ticks:
        py = ay.at(t)
        out.append(
            f'<line x1="{plot["left"]}" x2="{plot["right"]}" y1="{py:.2f}" y2="{py:.2f}"/>'
        )
    out.append("</g>")
    out.append(
        '<g class="axes" fill="GrayText" font-size="11" '
        'font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
    )
    out.append(
        f'<line x1="{plot["left"]}" x2="{plot["right"]}" y1="{plot["bottom"]}" '
        f'y2="{plot["bottom"]}" stroke="GrayText"/>'
        f'<line x1="{plot["left"]}" x2="{plot["left"]}" y1="{plot["top"]}" '
        f'y2="{plot["bottom"]}" stroke="GrayText"/>'
    )
    for t in ax.ticks:
        out.append(
            f'<text class="xtick" x="{ax.at(t):.2f}" y="{plot["bottom"] + 15}" '
            f'text-anchor="middle">{escape(tick_text(t))}</text>'
        )
    for t in ay.ticks:
        out.append(
            f'<text class="ytick" x="{plot["left"] - 6}" y="{ay.at(t) + 4:.2f}" '
            f'text-anchor="end">{escape(tick_text(t))}</text>'
        )
    out.append(
        f'<text class="xlabel" x="{(plot["left"] + plot["right"]) / 2}" '
        f'y="{height - 8}" text-anchor="middle" font-size="12">{escape(x_label)}</text>'
        f'<text class="ylabel" transform="translate(14,{(plot["top"] + plot["bottom"]) / 2}) '
        f'rotate(-90)" text-anchor="middle" font-size="12">{escape(y_label)}</text>'
    )
    out.append("</g>")
    for index, (item, keep) in enumerate(clean):
        if not keep:
            continue
        color = item.color or PALETTE[index % len(PALETTE)]
        out.append(f'<g class="series" data-series="{escape(item.label, quote=True)}">')
        order = sorted(keep, key=lambda i: item.x[i])
        if item.band is not None:
            lower, upper = item.band
            top = [f"{ax.at(item.x[i]):.2f},{ay.at(upper[i]):.2f}" for i in order]
            bottom = [
                f"{ax.at(item.x[i]):.2f},{ay.at(lower[i]):.2f}" for i in reversed(order)
            ]
            out.append(
                f'<polygon class="band" fill="{color}" fill-opacity="0.15" '
                f'stroke="none" points="{" ".join(top + bottom)}"/>'
            )
        if item.mode in ("line", "both") and len(order) > 1:
            shown = [order[i] for i in _thin(order, order, MAX_LINE_POINTS)]
            pixels = " ".join(
                f"{ax.at(item.x[i]):.2f},{ay.at(item.y[i]):.2f}" for i in shown
            )
            data = ";".join(f"{_number(item.x[i])},{_number(item.y[i])}" for i in shown)
            dash = ' stroke-dasharray="6 4"' if item.dashed else ""
            out.append(
                f'<polyline class="line" fill="none" stroke="{color}" '
                f'stroke-width="2"{dash} points="{pixels}" data-points="{data}"/>'
            )
        if item.mode in ("points", "both"):
            for i in order[:MAX_MARKERS]:
                text = (
                    item.labels[i]
                    if item.labels is not None
                    else f"{item.label}: {tick_text(item.x[i])}, {tick_text(item.y[i])}"
                )
                out.append(
                    f'<circle class="pt" cx="{ax.at(item.x[i]):.2f}" '
                    f'cy="{ay.at(item.y[i]):.2f}" r="{item.radius}" fill="{color}" '
                    f'fill-opacity="0.85" data-x="{_number(item.x[i])}" '
                    f'data-y="{_number(item.y[i])}"><title>{escape(text)}</title></circle>'
                )
        out.append("</g>")
    out.append(
        _legend(
            [
                (s.label, s.color or PALETTE[i % len(PALETTE)], s.mode, s.dashed)
                for i, (s, k) in enumerate(clean)
                if k
            ],
            plot,
            width,
        )
    )
    out.append("</svg>")
    return "".join(out)


def _legend(entries, plot, width) -> str:
    if len(entries) < 2:
        return ""
    rows = ['<g class="legend" font-size="11" fill="currentColor">']
    y = plot["top"] + 4
    x = plot["left"] + 10
    for label, color, mode, dashed in entries[:12]:
        dash = ' stroke-dasharray="4 3"' if dashed else ""
        mark = (
            f'<rect x="{x}" y="{y - 5}" width="14" height="10" fill="{color}" '
            'fill-opacity="0.25"/>'
            if mode == "band"
            else f'<circle cx="{x + 6}" cy="{y}" r="4" fill="{color}"/>'
            if mode == "points"
            else f'<line x1="{x}" x2="{x + 14}" y1="{y}" y2="{y}" stroke="{color}" '
            f'stroke-width="2"{dash}/>'
        )
        rows.append(
            f'<g class="legend-entry">{mark}<text x="{x + 20}" y="{y + 4}">'
            f"{escape(label)}</text></g>"
        )
        y += 16
    rows.append("</g>")
    return "".join(rows)


def empty_chart(message: str, width: int = 640, height: int = 120) -> str:
    return (
        f'<svg class="bf-chart" role="img" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        f'data-kind="empty"><text x="{width / 2}" y="{height / 2}" '
        f'text-anchor="middle" fill="GrayText" font-size="13">{escape(message)}</text></svg>'
    )


def bar_chart(
    items: Sequence[tuple[str, float, bool]],
    *,
    x_label: str,
    title: str = "",
    width: int = 640,
    limit: int = 40,
    highlight_label: str = "selected",
) -> str:
    """Horizontal bars, one per item; highlighted items are drawn in the accent colour.

    Each item is (label, value, highlighted). Bars start at zero, or at the lowest value
    when every value is negative.
    """
    rows = [(name, v, h) for name, v, h in items if _finite(v)][:limit]
    if not rows:
        return empty_chart(title or "No values to plot", width)
    row = 20
    left = min(260, 14 + 7 * max(len(name) for name, _, _ in rows))
    top, bottom = 24, 44
    height = top + row * len(rows) + bottom
    values = [v for _, v, _ in rows]
    low, high = min(0.0, min(values)), max(0.0, max(values))
    if high == low:
        high = low + 1.0
    axis = _Axis(low, high, False, left, width - 20)
    axis.ticks = [t for t in nice_ticks(low, high) if low <= t <= high]
    zero = axis.at(0.0)
    out = [
        f'<svg class="bf-chart" role="img" '
        f'viewBox="0 0 {width} {height}" width="100%" '
        f'data-kind="bar" aria-label="{escape(title or x_label, quote=True)}" '
        f'data-x-low="{_number(axis.low)}" data-x-high="{_number(axis.high)}" '
        f'data-px-left="{left}" data-px-right="{width - 20}">'
    ]
    if title:
        out.append(f"<title>{escape(title)}</title>")
    out.append('<g class="grid" stroke="currentColor" stroke-opacity="0.15">')
    for t in axis.ticks:
        out.append(
            f'<line x1="{axis.at(t):.2f}" x2="{axis.at(t):.2f}" y1="{top}" '
            f'y2="{top + row * len(rows)}"/>'
        )
    out.append("</g>")
    out.append(
        '<g font-size="11" fill="GrayText" '
        'font-family="-apple-system,Segoe UI,Roboto,sans-serif">'
    )
    for t in axis.ticks:
        out.append(
            f'<text class="xtick" x="{axis.at(t):.2f}" '
            f'y="{top + row * len(rows) + 14}" text-anchor="middle">'
            f"{escape(tick_text(t))}</text>"
        )
    out.append(
        f'<text x="{(left + width - 20) / 2}" y="{height - 8}" text-anchor="middle" '
        f'font-size="12">{escape(x_label)}</text>'
    )
    for i, (label, value, chosen) in enumerate(rows):
        y = top + i * row
        x0, x1 = sorted((zero, axis.at(value)))
        color = PALETTE[0] if chosen else "#999999"
        out.append(
            f'<g class="bar-row"><text class="bar-label" x="{left - 6}" y="{y + 14}" '
            f'text-anchor="end" fill="currentColor">{escape(label)}</text>'
            f'<rect class="bar" x="{x0:.2f}" y="{y + 3}" width="{max(x1 - x0, 0.5):.2f}" '
            f'height="{row - 6}" fill="{color}" data-label="{escape(label, quote=True)}" '
            f'data-value="{_number(value)}" data-highlighted="{str(chosen).lower()}">'
            f"<title>{escape(label)}: {tick_text(value)}"
            f"{' (' + highlight_label + ')' if chosen else ''}</title></rect></g>"
        )
    out.append("</g></svg>")
    return "".join(out)
