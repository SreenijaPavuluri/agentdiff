"""Render an analysis bundle to one self-contained HTML file (no external assets)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape
from pydantic import BaseModel, Field

from agentdiff.cluster import FailureMode
from agentdiff.divergence import DivergenceResult
from agentdiff.gate import GateResult
from agentdiff.reliability import SuiteReliability
from agentdiff.replay import ReplayReport


class AnalysisBundle(BaseModel):
    title: str = "agentdiff report"
    generated_at: str = Field(default_factory=lambda: datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"))
    reliability: SuiteReliability | None = None
    divergences: list[DivergenceResult] = Field(default_factory=list)
    failure_modes: list[FailureMode] = Field(default_factory=list)
    replays: list[ReplayReport] = Field(default_factory=list)
    gate: GateResult | None = None
    cost_usd: float = 0.0
    notes: list[str] = Field(default_factory=list)


def decay_svg(rel: SuiteReliability) -> str:
    """pass@k and pass^k vs k with CI whiskers; two series, legend + direct labels, hover titles."""
    w, h, left, right, top, bottom = 560, 240, 44, 90, 16, 34
    ks = rel.ks
    xs = {k: left + (w - left - right) * (i / max(1, len(ks) - 1)) for i, k in enumerate(ks)}

    def y(v: float) -> float:
        return top + (h - top - bottom) * (1 - v)

    parts = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Reliability decay: pass@k and pass^k by k" class="chart">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<line x1="{left}" x2="{w - right}" y1="{y(g):.1f}" y2="{y(g):.1f}" class="grid"/>'
                     f'<text x="{left - 6}" y="{y(g) + 4:.1f}" class="axis" text-anchor="end">{g:.2f}</text>')  # fmt: skip
    for k in ks:
        parts.append(f'<text x="{xs[k]:.1f}" y="{h - 12}" class="axis" text-anchor="middle">k={k}</text>')
    series = [("pass@", "s1", "pass@k"), ("pass^", "s2", "pass^k")]
    for prefix, cls, name in series:
        pts = []
        for k in ks:
            key = "pass@1" if k == 1 else f"{prefix}{k}"
            e = rel.metrics.get(key)
            if e is None:
                continue
            pts.append((k, e))
        if not pts:
            continue
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs[k]:.1f},{y(e.value):.1f}" for i, (k, e) in enumerate(pts))
        parts.append(f'<path d="{d}" class="line {cls}"/>')
        for k, e in pts:
            parts.append(
                f'<g class="pt"><line x1="{xs[k]:.1f}" x2="{xs[k]:.1f}" y1="{y(e.lo):.1f}" y2="{y(e.hi):.1f}" class="whisker {cls}"/>'
                f'<circle cx="{xs[k]:.1f}" cy="{y(e.value):.1f}" r="4.5" class="dot {cls}"/>'
                f'<circle cx="{xs[k]:.1f}" cy="{y(e.value):.1f}" r="12" class="hit"><title>{name}, k={k}: '
                f'{e.value:.3f} (95% CI {e.lo:.3f}-{e.hi:.3f})</title></circle></g>'
            )
        k_last, e_last = pts[-1]
        parts.append(f'<text x="{xs[k_last] + 10:.1f}" y="{y(e_last.value) + 4:.1f}" class="label">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def render(bundle: AnalysisBundle, path: str | Path) -> Path:
    env = Environment(loader=FileSystemLoader(Path(__file__).parent), autoescape=select_autoescape(["html", "j2"]))
    env.filters["pct"] = lambda v: "-" if v is None else f"{100 * v:.1f}%"
    env.filters["f3"] = lambda v: "-" if v is None else f"{v:.3f}"
    env.filters["pv"] = lambda v: "-" if v is None else (f"{v:.1e}" if v < 1e-3 else f"{v:.3f}")
    html = env.get_template("template.html.j2").render(
        b=bundle, decay=decay_svg(bundle.reliability) if bundle.reliability else "",
        div_by_task={d.task_id: d for d in bundle.divergences},
    )  # fmt: skip
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out
