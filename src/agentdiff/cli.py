"""``agentdiff`` command-line interface."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from agentdiff.pipeline import analyze, compare_bundle, replay_all
from agentdiff.report.html import render
from agentdiff.runner import Harness, run_suite
from agentdiff.schema import Trajectory, read_jsonl, write_jsonl
from agentdiff.tape import Budget

app = typer.Typer(add_completion=False, help="Find out why your agent is flaky.", no_args_is_help=True)
console = Console(stderr=True)


def load_harness(spec: str, version: str) -> Harness:
    """``package.module:factory`` where ``factory(version) -> Harness`` (with ``task_ids()``)."""
    mod_name, _, attr = spec.partition(":")
    sys.path.insert(0, str(Path.cwd()))
    factory = getattr(importlib.import_module(mod_name), attr or "make_harness")
    return factory(version)


def _tasks(h: Harness, tasks: str | None) -> list[str]:
    all_ids: list[str] = h.task_ids()  # type: ignore[attr-defined]
    return [t for t in all_ids if t in tasks.split(",")] if tasks else all_ids


def _run(h: Harness, tasks: list[str], k: int, concurrency: int, budget_usd: float | None, seed0: int) -> list[Trajectory]:
    budget = Budget(max_usd=budget_usd)
    done = {"n": 0}
    total = len(tasks) * k

    def progress(t: Trajectory) -> None:
        done["n"] += 1
        console.print(f"[dim]{done['n']}/{total}[/dim] {t.task_id} seed={t.seed} {'PASS' if t.outcome.passed else 'fail'}")

    return run_suite(h, tasks, k, concurrency=concurrency, budget=budget, seed0=seed0, on_result=progress)


@app.command()
def run(
    harness: Annotated[str, typer.Option(help="module:factory returning a harness for a version")],
    version: Annotated[str, typer.Option(help="agent version passed to the factory")],
    out: Annotated[Path, typer.Option(help="output JSONL of trajectories")],
    k: int = 8,
    tasks: Annotated[str | None, typer.Option(help="comma-separated task ids (default: all)")] = None,
    concurrency: int = 4,
    budget_usd: Annotated[float | None, typer.Option(help="hard cost cap")] = None,
    seed0: int = 0,
) -> None:
    """Run each task k times and save full trajectories."""
    h = load_harness(harness, version)
    trajs = _run(h, _tasks(h, tasks), k, concurrency, budget_usd, seed0)
    write_jsonl(trajs, out)
    console.print(f"wrote {len(trajs)} runs to {out}")


@app.command("analyze")
def analyze_cmd(
    runs: Path,
    out: Annotated[Path, typer.Option(help="HTML report path")] = Path("agentdiff-report.html"),
    json_out: Annotated[Path | None, typer.Option("--json", help="also write the analysis as JSON")] = None,
    replay: Annotated[str | None, typer.Option(help="module:factory to confirm divergences by replay")] = None,
    n_forks: int = 8,
    replay_tasks: int = 3,
    replay_budget_usd: float | None = None,
) -> None:
    """Reliability, flaky-task ranking, divergence and failure modes for a run file."""
    trajs = read_jsonl(runs)
    bundle = analyze(trajs)
    if replay:
        h = load_harness(replay, trajs[0].agent_version)
        bundle.replays = replay_all(h, trajs, bundle.divergences, n_forks=n_forks, max_tasks=replay_tasks,
                                    budget=Budget(max_usd=replay_budget_usd))  # fmt: skip
    render(bundle, out)
    if json_out:
        json_out.write_text(bundle.model_dump_json(indent=1))
    rel = bundle.reliability
    assert rel is not None
    t = Table(title=f"{rel.agent_version}: {rel.n_tasks} tasks x {rel.runs_per_task} runs")
    for col in ("metric", "value", "95% CI"):
        t.add_column(col)
    for key, e in rel.metrics.items():
        t.add_row(key, f"{e.value:.3f}", f"[{e.lo:.3f}, {e.hi:.3f}]")
    console.print(t)
    for d in bundle.divergences:
        console.print(f"{d.task_id}: diverges at {d.predicted_labels[:1]} ({d.confidence})")
    console.print(f"report: {out}")


@app.command("compare")
def compare_cmd(
    baseline: Annotated[str, typer.Option(help="baseline version (with --harness) or a runs .jsonl")],
    candidate: Annotated[str, typer.Option(help="candidate version (with --harness) or a runs .jsonl")],
    harness: Annotated[str | None, typer.Option(help="module:factory; runs both versions")] = None,
    k: int = 8,
    tasks: str | None = None,
    concurrency: int = 4,
    budget_usd: float | None = None,
    alpha: float = 0.05,
    min_effect: float = 0.0,
    report: Path = Path("agentdiff-compare.html"),
    save_runs: Annotated[Path | None, typer.Option(help="directory to save both run files")] = None,
    json_out: Annotated[Path | None, typer.Option("--json")] = None,
) -> None:
    """Paired statistical regression gate. Exit 1 only on a significant regression."""
    if harness:
        hb, hc = load_harness(harness, baseline), load_harness(harness, candidate)
        ids = _tasks(hb, tasks)
        base = _run(hb, ids, k, concurrency, budget_usd, 0)
        cand = _run(hc, ids, k, concurrency, budget_usd, 10_000)
        if save_runs:
            write_jsonl(base, save_runs / f"{baseline}.jsonl")
            write_jsonl(cand, save_runs / f"{candidate}.jsonl")
    else:
        base, cand = read_jsonl(baseline), read_jsonl(candidate)
    gate, bundle = compare_bundle(base, cand, alpha=alpha, min_effect=min_effect)
    render(bundle, report)
    if json_out:
        json_out.write_text(gate.model_dump_json(indent=1))
    t = Table(title=f"{gate.baseline} -> {gate.candidate}")
    for col in ("task", "baseline", "candidate", "p(worse)", "Holm", "verdict"):
        t.add_column(col)
    for row in sorted(gate.tasks, key=lambda r: r.delta):
        t.add_row(row.task_id, f"{row.base_pass}/{row.base_n}", f"{row.cand_pass}/{row.cand_n}",
                  f"{row.p_worse:.3g}", f"{row.p_worse_holm:.3g}", row.verdict)  # fmt: skip
    console.print(t)
    console.print(f"overall delta {gate.overall_delta:+.3f} CI [{gate.overall_ci[0]:+.3f}, {gate.overall_ci[1]:+.3f}] "
                  f"p(worse)={gate.overall_p_worse:.3g}")  # fmt: skip
    console.print(("[red]REGRESSION[/red]: " if gate.regression else "[green]OK[/green]: ") + gate.reason)
    console.print(f"report: {report}")
    raise typer.Exit(gate.exit_code)


@app.command("import")
def import_cmd(
    fmt: Annotated[str, typer.Argument(help="otel | langgraph | codeverify")],
    path: Path,
    out: Path,
    version: str = "unknown",
    results: Annotated[Path | None, typer.Option(help="codeverify: per-task results JSONL for grading")] = None,
) -> None:
    """Convert external traces into agentdiff trajectories."""
    if fmt == "otel":
        from agentdiff.adapters.otel import from_otel_jsonl

        trajs = from_otel_jsonl(path, agent_version=version)
    elif fmt == "langgraph":
        from agentdiff.adapters.langgraph import from_langgraph_jsonl

        trajs = from_langgraph_jsonl(path, agent_version=version)
    elif fmt == "codeverify":
        from agentdiff.adapters.codeverify import from_codeverify_events

        trajs = from_codeverify_events(path, results, agent_version=version)
    else:
        raise typer.BadParameter(f"unknown format {fmt}")
    write_jsonl(trajs, out)
    console.print(f"imported {len(trajs)} trajectories -> {out}")


@app.command()
def demo(out: Path = Path("agentdiff-demo"), k: int = 8) -> None:
    """Offline end-to-end demo on a toy agent with planted failure causes (no API key)."""
    from agentdiff.demo.toy import ToyAgent

    good = ToyAgent(version="toy-v1", p_truncate=0.5)
    bad = ToyAgent(version="toy-v2", p_truncate=0.5, p_wrong_file={t: min(0.9, p + 0.45) for t, p in good.p_wrong_file.items()})
    base = run_suite(good, good.task_ids(), k)
    cand = run_suite(bad, good.task_ids(), k, seed0=10_000)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(base, out / "toy-v1.jsonl")
    write_jsonl(cand, out / "toy-v2.jsonl")
    bundle = analyze(base, title="agentdiff demo · toy-v1 (offline toy agent)")
    bundle.replays = replay_all(good, base, bundle.divergences, n_forks=10)
    render(bundle, out / "report.html")
    gate, cbundle = compare_bundle(base, cand)
    render(cbundle, out / "compare.html")
    (out / "gate.json").write_text(json.dumps(gate.model_dump(mode="json"), indent=1))
    console.print(f"reliability report: {out / 'report.html'}\ncompare report: {out / 'compare.html'}\n"
                  f"gate: {'REGRESSION' if gate.regression else 'OK'} - {gate.reason}")  # fmt: skip


if __name__ == "__main__":
    app()
