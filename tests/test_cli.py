from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from agentdiff.cli import app
from agentdiff.demo.toy import ToyAgent
from agentdiff.runner import run_suite
from agentdiff.schema import read_jsonl, write_jsonl

runner = CliRunner()


def test_demo_writes_reports(tmp_path: Path) -> None:
    res = runner.invoke(app, ["demo", "--out", str(tmp_path), "--k", "6"])
    assert res.exit_code == 0, res.output
    html = (tmp_path / "report.html").read_text()
    assert "Replay confirmation" in html and "<script src" not in html
    assert len(read_jsonl(tmp_path / "toy-v1.jsonl")) == 36


def test_compare_exit_codes(tmp_path: Path) -> None:
    good = ToyAgent(version="a")
    write_jsonl(run_suite(good, good.task_ids(), k=8), tmp_path / "a.jsonl")
    write_jsonl(run_suite(ToyAgent(version="b"), good.task_ids(), k=8, seed0=77), tmp_path / "b.jsonl")
    bad = ToyAgent(version="c", p_wrong_file={t: 0.8 for t in good.p_wrong_file})
    write_jsonl(run_suite(bad, good.task_ids(), k=8, seed0=99), tmp_path / "c.jsonl")
    same = runner.invoke(app, ["compare", "--baseline", str(tmp_path / "a.jsonl"), "--candidate",
                               str(tmp_path / "b.jsonl"), "--report", str(tmp_path / "r1.html")])
    worse = runner.invoke(app, ["compare", "--baseline", str(tmp_path / "a.jsonl"), "--candidate",
                                str(tmp_path / "c.jsonl"), "--report", str(tmp_path / "r2.html")])
    assert same.exit_code == 0, same.output
    assert worse.exit_code == 1, worse.output


def test_harness_cli_run(tmp_path: Path, monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.chdir(Path(__file__).parents[1])
    out = tmp_path / "runs.jsonl"
    res = runner.invoke(app, ["run", "--harness", "examples.demo_pr.agent:make_harness", "--version", "v",
                              "--k", "2", "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert len(read_jsonl(out)) == 12
