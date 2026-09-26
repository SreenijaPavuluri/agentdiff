"""Score agentdiff on the injected faults. Usage: python evaluate.py [--no-judge] [--no-replay]

Reads results/runs/*.jsonl produced by run_all.py and writes
  results/eval/detection.json      gate verdict + measured effect per fault, false alarms
  results/eval/localization.json   per (fault, task) pair: truth and each method's top-3
  results/eval/replay.json         replay confirmation on a capped subset of pairs
  results/eval/timing.json         wall-clock per analysis
  results/tables.md                the tables quoted in the README (generated, never hand-edited)

Ground truth for localization is mechanical: the label of the first step at which the fault
hook fired (step.meta["fault_fired"]) in each failing candidate run; the pair's truth is the
most common such label across its failing runs.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from common import RESULTS, make_harness, model
from faults import BY_ID, FAULTS

from agentdiff.divergence import find_divergence
from agentdiff.gate import compare
from agentdiff.replay import confirm
from agentdiff.schema import Trajectory, group_by_task, read_jsonl, step_labels
from agentdiff.stats.intervals import clopper_pearson
from agentdiff.stats.tests import paired_bootstrap_diff

RUNS, OUT = RESULTS / "runs", RESULTS / "eval"
JUDGE_MODEL = "qwen2.5-coder:1.5b"
REPLAY_PAIRS_PER_FAULT = 2
REPLAY_FORKS = 6


def load(name: str) -> list[Trajectory]:
    p = RUNS / f"{name}.jsonl"
    return [t for t in read_jsonl(p) if t.outcome.infra_error is None] if p.exists() else []


def seeds(runs: list[Trajectory], lo: int, hi: int) -> list[Trajectory]:
    return [r for r in runs if lo <= (r.seed or 0) < hi]


# ------------------------------------------------------------------------------------------ truth


def first_fired(run: Trajectory) -> str | None:
    labels = step_labels(run.steps)
    return next((lab for s, lab in zip(run.steps, labels, strict=True) if s.meta.get("fault_fired")), None)


def truth_of(failing: list[Trajectory]) -> tuple[str | None, int]:
    fired = [lab for r in failing if (lab := first_fired(r))]
    return (Counter(fired).most_common(1)[0][0], len(fired)) if fired else (None, 0)


# ---------------------------------------------------------------------------------------- methods


def is_failing_obs(s) -> bool:  # noqa: ANN001
    if s.error:
        return True
    o = s.observation if isinstance(s.observation, dict) else {}
    res = o.get("result") if isinstance(o.get("result"), dict) else {}
    if o.get("type") == "TestReport":
        return not (res.get("total", 0) > 0 and res.get("passed") == res.get("total") and not res.get("collection_error")
                    and not res.get("timed_out"))  # fmt: skip
    return False


def last_failing_step(run: Trajectory) -> list[str]:
    """Naive rule: the last step with an error or a failing test observation, then earlier ones."""
    labels = step_labels(run.steps)
    hits = [lab for s, lab in zip(run.steps, labels, strict=True) if is_failing_obs(s)]
    if not hits:
        hits = [lab for s, lab in zip(run.steps, labels, strict=True) if s.kind == "llm"]
    return list(dict.fromkeys(reversed(hits)))[:3]


def _brief(v, n: int = 260) -> str:  # noqa: ANN001
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    s = re.sub(r"\s+", " ", s)
    return s if len(s) <= n else s[:n] + "..."


def format_run(run: Trajectory) -> str:
    lines = []
    for s, lab in zip(run.steps, step_labels(run.steps), strict=True):
        dec = s.view if s.view is not None else s.decision
        if s.kind == "tool":
            dec = {k: v for k, v in (dec or {}).items() if k != "test_code"}
        lines.append(f"[{lab}] decision: {_brief(dec)}" + (f" | observation: {_brief(s.observation)}" if s.observation is not None else "")
                     + (f" | error: {s.error}" if s.error else ""))  # fmt: skip
    return "\n".join(lines)


JUDGE_SYSTEM = (
    "You are debugging a failed run of a multi-agent bug-fixing system (planner, inspector, patch_generator, verifier). "
    "Each line is one step, labelled [node:kind:name#occurrence]. Identify the step that CAUSED the failure: the "
    "earliest decisive mistake or bad tool result, not the step where the failure became visible. "
    'Answer with JSON only: {"steps": ["<label>", "<label>", "<label>"]} ranked most likely first, using labels exactly as shown.'
)


def llm_judge(failing: Trajectory, passing: Trajectory | None) -> tuple[list[str], float]:
    user = f"FAILED RUN:\n{format_run(failing)}"
    if passing is not None:
        user = f"A PASSING RUN OF THE SAME TASK (for comparison):\n{format_run(passing)}\n\n" + user
    t0 = time.perf_counter()
    try:
        reply = model(JUDGE_MODEL).complete(JUDGE_SYSTEM, user[:24000], temperature=0.0, seed=0)
        text = reply.text
    except Exception as exc:  # noqa: BLE001
        return [f"<judge error {type(exc).__name__}>"], time.perf_counter() - t0
    valid = set(step_labels(failing.steps))
    found = re.findall(r"[a-z_]+:(?:llm|tool):[a-z_]+#\d+", text)
    ranked = list(dict.fromkeys(lab for lab in found if lab in valid))
    return ranked[:3], time.perf_counter() - t0


# ------------------------------------------------------------------------------------------- main


def hit(pred: list[str], truth: str, k: int) -> bool:
    return truth in pred[:k]


def comp(label: str) -> str:
    """Component = node:kind:name without the occurrence number."""
    return label.split("#")[0]


def hit_comp(pred: list[str], truth: str, k: int) -> bool:
    return comp(truth) in [comp(p) for p in list(dict.fromkeys(comp(x) for x in pred))[:k]]


def main(judge: bool = True, replay: bool = True) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base_all = load("baseline")
    base8 = seeds(base_all, 0, 8)
    detection, pairs, replays, timing = [], [], [], {"divergence_s": [], "judge_s": [], "judge_pair_s": [], "replay": []}

    # ---------------- false alarms
    null_rows = []
    for nid in ["N01"]:
        runs = load(nid)
        if runs:
            g = compare(base8, runs)
            null_rows.append({"id": nid, "description": BY_ID[nid].description, "regression": g.regression,
                              "delta": g.overall_delta, "p_worse": g.overall_p_worse, "reason": g.reason})  # fmt: skip
    aa_true = compare(seeds(base_all, 0, 8), [r.model_copy(update={"agent_version": "baseline-b"}) for r in seeds(base_all, 8, 16)])
    null_rows.append({"id": "A/A", "description": "baseline seeds 0-7 vs seeds 8-15 (same version)",
                      "regression": aa_true.regression, "delta": aa_true.overall_delta, "p_worse": aa_true.overall_p_worse,
                      "reason": aa_true.reason})  # fmt: skip
    rng = random.Random(0)
    by_task = group_by_task(base_all)
    n_split, n_alarm = 500, 0
    for _ in range(n_split):
        a, b = [], []
        for runs in by_task.values():
            idx = list(range(len(runs)))
            rng.shuffle(idx)
            h = len(idx) // 2
            a += [runs[i] for i in idx[:h]]
            b += [runs[i].model_copy(update={"agent_version": "split-b"}) for i in idx[h:]]
        n_alarm += compare(a, b).regression
    aa_splits = {"splits": n_split, "false_alarms": n_alarm, "rate": n_alarm / n_split,
                 "ci": clopper_pearson(n_alarm, n_split)}  # fmt: skip

    # ---------------- per fault
    for f in FAULTS:
        cand = load(f.id)
        if not cand:
            continue
        g = compare(base8, cand)
        xs = group_by_task(cand)
        xb = group_by_task(base_all)
        tids = sorted(set(xs) & set(xb))
        d16, lo16, hi16 = paired_bootstrap_diff([sum(r.outcome.passed for r in xs[t]) for t in tids], [len(xs[t]) for t in tids],
                                                [sum(r.outcome.passed for r in xb[t]) for t in tids], [len(xb[t]) for t in tids])  # fmt: skip
        fired_runs = sum(1 for r in cand if any(s.meta.get("fault_fired") for s in r.steps))
        detection.append({
            "id": f.id, "description": f.description, "intermittent": f.intermittent, "expected_cause": f.true_cause,
            "n_runs": len(cand), "runs_where_fault_fired": fired_runs,
            "base_pass1": float(np.mean([r.outcome.passed for r in base8])), "cand_pass1": float(np.mean([r.outcome.passed for r in cand])),
            "effect_vs_16": d16, "effect_ci_vs_16": [lo16, hi16], "effect_ci_excludes_0": hi16 < 0,
            "gate_regression": g.regression, "gate_p_worse": g.overall_p_worse, "gate_delta": g.overall_delta,
            "gate_regressed_tasks": g.regressed_tasks, "gate_reason": g.reason,
        })  # fmt: skip

        effective = hi16 < 0
        # localization pairs (regression mode): baseline passes (all 16) vs candidate failures.
        # Ground truth is only meaningful when the fault actually causes failures, so the judge and
        # replay (the expensive methods) run only for faults whose measured effect CI excludes 0.
        n_replayed = 0
        for tid in tids:
            passing = [r for r in xb[tid] if r.outcome.passed]
            failing = [r for r in xs[tid] if not r.outcome.passed]
            truth, n_fired = truth_of(failing)
            if len(passing) < 2 or not failing or truth is None:
                continue
            t0 = time.perf_counter()
            div = find_divergence(passing, failing, mode="regression")
            timing["divergence_s"].append(time.perf_counter() - t0)
            rep_fail = next(r for r in failing if first_fired(r))
            row = {"fault": f.id, "task": tid, "truth": truth, "n_pass": len(passing), "n_fail": len(failing),
                   "n_fail_fired": n_fired, "intermittent": f.intermittent, "effective": effective,
                   "n_labels": len(set(step_labels(rep_fail.steps))),
                   "agentdiff": div.predicted_labels, "agentdiff_confidence": div.confidence,
                   "last_failing_step": last_failing_step(rep_fail)}  # fmt: skip
            if judge and effective:
                row["judge"], dt = llm_judge(rep_fail, None)
                timing["judge_s"].append(dt)
                row["judge_with_pass"], dt = llm_judge(rep_fail, div and next(r for r in passing if r.run_id == div.reference_run))
                timing["judge_pair_s"].append(dt)
            if replay and effective and n_replayed < REPLAY_PAIRS_PER_FAULT:
                n_replayed += 1
                h = make_harness(f.id, fault=f, tasks=[tid])
                t0 = time.perf_counter()
                rep = confirm(h, div, failing, passing, n_forks=REPLAY_FORKS, top_k=2, intervention="transplant")
                row["agentdiff_replay"] = rep.predicted_labels
                row["replay_confirmed"] = rep.confirmed_label
                replays.append({"fault": f.id, "task": tid, "truth": truth, **json.loads(rep.model_dump_json())})
                timing["replay"].append({"fault": f.id, "task": tid, "wall_s": time.perf_counter() - t0, "forks": rep.total_forks})
            pairs.append(row)
            print(f"{f.id} {tid} truth={truth} agentdiff={div.predicted_labels[:1]} last={row['last_failing_step'][:1]}"
                  f" judge={row.get('judge', [])[:1]} replay={row.get('replay_confirmed')}", flush=True)
            (OUT / "localization.json").write_text(json.dumps(pairs, indent=1))

        # flaky mode for intermittent faults: passing and failing runs both from the faulty version
        if f.intermittent:
            for tid in tids:
                passing = [r for r in xs[tid] if r.outcome.passed]
                failing = [r for r in xs[tid] if not r.outcome.passed]
                truth, _ = truth_of(failing)
                if len(passing) >= 2 and failing and truth:
                    div = find_divergence(passing, failing, mode="flaky")
                    pairs.append({"fault": f.id, "task": tid, "truth": truth, "mode": "flaky", "n_pass": len(passing),
                                  "effective": effective,
                                  "n_fail": len(failing), "agentdiff": div.predicted_labels, "intermittent": True,
                                  "agentdiff_confidence": div.confidence})  # fmt: skip

    (OUT / "detection.json").write_text(json.dumps({"faults": detection, "nulls": null_rows, "aa_splits": aa_splits}, indent=1))
    (OUT / "localization.json").write_text(json.dumps(pairs, indent=1))
    (OUT / "replay.json").write_text(json.dumps(replays, indent=1))
    (OUT / "timing.json").write_text(json.dumps(timing, indent=1))
    write_tables(detection, null_rows, aa_splits, pairs, replays, timing)


def _acc(rows: list[dict], key: str, k: int, component: bool = False) -> tuple[int, int]:
    rows = [r for r in rows if key in r]
    f = hit_comp if component else hit
    return sum(f(r[key], r["truth"], k) for r in rows), len(rows)


def _fmt(h: int, n: int) -> str:
    if n == 0:
        return "-"
    lo, hi = clopper_pearson(h, n)
    return f"{h}/{n} = {h / n:.0%} [{lo:.0%}, {hi:.0%}]"


def write_tables(detection, nulls, aa, pairs, replays, timing) -> None:  # noqa: ANN001
    L = ["# Results (generated by experiments/evaluate.py, do not edit)\n"]
    eff = [d for d in detection if d["effect_ci_excludes_0"]]
    L.append("## Detection: does `agentdiff compare` flag the fault? (k=8 vs 8, 10 tasks, alpha=0.05)\n")
    L.append("| Fault | Description | pass@1 base -> fault | Effect vs 16-run baseline [95% CI] | Gate p(worse) | Flagged |")
    L.append("|---|---|---|---|---|---|")
    for d in detection:
        L.append(f"| {d['id']} | {d['description']} | {d['base_pass1']:.2f} -> {d['cand_pass1']:.2f} | "
                 f"{d['effect_vs_16']:+.2f} [{d['effect_ci_vs_16'][0]:+.2f}, {d['effect_ci_vs_16'][1]:+.2f}] | "
                 f"{d['gate_p_worse']:.2g} | {'yes' if d['gate_regression'] else 'no'} |")  # fmt: skip
    n_flag = sum(d["gate_regression"] for d in detection)
    n_eff_flag = sum(d["gate_regression"] for d in eff)
    L.append(f"\n- Detection rate, all faults: {_fmt(n_flag, len(detection))}")
    L.append(f"- Detection rate, faults whose measured effect CI excludes 0: {_fmt(n_eff_flag, len(eff))}")
    L.append("\n## False alarms on unchanged or harmless versions\n")
    L.append("| Check | Flagged | Delta | p(worse) |\n|---|---|---|---|")
    for n in nulls:
        L.append(f"| {n['id']}: {n['description']} | {'yes' if n['regression'] else 'no'} | {n['delta']:+.3f} | {n['p_worse']:.2g} |")
    L.append(f"\nA/A resampling: {aa['false_alarms']}/{aa['splits']} random 8-vs-8 splits of the 16-run baseline flagged "
             f"= {aa['rate']:.1%} (95% CI {aa['ci'][0]:.1%}-{aa['ci'][1]:.1%}); splits overlap, so this is a rate estimate, not {aa['splits']} independent tests.")
    reg_all = [p for p in pairs if p.get("mode") != "flaky"]
    reg = [p for p in reg_all if p.get("effective")]
    flk = [p for p in pairs if p.get("mode") == "flaky" and p.get("effective")]
    eff_ids = sorted({p["fault"] for p in reg})
    L.append(f"\n## Localization: is the true cause step in the top-1 / top-3? ({len(reg)} regression-mode pairs "
             f"from the {len(eff_ids)} faults with a measured effect: {', '.join(eff_ids)})\n")
    L.append("| Method | Step top-1 | Step top-3 | Component top-1 | Component top-3 |\n|---|---|---|---|---|")
    rnd1 = np.mean([1 / p["n_labels"] for p in reg]) if reg else 0
    rnd3 = np.mean([min(1, 3 / p["n_labels"]) for p in reg]) if reg else 0
    for name, key in [("agentdiff divergence", "agentdiff"), ("LLM judge (failing run only)", "judge"),
                      ("LLM judge (+ one passing run)", "judge_with_pass"), ("Last failing step (naive rule)", "last_failing_step")]:
        L.append(f"| {name} | {_fmt(*_acc(reg, key, 1))} | {_fmt(*_acc(reg, key, 3))} | "
                 f"{_fmt(*_acc(reg, key, 1, True))} | {_fmt(*_acc(reg, key, 3, True))} |")
    L.append(f"| Random step (expected) | {rnd1:.0%} | {rnd3:.0%} | - | - |")
    L.append("\nStep = exact label incl. occurrence (e.g. `patch_generator:llm:llm#1`); component = node:kind:name "
             "(occurrence ignored; the agent's structured-output repair loop adds extra calls that shift occurrence numbers).")
    sub = [p for p in reg if "agentdiff_replay" in p]
    if sub:
        L.append(f"\nReplay subset ({len(sub)} pairs, up to {REPLAY_PAIRS_PER_FAULT} per fault):\n")
        L.append("| Method (same pairs) | Top-1 | Top-3 |\n|---|---|---|")
        for name, key in [("agentdiff divergence", "agentdiff"), ("agentdiff + replay", "agentdiff_replay"),
                          ("LLM judge (+ one passing run)", "judge_with_pass"), ("Last failing step", "last_failing_step")]:
            L.append(f"| {name} | {_fmt(*_acc(sub, key, 1))} | {_fmt(*_acc(sub, key, 3))} |")
        conf = sum(1 for p in sub if p.get("replay_confirmed"))
        conf_ok = sum(1 for p in sub if p.get("replay_confirmed") == p["truth"])
        L.append(f"\nReplay confirmed a step on {conf}/{len(sub)} pairs; the confirmed step was the true cause on {conf_ok}/{conf}.")
    L.append("\n### By fault type (agentdiff divergence, top-1)\n")
    L.append("| Subset | Top-1 | Top-3 |\n|---|---|---|")
    for name, rows in [("decision faults (prompt/model/temperature)", [p for p in reg if not p["intermittent"]]),
                       ("observation faults (tool results)", [p for p in reg if p["intermittent"]]),
                       ("flaky mode (intermittent faults, within the faulty version)", flk)]:
        L.append(f"| {name} | {_fmt(*_acc(rows, 'agentdiff', 1))} | {_fmt(*_acc(rows, 'agentdiff', 3))} |")
    L.append("\n### Per fault (top-1 step hits / pairs; faults without a measured effect marked *)\n")
    L.append("| Fault | Pairs | agentdiff | judge (+pass) | last step |\n|---|---|---|---|---|")
    for f in FAULTS:
        rows = [p for p in reg_all if p["fault"] == f.id]
        if rows:
            star = "" if rows[0].get("effective") else "*"
            L.append(f"| {f.id}{star} | {len(rows)} | {_acc(rows, 'agentdiff', 1)[0]} | {_acc(rows, 'judge_with_pass', 1)[0]} | "
                     f"{_acc(rows, 'last_failing_step', 1)[0]} |")
    L.append("\n## Cost and time per analysis (local model, $0 API cost)\n")
    for key, name in [("divergence_s", "divergence finder per pair"), ("judge_s", "LLM judge per pair (failing only)"),
                      ("judge_pair_s", "LLM judge per pair (+passing)")]:
        v = timing.get(key) or []
        if v:
            L.append(f"- {name}: median {np.median(v):.2f}s, max {np.max(v):.2f}s (n={len(v)})")
    if timing.get("replay"):
        w = [r["wall_s"] for r in timing["replay"]]
        fk = [r["forks"] for r in timing["replay"]]
        L.append(f"- replay confirmation per pair: median {np.median(w) / 60:.1f} min, {np.median(fk):.0f} forks (n={len(w)})")
    (RESULTS / "tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(judge="--no-judge" not in sys.argv, replay="--no-replay" not in sys.argv)
    _ = Path
