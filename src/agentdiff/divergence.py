"""Divergence finder: the earliest step where failing runs consistently split from passing runs.

Algorithm (documented in docs/design.md section 5)
-------------------------------------------------
1. Reference R = medoid of the passing runs (highest mean alignment score to the others).
2. Align every run to R. Reference coordinates are a sequence of *columns*:
   ``ins@0, step@0, ins@1, step@1, ..., ins@L`` where ``step@j`` holds the run's step paired
   with R's j-th step (or nothing) and ``ins@j`` holds the run's extra steps inserted before
   R's j-th step (``ins@L`` = steps after R ended, e.g. extra retries).
3. Column dissimilarity between two runs: for ``step`` columns ``1 - content_sim`` (1 if one
   side is missing, 0 if both are); for ``ins`` columns ``1 - ratio`` of the SequenceMatcher
   ratio over the inserted step signatures.
4. For every run and column, ``d`` = distance to the *nearest passing run* (leave-one-out for
   passing runs). Nearest-neighbour distance tolerates several distinct successful strategies.
5. Per column, a one-sided exact permutation test of mean(d_fail) > mean(d_pass) gives p_j
   and effect_j = mean(d_fail) - mean(d_pass).
6. The divergence column is the earliest column with p_j < alpha and effect_j >= min_effect.
   Candidates for top-k: qualifying columns in time order, then the rest by effect.
   "Earliest significant" rather than "largest effect" because once runs diverge every later
   step keeps diverging (the final test step differs in every failing run).
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
from itertools import combinations
from math import comb
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from agentdiff.align.nw import align, alignment_score
from agentdiff.align.similarity import content_sim, decision_sim, observation_sim
from agentdiff.schema import Step, Trajectory, step_labels

ColumnKind = Literal["step", "ins"]


@dataclass
class _Cols:
    """One run projected onto the reference coordinates."""

    step: list[Step | None]  # len L
    ins: list[list[Step]]  # len L + 1
    labels: dict[int, str]  # step.index -> label


def project(run: Trajectory, ref: Trajectory) -> _Cols:
    L = len(ref.steps)
    step: list[Step | None] = [None] * L
    ins: list[list[Step]] = [[] for _ in range(L + 1)]
    pending: list[Step] = []
    for i_ref, i_run in align(ref.steps, run.steps):
        if i_ref is not None:
            ins[i_ref].extend(pending)
            pending = []
            if i_run is not None:
                step[i_ref] = run.steps[i_run]
        elif i_run is not None:
            pending.append(run.steps[i_run])
    ins[L].extend(pending)
    labels = {s.index: lab for s, lab in zip(run.steps, step_labels(run.steps), strict=True)}
    return _Cols(step, ins, labels)


def _col_dist(a: _Cols, b: _Cols, kind: ColumnKind, j: int) -> float:
    if kind == "step":
        x, y = a.step[j], b.step[j]
        if x is None and y is None:
            return 0.0
        if x is None or y is None:
            return 1.0
        return 1.0 - content_sim(x, y)
    xs = [s.signature for s in a.ins[j]]
    ys = [s.signature for s in b.ins[j]]
    if xs == ys:
        return 0.0
    return 1.0 - SequenceMatcher(None, xs, ys, autojunk=False).ratio()


def perm_test_greater(fail: np.ndarray, pas: np.ndarray, max_exact: int = 20000, mc: int = 10000,
                      seed: int = 0) -> float:  # fmt: skip
    """One-sided permutation p-value for mean(fail) > mean(pas) (exact when feasible)."""
    pooled = np.concatenate([fail, pas])
    nf, n = len(fail), len(pooled)
    obs = fail.mean() - pas.mean()
    tol = 1e-12
    total = pooled.sum()
    if comb(n, nf) <= max_exact:
        count = 0
        num = 0
        for idx in combinations(range(n), nf):
            f_sum = pooled[list(idx)].sum()
            stat = f_sum / nf - (total - f_sum) / (n - nf)
            count += stat >= obs - tol
            num += 1
        return count / num
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(mc):
        perm = rng.permutation(n)
        f_sum = pooled[perm[:nf]].sum()
        hits += (f_sum / nf - (total - f_sum) / (n - nf)) >= obs - tol
    return (1 + hits) / (mc + 1)


class Column(BaseModel):
    kind: ColumnKind
    j: int
    ref_label: str | None  # label of R's step (step columns) / step before the slot (ins)
    fail_label: str | None  # label of the representative failing run's step here
    effect: float
    p: float
    mean_d_fail: float
    mean_d_pass: float
    frac_fail_deviating: float  # share of failing runs with d >= 0.5
    frac_pass_deviating: float
    deviation: str  # decision | observation | missing | inserted | error | none


class Evidence(BaseModel):
    ref_step: dict[str, Any] | None
    fail_step: dict[str, Any] | None
    decision_diff: str
    observation_diff: str
    ref_run: str
    fail_run: str


class DivergenceResult(BaseModel):
    task_id: str
    mode: Literal["flaky", "regression"]
    n_pass: int
    n_fail: int
    reference_run: str
    columns: list[Column]
    ranked: list[int]  # indices into columns, best first
    chosen: int | None
    confidence: Literal["significant", "low"]
    predicted_labels: list[str]  # top-k labels (failing-run coordinates), best first
    evidence: Evidence | None = None


def medoid(runs: list[Trajectory]) -> Trajectory:
    if len(runs) == 1:
        return runs[0]
    scores = []
    for i, r in enumerate(runs):
        tot = sum(alignment_score(r.steps, q.steps) / max(len(r.steps), len(q.steps), 1)
                  for k, q in enumerate(runs) if k != i)  # fmt: skip
        scores.append(tot)
    best = max(range(len(runs)), key=lambda i: (scores[i], -i))
    return runs[best]


def _short(obj: Any, limit: int = 4000) -> str:
    import json

    s = obj if isinstance(obj, str) else json.dumps(obj, indent=1, sort_keys=True, default=str)
    return s if len(s) <= limit else s[:limit] + "\n...[truncated]"


def _diff(a: Any, b: Any) -> str:
    sa, sb = _short(a).splitlines(), _short(b).splitlines()
    return "\n".join(unified_diff(sa, sb, "passing(ref)", "failing", lineterm="", n=2))


def _step_brief(s: Step | None, label: str | None) -> dict[str, Any] | None:
    if s is None:
        return None
    return {"label": label, "kind": s.kind, "node": s.node, "name": s.name, "error": s.error,
            "decision": s.view if s.view is not None else s.decision, "observation": s.observation}  # fmt: skip


def find_divergence(
    passing: list[Trajectory],
    failing: list[Trajectory],
    *,
    mode: Literal["flaky", "regression"] = "flaky",
    alpha: float = 0.05,
    min_effect: float = 0.1,
    top_k: int = 3,
) -> DivergenceResult:
    if not passing or not failing:
        raise ValueError("need at least one passing and one failing run")
    task_id = failing[0].task_id
    ref = medoid(passing)
    L = len(ref.steps)
    pas_cols = [project(r, ref) for r in passing]
    fail_cols = [project(r, ref) for r in failing]
    ref_labels = step_labels(ref.steps)

    columns: list[Column] = []
    reps: list[int] = []
    order: list[tuple[ColumnKind, int]] = []
    for j in range(L + 1):
        order.append(("ins", j))
        if j < L:
            order.append(("step", j))

    for kind, j in order:
        d_fail = np.array([min(_col_dist(f, p, kind, j) for p in pas_cols) for f in fail_cols])
        if len(pas_cols) >= 2:
            d_pass = np.array([min(_col_dist(p, q, kind, j) for k2, q in enumerate(pas_cols) if k2 != k1)
                               for k1, p in enumerate(pas_cols)])  # fmt: skip
        else:
            d_pass = np.array([])
        p = perm_test_greater(d_fail, d_pass) if len(d_pass) >= 2 and d_fail.max() > 0 else 1.0
        effect = float(d_fail.mean() - (d_pass.mean() if len(d_pass) else 0.0))
        rep = int(np.argmax(d_fail))  # representative failing run: the most deviating one
        reps.append(rep)
        fc = fail_cols[rep]
        if kind == "step":
            ref_label = ref_labels[j]
            fs = fc.step[j]
            fail_label = fc.labels[fs.index] if fs is not None else ref_label
            deviation = _deviation_type(ref.steps[j], fs) if d_fail[rep] > 0 else "none"
        else:
            ref_label = ref_labels[j - 1] if j > 0 else None
            src = next((c for c in [fc, *fail_cols] if c.ins[j]), None)
            fail_label = src.labels[src.ins[j][0].index] if src is not None else None
            deviation = "inserted" if d_fail[rep] > 0 else "none"
        columns.append(Column(
            kind=kind, j=j, ref_label=ref_label, fail_label=fail_label, effect=round(effect, 6), p=float(p),
            mean_d_fail=float(d_fail.mean()), mean_d_pass=float(d_pass.mean()) if len(d_pass) else 0.0,
            frac_fail_deviating=float((d_fail >= 0.5).mean()),
            frac_pass_deviating=float((d_pass >= 0.5).mean()) if len(d_pass) else 0.0,
            deviation=deviation,
        ))  # fmt: skip

    qualifying = [i for i, c in enumerate(columns) if c.p < alpha and c.effect >= min_effect]
    rest = sorted((i for i in range(len(columns)) if i not in qualifying and columns[i].effect > 0),
                  key=lambda i: (-columns[i].effect, i))  # fmt: skip
    ranked = qualifying + rest
    chosen = ranked[0] if ranked else None
    confidence: Literal["significant", "low"] = "significant" if qualifying else "low"

    labels: list[str] = []
    for i in ranked:
        lab = columns[i].fail_label
        if lab and lab not in labels:
            labels.append(lab)
        if len(labels) >= top_k:
            break

    evidence = None
    if chosen is not None:
        c = columns[chosen]
        rep_run = failing[reps[chosen]]
        fc = fail_cols[reps[chosen]]
        if c.kind == "step":
            rs, fs = ref.steps[c.j], fc.step[c.j]
        else:
            rs = ref.steps[c.j] if c.j < L else None
            fs = fc.ins[c.j][0] if fc.ins[c.j] else None
        evidence = Evidence(
            ref_step=_step_brief(rs, c.ref_label), fail_step=_step_brief(fs, c.fail_label),
            decision_diff=_diff(_dec(rs), _dec(fs)), observation_diff=_diff(_obs(rs), _obs(fs)),
            ref_run=ref.run_id, fail_run=rep_run.run_id,
        )  # fmt: skip
    return DivergenceResult(
        task_id=task_id, mode=mode, n_pass=len(passing), n_fail=len(failing), reference_run=ref.run_id,
        columns=columns, ranked=ranked, chosen=chosen, confidence=confidence, predicted_labels=labels,
        evidence=evidence,
    )  # fmt: skip


def _dec(s: Step | None) -> Any:
    if s is None:
        return "(no step)"
    return s.view if s.view is not None else s.decision


def _obs(s: Step | None) -> Any:
    if s is None:
        return "(no step)"
    return {"error": s.error, "observation": s.observation} if s.error else s.observation


def _deviation_type(ref: Step, fs: Step | None) -> str:
    if fs is None:
        return "missing"
    if (ref.error is None) != (fs.error is None):
        return "error"
    ds, os_ = decision_sim(ref, fs), observation_sim(ref, fs)
    if ds >= 0.999 and os_ >= 0.999:
        return "none"
    return "decision" if ds <= os_ else "observation"
