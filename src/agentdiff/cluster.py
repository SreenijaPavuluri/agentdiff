"""Cluster divergence points across tasks into recurring failure modes.

Membership is decided deterministically: every divergence becomes a feature vector of
one-hot categoricals (step label without occurrence, node, kind, deviation type) and TF-IDF
weighted diff tokens (words that appear only on the failing or only on the passing side of the
step diff, prefixed with ``+``/``-``). Categoricals and tokens are L2-normalised separately and
concatenated with weights 0.6 / 0.4. Average-linkage agglomerative clustering on cosine
distance, cut at ``threshold``, gives the clusters.

An optional ``namer`` (e.g. an LLM) receives each finished cluster and returns a short name.
It is never consulted about which divergences belong together.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable

import numpy as np
from pydantic import BaseModel
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from agentdiff.divergence import DivergenceResult

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


class FailureMode(BaseModel):
    cluster_id: int
    name: str
    named_by: str  # "template" or the namer's identifier
    members: list[str]  # task ids
    step_labels: list[str]
    deviation_types: list[str]
    top_tokens: list[str]


def _signature(d: DivergenceResult) -> tuple[list[str], list[str]]:
    assert d.chosen is not None
    c = d.columns[d.chosen]
    label = (c.fail_label or c.ref_label or "?").split("#")[0]
    node, kind, name = ([*label.split(":"), "", ""])[:3]
    cats = [f"label={label}", f"node={node}", f"kind={kind}", f"name={name}", f"dev={c.deviation}"]
    toks: list[str] = []
    if d.evidence:
        for diff in (d.evidence.decision_diff, d.evidence.observation_diff):
            for line in diff.splitlines():
                if line.startswith(("+++", "---", "@@")):
                    continue
                if line[:1] in "+-":
                    toks += [f"{line[0]}{w.lower()}" for w in _WORD.findall(line[1:])]
    return cats, toks


def _matrix(rows: list[list[str]], idf: bool) -> tuple[np.ndarray, list[str]]:
    vocab = sorted({t for r in rows for t in r})
    index = {t: i for i, t in enumerate(vocab)}
    m = np.zeros((len(rows), len(vocab)))
    for i, r in enumerate(rows):
        for t, n in Counter(r).items():
            m[i, index[t]] = n
    if idf and len(vocab):
        df = (m > 0).sum(axis=0)
        m = m * (np.log((1 + len(rows)) / (1 + df)) + 1)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return m / norms, vocab


def cluster_divergences(
    results: list[DivergenceResult],
    threshold: float = 0.6,
    namer: Callable[[list[DivergenceResult]], str] | None = None,
    namer_id: str = "llm",
) -> list[FailureMode]:
    results = [r for r in results if r.chosen is not None]
    if not results:
        return []
    sigs = [_signature(r) for r in results]
    cat_m, _ = _matrix([s[0] for s in sigs], idf=False)
    tok_m, tok_vocab = _matrix([s[1] for s in sigs], idf=True)
    x = np.hstack([0.6 * cat_m, 0.4 * tok_m]) if tok_m.shape[1] else cat_m
    if len(results) == 1:
        labels = np.array([1])
    else:
        dist = pdist(x, metric="cosine")
        dist = np.nan_to_num(dist, nan=1.0)
        labels = fcluster(linkage(dist, method="average"), t=threshold, criterion="distance")
    modes = []
    for cid in sorted(set(labels.tolist()), key=lambda c: (-int((labels == c).sum()), c)):
        idx = [i for i, lab in enumerate(labels) if lab == cid]
        members = [results[i] for i in idx]
        step_labels = Counter((m.columns[m.chosen].fail_label or "?").split("#")[0] for m in members if m.chosen is not None)
        devs = Counter(m.columns[m.chosen].deviation for m in members if m.chosen is not None)
        tok_weight = tok_m[idx].sum(axis=0) if tok_m.shape[1] else np.array([])
        top = [tok_vocab[i] for i in np.argsort(-tok_weight)[:6] if tok_weight[i] > 0] if len(tok_weight) else []
        lab, dev = step_labels.most_common(1)[0][0], devs.most_common(1)[0][0]
        name, by = f"{dev} divergence at {lab}", "template"
        if namer is not None:
            try:
                name, by = namer(members).strip()[:80] or name, namer_id
            except Exception:  # noqa: BLE001, SIM105 - naming is cosmetic; keep the template name
                name, by = name, "template"
        modes.append(FailureMode(cluster_id=len(modes) + 1, name=name, named_by=by, members=[m.task_id for m in members],
                                 step_labels=[s for s, _ in step_labels.most_common()],
                                 deviation_types=[d for d, _ in devs.most_common()], top_tokens=top))  # fmt: skip
    return modes
