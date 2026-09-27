# agentdiff

**Find out why your agent is flaky.** agentdiff does five things:

1. **Reruns** your agent's tasks many times.
2. **Measures** how reliable the agent is, with confidence intervals.
3. **Pinpoints** the step where passing and failing runs split.
4. **Tests** that step by replaying the run from it.
5. **Blocks** regressions in CI, but only when they are statistically real.

![Reliability decay of a real LangGraph agent](docs/img/reliability_decay.svg)

*Real data from the unmodified [code-verify-agents](https://github.com/SreenijaPavuluri/code-verify-agents) LangGraph agent:*

- *Setup: `qwen2.5-coder:1.5b`, 10 tasks × 16 runs, graded on hidden tests.*
- *The agent passes a single attempt 47% of the time (95% CI 31–64%).*
- *It passes 8 attempts in a row only 3% of the time (0–17%).*
- *Whiskers are two-stage bootstrap CIs. Produced by `agentdiff analyze`; full report in [`results/reports/baseline_report.html`](results/reports/baseline_report.html).*

## Quickstart

```bash
pip install "agentdiff @ git+https://github.com/SreenijaPavuluri/agentdiff.git"
agentdiff demo --out demo/          # offline toy agent, no API key, ~5 s
open demo/report.html demo/compare.html
```

Use it on your own agent in one of two ways.

**Harness mode (all features).** Route your model and tool calls through the `tape` agentdiff passes in:

```python
from agentdiff.schema import Outcome
from agentdiff.tape import LLMReply, Tape

class MyAgent:
    version = "v1"
    def task_ids(self): return ["task-1", "task-2"]
    def run(self, task_id: str, tape: Tape, seed: int) -> Outcome:
        reply = tape.llm("planner", {"prompt": "..."}, lambda: LLMReply(text=call_my_model(...)))
        result = tape.tool("coder", "read_file", {"path": "a.py"}, lambda: read("a.py"), replay_safe=True)
        return Outcome(passed=run_my_tests())          # a rule, not an LLM judge

def make_harness(version: str) -> MyAgent: ...
```

```bash
agentdiff run --harness mypkg.agent:make_harness --version v1 --k 8 --out v1.jsonl
agentdiff analyze v1.jsonl --out report.html --replay mypkg.agent:make_harness
agentdiff compare --baseline v1.jsonl --candidate v2.jsonl   # exit 1 only on a significant regression
```

**Import mode (statistics and divergence, no replay).** Convert existing traces:

```bash
agentdiff import otel spans.jsonl --out runs.jsonl        # OpenTelemetry GenAI spans
agentdiff import langgraph runs.jsonl --out runs.jsonl    # graph.stream(stream_mode="updates") records
agentdiff import codeverify events.jsonl --results results.jsonl --out runs.jsonl
```

**CI.** Call the reusable workflow. It runs the PR's base commit and head commit on the same tasks and uploads the HTML report:

```yaml
jobs:
  gate:
    uses: SreenijaPavuluri/agentdiff/.github/workflows/agentdiff-gate.yml@main
    with: { harness: "mypkg.agent:make_harness", k: 8 }
```

[`examples/demo_pr/`](examples/demo_pr) shows two things:

- A deliberately bad prompt change is caught: exit 1, p = 0.0009.
- A harmless rewording passes: exit 0. It passes even though one task swung from 7/8 to 3/8 by chance, a drop that a fixed pass-rate threshold would have failed on.

## How it works

```mermaid
flowchart LR
  A[Agent harness] -->|k runs per task| T[(Trajectories<br/>full payloads)]
  I[OTel / LangGraph / code-verify<br/>adapters] --> T
  T --> R[Reliability<br/>pass@k, pass^k + CIs]
  T --> D[Divergence finder<br/>align pass vs fail runs]
  D --> C[Failure-mode clusters]
  D --> P[Replay confirmation<br/>fork at step, measure Δ]
  T --> G[Regression gate<br/>paired exact tests]
  R & D & C & P & G --> H[Self-contained HTML report<br/>+ CI exit code]
```

| Piece | Method | Why this choice |
|---|---|---|
| Reliability | Unbiased pass@k = 1 − C(n−c,k)/C(n,k) (Chen et al. 2021) and pass^k = C(c,k)/C(n,k) (τ-bench). Per-task CIs are Clopper–Pearson mapped through p^k. Suite CIs are a two-stage bootstrap over tasks, then runs. | Exact coverage per task, and no degenerate [1, 1] interval when a task passes every run. Tests check unbiasedness exactly against the Binomial pmf. |
| Divergence | Needleman–Wunsch alignment of each run against the medoid passing run, over steps with the same node, kind and name. See the note below the table for how each step position is scored. | Once runs split, every later step differs too; picking the largest difference would just rediscover the final failing test step. |
| Step similarity | Canonical JSON with volatile tokens masked (ids, temp paths, durations). A harness can supply a normalized *view* of each decision (for code-verify-agents: the AST-canonical patch, the planned tool list, the cited suspect lines). | Free-text LLM rationales change every run and must not count as divergence. |
| Replay | Every model and tool call goes through a tape. A fork replays steps before *s* from cache (checking that requests match), intervenes at *s*, then runs the rest live. Two treatment arms: *resample* re-draws step *s* (flaky mode); *transplant* substitutes the baseline's passing step (regression mode). Each is compared with *keep*. A step is confirmed only if the Newcombe 95% CI of Δ excludes 0. | Resampling under a broken prompt can't fix a deterministic regression, so regressions need transplant. Tools not marked replay-safe are never re-executed. |
| Gate | Stratified exact test with a within-task label-permutation null (the conditional CMH test; with one run per task it reduces to exact McNemar), plus per-task one-sided Fisher tests with Holm correction, each at α/2. | Family-wise false-alarm rate ≤ α. Runs are paired by task, not by run. |
| Clustering | Average-linkage clustering on one-hot step features plus TF-IDF of diff tokens. An LLM may *name* a cluster but never assigns members. | Deterministic, so an LLM judge can't add noise. |

*How the divergence finder scores each step position:*

- For every run, take the distance to the nearest passing run's step at that position.
- Run an exact one-sided permutation test of failing vs passing distances.
- Report the **earliest** significant position.

The design, including what changed during implementation, is in [`docs/design.md`](docs/design.md). The landscape survey is in [`docs/landscape.md`](docs/landscape.md).

## Results

All numbers below come from scripts in this repo, with raw outputs in `results/`:

- runs: `experiments/run_all.py` and `run_batch2.py` → `results/runs/*.jsonl`, full trajectories;
- scores: `experiments/evaluate.py` → `results/eval/*.json` and [`results/tables.md`](results/tables.md).

**Setup**

- **Agent:** code-verify-agents (Planner → Inspector → Patch Generator → Verifier → Review), unmodified, driven through its `build_graph(llm=, tools=)` seams.
  - LLM planner and LLM inspector enabled.
  - 2 patch attempts.
  - Temperature 0.7.
- **Model:** `qwen2.5-coder:1.5b` on Ollama, on an 8 GB M1 laptop. **Total API cost: $0.**
- **Tasks:** 10 of the repo's 20 bug-fix tasks, chosen by a separate 4-run screen whose seeds are not reused.
- **Outcome:** the final code must pass the visible **and hidden** tests.
- **Ground truth:** 14 injected faults with mechanically recorded cause steps. The harness marks the exact step where a fault hook changed a request, model or tool result.

<!-- RESULTS:BEGIN -->
<!-- RESULTS:END -->

## Prior work and what is different

The full survey, with citations, is in [`docs/landscape.md`](docs/landscape.md). In short:

- **Where the ideas come from:**
  - **τ-bench** ([Yao et al. 2024](https://arxiv.org/abs/2406.12045)) defined pass^k. agentdiff uses it directly.
  - **Inspect** ([UK AISI](https://inspect.aisi.org.uk/)) already offers `pass_k` reducers with bootstrap and Wilson CIs. agentdiff's reliability layer overlaps with Inspect, and says so.
  - **Causal Agent Replay** ([Shah 2026](https://arxiv.org/abs/2606.08275)) introduced counterfactual interventions on agent steps with CIs, evaluated on synthetic causal models.
  - **LangGraph time travel** ([docs](https://docs.langchain.com/oss/python/langgraph/use-time-travel)) is the checkpoint-fork mechanism agentdiff's replay is analogous to.
- **Where agentdiff differs:**
  - agentdiff uses the agent's own *passing* runs to choose which 1–3 steps to test, instead of searching every step of a single failing trace.
  - It evaluates that approach on a real agent.
- **LLM-judge attribution:**
  - **Who&When** ([Zhang et al. 2025](https://arxiv.org/abs/2505.00212)) is the source of the all-at-once LLM-judge baseline used here.
  - **Root-Cause Attribution Is a Search Problem** ([Raj et al. 2026](https://arxiv.org/abs/2609.13463)) is a better LLM judge for long traces, and complementary.
- **Repeated-run analysis:** **FAMAS** ([2025](https://arxiv.org/abs/2509.13782)) uses repeated runs for spectrum-based fault localization. agentdiff aligns runs step by step instead of scoring action frequencies.
- **Eval platforms:** DeepEval, Ragas, Phoenix, LangSmith, MLflow and promptfoo offer tracing, reference-trajectory checks and threshold gates. None of them, per their docs, does statistical gating or pass-vs-fail alignment.
- **Trace format:** agentdiff reads **OpenTelemetry GenAI** spans ([semconv](https://github.com/open-telemetry/semantic-conventions-genai)).

No code was copied from any of these projects. All algorithms here were implemented from their published descriptions.

## Limitations

- **Small model, small study.** It was one agent, one 1.5B local model, 10 tasks and k ≤ 16, on a laptop, to keep the cost at $0. Per-task tests have little power at n = 8: Holm-corrected Fisher tests can only flag near-total collapses, such as 8/8 → 1/8.
- **Divergence needs both passing and failing runs of the same task.** It can't diagnose a task that always fails. With fewer than 2 passing runs, it reports low confidence.
- **Invisible causes get placed at their first visible effect.** Prompts and model names are deliberately excluded from step comparison, so a config change is localized to the first step whose *output* changed.
- **Replay needs a harness and replay-safe tools.** Agents with real side effects need sandboxing or mocks. Nondeterminism outside the tape (clocks, unseeded RNGs) is detected, not fixed.
- **The OTel import is structural-only unless content capture is on.** `gen_ai.tool.call.result` and the message attributes are opt-in in the spec.
- **The LLM-judge baseline used the same 1.5B model.** A frontier judge would likely do better; see the note next to the results.

## Roadmap

- Inspect log adapter and a τ-bench slice (not run: needs a paid model and a user simulator).
- Sequential testing, so the gate can stop early once significance is reached.
- Shapley-style credit across several steps (following Causal Agent Replay) when no single step is confirmed.

## Development

```bash
pip install -e ".[dev]"
ruff check src tests experiments examples && pytest -q
```

MIT licensed.
