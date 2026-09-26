# Landscape: what exists, and where agentdiff fits

*Researched 2026-09-25 from public docs, READMEs and arXiv abstract/HTML pages only. No code was cloned or copied.
"No evidence" means we did not find the feature in the pages we read, not that it provably does not exist.
Corrections are welcome. Please open an issue.*

agentdiff does four things. The table below uses them as its columns:

- **(R) Reliability.** Rerun each task k times and report pass@1, pass@k and pass^k with confidence intervals.
- **(D) Divergence.** Align the passing and failing trajectories of the same task, step by step, and find the earliest step where they consistently split.
- **(C) Causal confirmation.** Fork a run from a candidate step, with tool observations cached, resample the step, and measure the change in pass rate with a CI.
- **(G) Statistical CI gate.** Compare two agent versions on the same tasks with a paired test, and fail CI only on significant regressions.

| | (R) k-run reliability / CIs | (D) pass-vs-fail step alignment | (C) fork + replay with CI | (G) statistical CI gate |
|---|---|---|---|---|
| DeepEval | partial (`-r` repeat, `flaky=True`), no pass^k, no CIs | no | no | threshold-based |
| Ragas | no | single run vs reference (ToolCallAccuracy) | no | no documented gate |
| Arize Phoenix | partial (`repetitions`, average only) | open issue | single LLM span replay | threshold scripts |
| LangSmith | partial (`num_repetitions`, mean/std) | single run vs reference (agentevals) | single-call playground; LangGraph time-travel | assertion-based |
| MLflow GenAI | no | no | no | assertion-based |
| promptfoo | partial (`--repeat`) | single run assertions | no | pass-rate threshold |
| Inspect (UK AISI) | **yes** (`epochs`, `pass_k_{k}`, bootstrap/Wilson CIs) | no | no | no built-in gate |
| τ-bench (paper) | **defines pass^k**, no CIs | no | no | no |
| Who&When (paper) | no | no (one failing log per task) | no | no |
| Causal Agent Replay (paper) | no | no (starts from one failing trace) | **yes**, on synthetic SCMs | no |
| Root-Cause Attribution Is a Search Problem (paper) | no | no | no | no |

## Evaluation tools

**DeepEval** ([docs](https://deepeval.com/docs/evaluation-trajectory-based-llm-evals)).
DeepEval makes agent evaluation feel like unit testing.

- **What it does well:**
  - `@observe` tracing builds a trace tree.
  - Trace-level metrics: TaskCompletion, StepEfficiency, PlanAdherence.
  - Span-level metrics: ToolCorrectness.
  - `deepeval test run` plugs into pytest for CI ([CI docs](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd)).
  - A `-r` flag repeats test cases, and `flaky=True` downgrades a failure to a warning ([flags](https://deepeval.com/docs/evaluation-flags-and-configs)).
- **What agentdiff adds:**
  - Its gate is threshold-based: a metric below threshold fails the assertion.
  - Repetition is used to rerun tests, not to estimate reliability. We found no pass^k and no confidence intervals in the official docs.
  - Its metrics are mostly LLM judges that score one trace. agentdiff compares many runs of the same task against each other, prefers rule-based outcomes, and treats flakiness as the thing to measure rather than something to suppress.

**Ragas** ([agent metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/agents/)).
Ragas has the most mature RAG metric suite.

- **What it does well:**
  - Agent metrics: ToolCallAccuracy (strict or flexible order against a reference tool sequence), ToolCallF1 and AgentGoalAccuracy.
  - An experiment runner, and a CLI that can compare against a `--baseline` ([CLI](https://docs.ragas.io/en/latest/howtos/cli/)).
- **What agentdiff adds:**
  - ToolCallAccuracy is the closest analogue to our alignment. But it compares one run to a hand-written reference. agentdiff aligns *observed passing runs against observed failing runs*, so no reference trajectory is needed.
  - We found no repeated-run statistics, CIs or significance testing.

**Arize Phoenix** ([experiments](https://arize.com/docs/phoenix/datasets-and-experiments/overview-datasets)).
Phoenix is an open-source, OpenTelemetry/OpenInference-native tracing and experiment platform with a strong UI.

- **What it does well:**
  - `repetitions` in `run_experiment` is explicitly aimed at variance in agentic systems ([repetitions](https://arize.com/docs/phoenix/datasets-and-experiments/how-to-experiments/repetitions)). However, the UI reports the average.
  - Span Replay reloads one LLM span into the playground ([span replay](https://arize.com/docs/phoenix/prompt-engineering/overview-prompts/span-replay)).
- **Gaps:**
  - Multi-step trajectory evaluation is an open, high-priority issue ([#11654](https://github.com/Arize-ai/phoenix/issues/11654)).
  - CI gating is described as user-written threshold scripts.
- **What agentdiff adds:** Phoenix is an excellent place to *collect* traces. agentdiff consumes the same OpenTelemetry GenAI spans, then does the statistics, alignment and whole-run forking that Phoenix does not.

**LangSmith** ([repetitions](https://docs.langchain.com/langsmith/repetition), [comparison view](https://docs.langchain.com/langsmith/compare-experiment-results)).
LangSmith is the most complete commercial platform here.

- **What it does well:**
  - `num_repetitions` with mean and standard deviation.
  - An experiment comparison view that highlights regressed and improved examples against a baseline.
  - agentevals trajectory match against a reference ([trajectory evals](https://docs.langchain.com/langsmith/trajectory-evals)).
  - pytest integration.
  - LangGraph "time travel", which forks a graph from a prior checkpoint ([time travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)). That is the mechanical primitive our replay also needs.
- **What agentdiff adds:**
  - LangSmith's red/green comparison marks per-example changes without a significance test. With temperature > 0, many of those flips are noise.
  - Time travel is a manual debugging tool. agentdiff automates *which* step to fork, *how many* resamples to run, and *whether* the effect is statistically real.

**MLflow GenAI evaluation** ([eval & monitor](https://mlflow.org/docs/latest/genai/eval-monitor/)).
MLflow is strong at lifecycle integration.

- **What it does well:**
  - Trace-based `mlflow.genai.evaluate`.
  - Built-in judges, including ToolCallCorrectness and ToolCallEfficiency.
  - Version comparison.
  - A pytest plugin where a test passes when every scorer passes every row ([regression testing](https://mlflow.org/docs/latest/genai/eval-monitor/regression-testing/)).
- **What agentdiff adds:** We found no repeated-run reliability metrics, no CIs, no cross-run trajectory comparison and no statistical gate.

**promptfoo** ([CLI](https://www.promptfoo.dev/docs/usage/command-line/), [assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs/deterministic/)).
promptfoo excels at declarative test matrices across prompts and providers, and at red-teaming.

- **What it does well:**
  - Trajectory assertions over OpenTelemetry traces: `trajectory:tool-used`, `tool-args-match`, `tool-sequence`, `step-count`.
  - A GitHub Action that posts before/after comparisons on PRs ([action](https://www.promptfoo.dev/docs/integrations/github-action/)).
- **Gaps:**
  - `--repeat` exists, but a per-test pass-rate threshold over repeats is still an open request ([#5847](https://github.com/promptfoo/promptfoo/issues/5847)).
  - The CI exit code is driven by a global pass-rate threshold (`PROMPTFOO_PASS_RATE_THRESHOLD`).
- **What agentdiff adds:** agentdiff's gate replaces that fixed threshold with a paired test against the baseline, so a 95% → 93% wobble on 20 tasks does not fail a build.

**Inspect** (UK AI Security Institute) ([docs](https://inspect.aisi.org.uk/), [metrics](https://inspect.aisi.org.uk/metrics.html)).
Inspect is the strongest prior art for **(R)**, and agentdiff overlaps with it there. We say so plainly.

- **What it does well:**
  - Agents, tools and sandboxes (Docker, Kubernetes and others).
  - `epochs` with reducers `pass_at_{k}`, `pass_k_{k}` (pass^k) and `at_least_{k}`.
  - `stderr` (including clustered), `bootstrap_stderr`, `ci()` and `ci_wilson()`.
- **What agentdiff adds:**
  - Its statistics describe a single evaluation. We found no paired comparison between two evals and no built-in regression gate.
  - The log viewer shows transcripts one at a time. There is no cross-run alignment and no step-level forking.
- **Why agentdiff does not just build on Inspect:** agentdiff's reliability layer is intentionally small. It is mostly there so that (D), (C) and (G) share one data model. For Inspect users, an Inspect-log adapter is on the roadmap.

## Papers

**τ-bench** (Yao, Shinn, Razavi, Narasimhan, 2024; [arXiv:2406.12045](https://arxiv.org/abs/2406.12045)).
τ-bench introduced **pass^k**, the probability that *all* k independent trials of a task succeed. It is estimated per task as C(c,k)/C(n,k) from n trials with c successes, then averaged over tasks.

- **Headline finding:** reliability collapses with k. Strong function-calling agents that succeed on about half of retail tasks drop below 25% at pass^8.
- τ-bench grades by comparing final database state, which is a rule-based outcome.
- **What agentdiff adds:** agentdiff adopts pass^k directly and credits it. τ-bench reports point estimates and does not diagnose *why* a task is flaky. agentdiff adds bootstrap CIs, flaky-task ranking and localization.

**Who&When** (Zhang et al., ICML 2025; [arXiv:2505.00212](https://arxiv.org/abs/2505.00212), [code](https://github.com/ag2ai/Agents_Failure_Attribution)).
This is the standard benchmark for failure attribution. It contains failure logs from 127 multi-agent systems with human labels for the responsible agent and the decisive error step.

- It evaluates LLM-judge strategies: all-at-once, step-by-step and binary search.
- **Its main result:** step-level attribution is hard. The best method reaches about 14% step accuracy, per the abstract.
- **What agentdiff adds:**
  - Who&When has one failing log per task and no passing counterparts. Every method there reads a single failure and guesses.
  - agentdiff tests a different premise: when an agent is *flaky*, its own passing runs of the same task give a free counterfactual. Where they exist, the failure point can be located by comparison and confirmed by intervention instead of guessed.
  - We use Who&When's all-at-once LLM judge as one of our baselines.
  - Caveat: our evaluation uses a different setting (single-agent, injected faults), so numbers are not comparable across papers.

**Causal Agent Replay: Counterfactual Attribution for LLM-Agent Failures** (J. Shah, 2026; [arXiv:2606.08275](https://arxiv.org/abs/2606.08275)).
This is the **closest prior work to agentdiff's replay confirmation (C)**, and we credit it.

- **How it works:**
  - It frames an agent run as a structural causal model.
  - It defines interventions: resample or force an action, replace an observation, edit context, swap model.
  - It estimates step effects with Wilson/bootstrap CIs and attributes the failure to a "point of commitment".
  - It also gives a Monte-Carlo Shapley estimator and argues for record-replay discipline.
- **What is different in agentdiff:**
  1. **Where the candidates come from.** Causal Agent Replay starts from one failing trace and must search over its steps. agentdiff first uses *observed passing runs* (via alignment, D) to propose a few candidate steps, then spends the replay budget only on those. The replay budget is the expensive part, so that is the practical contribution.
  2. **Real-agent evaluation.** From what we could read, its evaluation uses synthetic causal models with planted ground truth. agentdiff evaluates on a real LangGraph agent with injected faults.
  3. **Pairing with regression gating.** agentdiff connects localization to a statistical regression gate.

  We could not confirm from the abstract/HTML whether its post-intervention tool calls are replayed or live. We do not claim a difference there.

**Root-Cause Attribution Is a Search Problem: Continual Search for Long-Horizon Agent Failures** (Raj et al., 2026; [arXiv:2609.13463](https://arxiv.org/abs/2609.13463)).
This paper improves LLM-judge attribution on very long traces by nudging the judge to keep searching for unresolved evidence across turns, instead of re-confirming its first answer.

- It evaluates on TRAIL, TELBench, AgentRx, Who&When and a new MegaRCA-Mix set.
- It reports roughly +15 absolute F1 points for a frontier model on long traces. The gains shrink on short traces.
- **What agentdiff adds:** it is still a judge reading one failing trace, with no execution, no passing-run comparison and no significance test. It is complementary to agentdiff: a better judge could replace agentdiff's LLM-judge baseline, or rank candidates when no passing runs exist.

**Other related work we checked** (credited, not built on):

- **Replay or counterfactuals used for other purposes:**
  - [AgenTracer](https://arxiv.org/abs/2509.03312) uses counterfactual replay and fault injection to *generate training labels* for a learned tracer.
  - [DoVer](https://arxiv.org/abs/2512.06749) intervenes on messages and plans and re-runs, to *repair* failures. It reports rates without CIs.
  - [Abduct-Act-Predict](https://arxiv.org/abs/2509.10401) has an LLM *simulate* counterfactuals rather than execute them.
- **Using many runs:** [FAMAS](https://arxiv.org/abs/2509.13782) is closest in spirit to (D). It does spectrum-based fault localization over repeated runs, scoring actions by how often they co-occur with failure. That is a correlational score over action frequencies. agentdiff instead aligns trajectories by order and looks for the earliest consistent split, then tests it by intervention.
- **Single failing trace:** [AgentRx](https://arxiv.org/abs/2602.02475) synthesizes constraints and checks them step by step on the failing trace.

## Tracing standard

**OpenTelemetry GenAI semantic conventions** ([spec](https://opentelemetry.io/docs/specs/semconv/gen-ai/), now maintained at [semantic-conventions-genai](https://github.com/open-telemetry/semantic-conventions-genai)).

- **Status:** Development, not stable.
- **Spans:**
  - Inference spans have `gen_ai.operation.name` values such as `chat`, `text_completion` and `generate_content`.
  - Tool spans are `execute_tool {gen_ai.tool.name}`.
  - Agent spans are `invoke_agent {gen_ai.agent.name}`.
- **Key attributes:** `gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.tool.call.id`, `gen_ai.conversation.id` and `error.type`.
- **Content capture is opt-in:** tool arguments and results (`gen_ai.tool.call.arguments`, `gen_ai.tool.call.result`) and messages (`gen_ai.input.messages`, `gen_ai.output.messages`) are all off by default.
- **How agentdiff uses it:** agentdiff reads these spans as an input format.
- **Consequences:**
  - Without opt-in content capture, alignment falls back to structure only (tool names and order).
  - Replay is impossible without captured tool results. That is why agentdiff also ships its own recorder.
  - Because the spec is still in Development, the adapter tolerates both the current and older attribute names, e.g. `gen_ai.system` vs `gen_ai.provider.name`.

## Summary of what is new

1. **Pass-vs-fail alignment within a task (D).** Using a flaky agent's own successful runs as the reference to find the earliest consistent divergence step. FAMAS uses many runs but not alignment. The eval tools compare against hand-written references.
2. **Alignment-guided causal confirmation (C).** Using (D) to pick which steps to intervene on, so replay cost is spent on 1–3 candidates rather than every step. The intervention machinery itself follows ideas credited to Causal Agent Replay and LangGraph time travel.
3. **Evaluated on a real agent with known ground truth.** Injected faults with documented cause steps, compared against an LLM-judge baseline and a naive rule.
4. **A regression gate that is statistical rather than threshold-based (G).** Built on paired tests at the task level, with a measured false-alarm rate.

Items 1, 2 and 4 are combinations of known statistical and sequence-analysis techniques (pass^k estimators, bootstrap, Needleman–Wunsch alignment, Fisher/permutation tests). The novelty claim is about applying them to agent trajectories and validating them, not about new statistics.
