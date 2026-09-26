# Demo: the CI gate catches a bad prompt change

`agent.py` wraps agentdiff's offline toy agent so it behaves like a prompt-sensitive agent:
when the system prompt (`prompt.txt`) tells it to read the failing traceback before choosing
a file, it opens the right file more often. This is a **simulated** model, so the demo runs
in CI in seconds with no API key. Real-agent evidence is in [`experiments/`](../../experiments).

Two candidate pull requests:

| PR | Change | What should happen |
|---|---|---|
| `bad-prompt.patch` | deletes the "read the traceback" instruction | gate fails (exit 1) |
| `harmless-prompt.patch` | rewords the prompt, same instruction | gate passes (exit 0) |

Outputs from running both locally (k = 8 runs per task, 6 tasks) are in `output/`.

- **Bad prompt: exit 1.** The overall pass rate dropped by 0.312, 95% CI [-0.562, -0.083], stratified exact p = 0.00092.
- **Harmless reword: exit 0.** The overall change was -0.042, p = 0.41. Note that in this run one task, `t3_sorting`, fell from 7/8 to 3/8 by chance. A fixed pass-rate threshold would have failed that build. The paired test does not flag it, because a single 8-vs-8 swing like that is within noise once 6 tasks are tested.

In GitHub Actions, `.github/workflows/demo-gate.yml` calls the reusable
`.github/workflows/agentdiff-gate.yml` on every PR touching this folder. The reusable workflow
runs the PR's base commit and head commit on the same tasks and uploads the HTML report.

Reproduce locally:

```bash
agentdiff run --harness examples.demo_pr.agent:make_harness --version baseline --out base.jsonl
patch -p1 < examples/demo_pr/bad-prompt.patch
agentdiff run --harness examples.demo_pr.agent:make_harness --version candidate --seed0 10000 --out cand.jsonl
agentdiff compare --baseline base.jsonl --candidate cand.jsonl   # exit code 1
git checkout examples/demo_pr/prompt.txt
```
