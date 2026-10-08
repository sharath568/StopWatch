# StopWatch

**How much time do you have to say stop?**

**A two-layer detection-and-recovery framework for safe tool use in on-device agentic AI — plus the complete, re-runnable harness used to evaluate it.**

> *"How much time do you have to say stop?"* On-device AI agents (smart home, wearable, in-vehicle) plan and execute multi-step tool calls with no server to roll back to. If an agent drifts from the user's goal or is interrupted mid-workflow, there is no transaction to undo — the device is left in whatever state the completed steps produced. This repo measures that problem and implements two defenses against it.

## Key findings from the included evaluation

Across 4 local models (1B–8B parameters, Q4/Q8 quantization), 12 multi-step household tasks, and 298 labelled tool-call test cases — all run on a single 12 GB Windows laptop, no cloud, no API keys:

- **Faster inference means less time to intervene.** The median time before an agent's first irreversible action (the *intervention window*) ranged from **14.2s to 48.3s** across models, tracking decode speed almost perfectly (Spearman ρ = **−1.00**, 4 models). Explicit pausing closes this gap at bounded cost.
- **Recovery works.** When a workflow is interrupted, residual-state divergence drops from **52.4% → 9.1%** with the recovery layer active, while the layer itself adds a median of under 1% to total task duration.
- **A detection layer matters even with zero adversaries.** In ordinary, non-adversarial task runs with the detection layer switched *off*, agents spontaneously called tools outside what was asked **35.3% of the time** overall — including unrequested alarm-disarms and door-unlocks.
- **A naive keyword-matching judge and a history-aware local-LLM judge block benign calls at about the same rate** (33.6% vs. 32.7%) — but the LLM judge catches **13× more actual drift** (45.0% → 3.3% missed), at a real latency cost (~7.2s/call median).

Full methodology, statistics, and honest limitations are in [`paper/`](paper/) — every number above is generated directly from `results/` by `analyze.py`, not typed by hand.

## How it works

- **Detection layer** (`gate.py`) — four pre-execution checks applied to every proposed tool call, cheapest first: tool-scope allowlist (C1), argument-bounds validation (C2), a pluggable goal-consistency judge (C3), and an irreversibility confirmation gate (C4). Three C3 judges are implemented and compared: keyword matching, a naive local-LLM judge, and a history-aware local-LLM judge with few-shot examples.
- **Recovery layer** (`agent.py`) — a saga-style wrapper: announce an irreversible action, pause for a configurable `tau` during which it can be cancelled, execute, and compensate completed reversible/compensatable steps in reverse order if interrupted.

```
Agent planner (local quantized LLM)
        │
        ▼
Detection layer ── C1 scope · C2 bounds · C3 goal judge · C4 irreversibility ──► blocked / confirm
        │ allowed
        ▼
Recovery layer ── announce → pause(τ) → execute → compensation log ──────────► Human: stop / approve
        │
        ▼
Smart-home tools (reversible · compensatable · irreversible)
```

Everything runs locally against [Ollama](https://ollama.com) — `llm.py` refuses any non-localhost endpoint, so no prompt or device state ever leaves the machine.

## Repo layout

| path | what it is |
|---|---|
| `home.py` | Simulated smart-home environment: 10 tools across 3 effect classes, explicit state, and a final-state classifier (`goal` / `initial` / `diverged`). |
| `gate.py` | Detection layer: C1–C4 checks and the three C3 judge implementations. |
| `agent.py` | Agent loop and recovery layer, plus a repeat-call guard and failed-call feedback into the model's own context. |
| `llm.py` | Ollama HTTP client (localhost-only) and a deterministic stub model for plumbing tests. |
| `make_scenarios.py` | Generates `scenarios/tasks.json` (workflows, including prompt-injection variants) and `scenarios/gate_cases.json` (labelled single-call cases). |
| `run.py` | Experiment driver: `gate`, `window`, `interrupt`, `inject` subcommands. |
| `analyze.py` | Raw `results/*.jsonl` → summary stats, LaTeX tables, figures (task-level bootstrap CIs). |
| `check_pilot.py` | Fast sanity check on a small run before committing to a long one. |
| `label_sheet.py` | Blinded export for a second, independent labeller + Cohen's κ scoring. |
| `demo.py` | Interactive demo — press Enter to say "stop," watch the recovery layer react live. |
| `paper/` | The paper this harness supports (LaTeX + compiled PDF), plus `paper_out/` where `analyze.py` writes every table, figure, and in-text number the paper uses directly. |
| `results/` | Raw experiment output (JSON Lines), including the exact data behind every number above. |

## Quickstart

```bash
git clone https://github.com/sharath568/StopWatch.git
cd StopWatch
python -m venv .venv && source .venv/bin/activate   # .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt

ollama pull qwen2.5:3b-instruct-q4_K_M
ollama pull llama3.2:1b-instruct-q4_K_M

python make_scenarios.py
```

Smoke-test the whole pipeline with no model server required:
```bash
python run.py window --stub --reps 1 --models m1
python analyze.py results/window_*STUB.jsonl --allow-stub
rm results/*STUB*.jsonl   # stub output is not meaningful and must never be analyzed for real
```

## Running the experiments

```bash
python run.py gate --models qwen2.5:3b-instruct-q4_K_M
python run.py window --models llama3.2:1b-instruct-q4_K_M qwen2.5:3b-instruct-q4_K_M qwen2.5:3b-instruct-q8_0 llama3.1:8b-instruct-q4_K_M --reps 3
python run.py interrupt --models llama3.2:1b-instruct-q4_K_M qwen2.5:3b-instruct-q4_K_M qwen2.5:3b-instruct-q8_0 --reps 3
python run.py inject --models qwen2.5:3b-instruct-q4_K_M --reps 3
```

Each command writes one timestamped `results/<cmd>_<timestamp>.jsonl`. The first row of every file records the full environment (CPU, RAM, OS, Ollama version, git commit) so every run is traceable. Failed calls are recorded as error rows, not crashes.

Optional — evaluate the detection layer running live inside a normal workflow, not just on isolated gate cases:
```bash
python run.py window --models qwen2.5:3b-instruct-q4_K_M qwen2.5:3b-instruct-q8_0 --reps 3 --use-gate --judge objective_plus
```

### A note on the small (1B) model

`llama3.2:1b` is included deliberately as a capability floor, not a bug report. In our runs it sometimes sends malformed tool-call shapes (e.g. one dict key per room instead of a `{"room": ..., "level": ...}` pair) despite the system prompt stating the exact expected shape — a real finding about model scale, not something we prompt-engineered away. The repeat-call guard in `agent.py` exists to stop a stuck model from burning its entire step budget on an identical failing call (`stopped_by: "repeat_loop"` in the output), not to paper over the underlying limitation.

## Reproducing the paper's numbers

```bash
python analyze.py results/gate_*.jsonl results/window_*.jsonl results/interrupt_*.jsonl results/inject_*.jsonl
cat paper/paper_out/summary.txt
cd paper && pdflatex ss06.tex && pdflatex ss06.tex
```

`paper/ss06.tex` pulls every table, figure, and in-text statistic directly from `paper_out/`, so the paper and the data can never silently drift apart.

**Deterministic sanity check:** `KeywordJudge` is plain code with no model in the loop, so a `gate` run must always reproduce FP = 33.6%, FN = 45.0%, C1 catching 48/48 out-of-scope calls, and C2 catching 20/20 out-of-bounds calls. If your numbers differ, something in `scenarios/` differs from what's committed here.

## Second-labeller / inter-annotator agreement

```bash
python label_sheet.py export     # writes label_sheet.csv (blinded) + label_key.private.json (keep private)
# send label_sheet.csv to one other person; have them fill in the last column
python label_sheet.py score      # prints raw agreement and Cohen's kappa, with each disagreement shown
```

## Demo

```bash
python demo.py --model qwen2.5:3b-instruct-q4_K_M --task movie_night --mode none              # no recovery layer
python demo.py --model qwen2.5:3b-instruct-q4_K_M --task movie_night --mode recovery --tau 5  # with it
```
Press Enter at any point to say "stop" and see what each mode leaves behind.

## Citation

```bibtex
@inproceedings{kampili2026stoptime,
  title     = {How Much Time Do You Have to Say Stop? A Detection-and-Recovery Framework for Safe Tool Use in On-Device Agentic AI},
  author    = {Kampili, Sharath Chandra},
  year      = {2026}
}
```
(Full venue and page details will be added once publication is finalized.)

## License

MIT — see [LICENSE](LICENSE).
