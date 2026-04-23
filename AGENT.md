# AGENT.md

## Project overview

`aevyra-origin` is the **failure attribution** layer of the Aevyra stack.
Given an `AgentTrace` (from `aevyra-witness`) and a judge score (from
`aevyra-verdict`), Origin identifies which span(s) in the pipeline caused
the low score — and rolls span-level blame up to the prompt level so
Reflex knows what to optimize.

```
Witness  →  captures what happened         (aevyra-witness)
Verdict  →  judges it                      (aevyra-verdict)
Origin   →  finds where it went wrong      (this package)
Reflex   →  fixes it                       (aevyra-reflex)
```

v0 ships with three attribution methods (LLM-as-critic, score decomposition,
and causal ablation) and a Python API that handles full DAG traces
(N-step reasoning with M-parallel tools). The CLI and test suite are built.

The package also exposes a **turnkey entry point** — `diagnose_pipeline` —
that runs the user's instrumented pipeline under a Witness tracer, scores
the captured trace with a user-supplied judge, and dispatches into the
attribution engine. This is the on-ramp most users should reach for. The
raw `Origin.diagnose(trace, score, rubric)` surface still exists for users
who capture traces some other way.

## Architecture

```
src/aevyra_origin/
├── __init__.py          # Public API exports
├── result.py            # Attribution, NodeAttribution, PromptAttribution
├── llm.py               # LLMFn type + anthropic_llm / openai_llm factories
├── prompts.py           # CRITIC_PROMPT, DECOMPOSITION_PROMPT, formatters
├── _json.py             # extract_json() — tolerant JSON parser
├── critic.py            # LLM-as-critic method — run_critic(), CriticError
├── decomposition.py     # Score decomposition method — run_decomposition(),
│                        # DecompositionError
├── ablation.py          # Causal ablation method — run_ablation(),
│                        # AblationError, Runner, Judge, VALID_PLACEHOLDERS
├── diagnose.py          # Origin class + top-level diagnose() convenience
├── pipeline.py          # Turnkey — diagnose_pipeline(), PipelineError, Pipeline
└── judges.py            # Verdict adapter — judge_from_verdict(),
                         # default_response_from_trace, default_messages_from_trace
```

Public API surface (from `aevyra_origin`):

```python
Origin, diagnose                         # raw on-ramp
diagnose_pipeline                        # turnkey on-ramp (wraps Witness runtime + Origin)
Attribution, NodeAttribution,            # result types
PromptAttribution
VALID_METHODS, VALID_SEVERITIES,         # string constants
VALID_PLACEHOLDERS
CriticError, DecompositionError,         # method-specific exceptions
AblationError, PipelineError
Pipeline, Runner, Judge                  # callable type aliases
```

From `aevyra_origin.judges` (kept off the top-level surface to avoid an
import-time edge that would pull Verdict types into scope):

```python
judge_from_verdict                       # turn Verdict Metric into a Judge
default_response_from_trace              # pull final output from trace
default_messages_from_trace              # pull user-facing messages from trace
```

From `aevyra_origin.llm`:

```python
LLMFn                                    # Callable[[str], str] type alias
anthropic_llm, openai_llm                # factories
```

## Key concepts

### Three on-ramps

Users arrive with different amounts of structure already in place:

1. **Turnkey — I have a pipeline.** Hand in a callable instrumented
   with `@span`, plus a rubric and a judge. `diagnose_pipeline` handles
   tracing, scoring, and attribution in one call. This is the default
   recommendation — it hides Witness and Verdict as implementation
   details.
2. **Adapter — I have framework logs.** Parse the logs into an
   `AgentTrace` via `aevyra_witness.adapters` (OpenClaw JSONL today;
   additive for LangSmith, OTel, etc.), then call `Origin.diagnose`.
3. **Raw — I have a trace and a score.** Call `Origin.diagnose(trace,
   score, rubric)` directly. This is what every on-ramp eventually
   funnels into, and what the CLI consumes.

All three paths produce the same `Attribution`. Keep it that way — new
on-ramps should compose with the existing engine, not fork it.

### The Witness runtime dependency

`diagnose_pipeline` depends on `aevyra_witness.runtime` (the `@span`
decorator and `trace()` context manager). The runtime is a live tracer
that populates an `AgentTrace` — the same `AgentTrace` the adapter and
raw on-ramps use. This is deliberate: runtime capture, adapter import,
and hand-built traces all converge on one schema.

Origin does **not** own the runtime — it lives in Witness. Origin's
turnkey path is just orchestration: open a Witness trace, run the
pipeline inside it, hand the captured trace to the attribution engine.

### The Verdict adapter

Origin's `Judge` type is `Callable[[AgentTrace], float]`. Verdict's
`Metric.score(response, ideal, messages)` returns a `ScoreResult`. The
shapes don't match, so `judges.judge_from_verdict(metric)` bridges them:

- Extracts a response string from the trace (default: last root span's
  output, JSON-encoded if non-string).
- Extracts a messages list from the trace (default: first root span's
  input wrapped as a user message if it's a string; `None` otherwise).
- Calls the metric, unwraps `.score` via duck-typing, coerces to float.
- No top-level `aevyra_verdict` import — Verdict stays an optional
  dependency. The adapter works with any object that quacks like a
  Metric (useful for tests and for users who don't want the full
  Verdict install).

### Span-level attribution, prompt-level rollup

Origin attributes failures at the **span** level — a specific execution
of a prompt or tool, identified by `node_id` from the trace. For DAG
traces where the same prompt fires at many call sites (a planner at
step 1, step 2, step 3, ...), every call site is a separate culprit
candidate with its own confidence and reasoning.

Callers who want to know which *prompt* to optimize (Reflex) should
use `Attribution.by_prompt()`, which rolls span-level blame up to the
prompt level: mean confidence across spans sharing a `prompt_id`, max
severity, concatenated reasoning.

### The `LLMFn` contract

Origin's public API takes any callable `(prompt: str) -> str` as its LLM.
No class hierarchy, no configuration object — just a callable. All the
factories in `llm.py` return callables that default to `temperature=0.0`
because attribution is a reasoning task and stability matters.

This means:
- Any Reflex `LLM` can be wrapped in a lambda: `lambda p: reflex_llm.generate(p, temperature=0.0)`.
- Any dummy/stub in tests is trivially `def stub(p): return ...`.
- Anthropic, OpenAI-compat, Ollama, custom clients — all reduce to the same shape.

### Three attribution methods

The two LLM methods are **reference-free** (no known-good output
required) and **single-LLM-call**. Ablation is a **causal** method that
perturbs the pipeline and re-scores. All three are orchestrated by
`diagnose.py`.

- **LLM-as-critic** (`run_critic`) — The LLM reads the rubric, the score,
  the ideal (if any), and the full execution trace. It returns a ranked
  list of culprit spans with severity, confidence, and reasoning. This
  is Origin's default for single-cause failures.

- **Score decomposition** (`run_decomposition`) — The LLM enumerates the
  rubric's underlying criteria, attributes each criterion to the span(s)
  responsible (with contribution weights summing to 1.0 per criterion),
  and per-span blame is aggregated across failed criteria. Better at
  surfacing distributed failures.

- **Ablation** (`run_ablation`) — For each candidate span, replaces its
  output with a neutral placeholder, re-runs the pipeline via a
  user-supplied `runner`, re-scores via a user-supplied `judge`, and
  computes the score delta. The only method that makes a causal claim
  — a large delta means the span is really responsible, independent of
  whether an LLM thinks it looks suspicious. Expensive: one pipeline
  re-run per candidate. Requires a deterministic runner (user caches /
  mocks side effects).

- **`method="all"`** — runs every available method and merges. LLM
  methods always participate (2 LLM calls). Ablation participates only
  when `Origin` was constructed with `runner=` and `judge=`; otherwise
  it's silently skipped with a debug log. Confidence is merged via a
  corroboration-bonus formula — single-method spans keep their
  confidence, multi-method spans lie between the arithmetic mean and
  the max (weighted toward the max by how many methods agreed).
  Severity is the max; reasoning is concatenated with `[critic]`,
  `[decomposition]`, `[ablation]` prefixes so the reader can see where
  the methods agree and where they disagree.

### Ablation contracts

`run_ablation` requires two user-supplied callables with deliberately
narrow contracts:

```python
Runner = Callable[[AgentTrace, dict[str, Any]], AgentTrace]
Judge  = Callable[[AgentTrace], float]
```

The runner takes the original trace and a `span_id → forced_output`
override map; it must replay the pipeline and return a new trace where
the overridden spans have the forced outputs and everything downstream
re-executed against them. The judge takes a trace and returns a score.

Both are user-supplied because Origin must not know how to execute the
user's pipeline — that's what makes it framework-agnostic. In return,
the user is responsible for determinism: LLM calls must be cached or
mocked, tool calls must be replayable. Without determinism, score
deltas reflect noise, not span contribution.

Two placeholder strategies (`placeholder="null"` or `"ideal"`):

- `"null"` — replace output with a neutral value (`""`, `[]`, `{}`,
  `None` depending on the original type). Answers "does this span
  carry any signal?" Default.
- `"ideal"` — replace with the trace's `ideal` output. Answers "would
  a perfect output here have saved the run?" Falls back to `"null"`
  with a warning if `trace.ideal is None`.

Cost control: `candidates=["id1", "id2"]` limits ablation to specific
span ids; `budget=N` caps total runs (candidates taken in trace order).
`min_delta=0.05` is the noise floor — spans whose ablated score moves
less than this are reported in `raw["ablation"]["effects"]` but not
promoted to the `culprits` list.

Harmful spans: when `raw_delta < 0`, ablating the span *improved* the
score. The direction is surfaced in the effect dict and called out in
the span's reasoning ("actively degrading the output — its removal is
a net win"). These are valuable diagnostic signals and are included as
culprits with the appropriate absolute-delta confidence.

### Span identity in LLM outputs

Both prompts instruct the LLM to cite culprits by both `node_id` (from
the trace header `id=...`) and `node_name`. The validator (`_resolve_span`
in `critic.py` and `decomposition.py`) enforces:

- If `node_id` is provided → must exist in the trace (`trace.by_id()`).
  Authoritative if both are given.
- If only `node_name` is given → it must be unique in the trace; if
  the name repeats, the relevant method's Error is raised.
- Mismatched name/id (same id, wrong name) → warning logged, id wins.

This is intentional: in DAG traces with repeated names, we refuse to
guess which span the LLM meant.

### Attribution output

`Attribution` has:
- `summary` — one-paragraph overview
- `culprits` — ranked `list[NodeAttribution]` (descending confidence)
- `method` — "critic" | "decomposition" | "all"
- `score` — the input score being explained
- `raw` — method-level raw outputs (including the unparsed LLM response)

`NodeAttribution` has:
- `node_name` — human label, matches a name in the trace
- `severity` — "primary" | "contributing" | "minor"
- `confidence` — float in [0.0, 1.0]
- `reasoning` — one paragraph, grounded in trace content
- `node_id` — optional span id (required when names repeat)
- `prompt_id` — optional prompt identity, copied from the span

`Attribution.by_prompt()` → `list[PromptAttribution]`:
- `prompt_id` — the prompt the rollup is for
- `severity` — max severity across its spans
- `confidence` — mean confidence across its spans
- `spans` — underlying per-span culprits
- `reasoning` — concatenated per-span reasoning, labeled by span

### Prompt design

Both prompts live in `prompts.py` as module-level strings. Each demands
strict JSON output and explicitly forbids prose, markdown fences, and
commentary. The tolerant parser in `_json.py` nonetheless handles the
common violations (fences, trailing text) because LLMs defy instructions.

A shared `_TRACE_GUIDE` block explains the DAG model to the LLM (parallel
siblings, repeated names, the meaning of `id=...`), and is substituted
into both prompts via `format_critic_prompt()` / `format_decomposition_prompt()`.
The guide also enumerates three tool-failure modes — wrong arguments
(blame the reasoning), tool error (blame the tool, possibly with
contributing blame on the reasoning that didn't handle it), and
misinterpreted output (blame the downstream reasoning) — so the critic
reasons about tool failures structurally rather than defaulting to
blame-the-prompt.

If you rewrite a prompt, keep the placeholder contract the same:
- Both templates expect: `trace_guide`, `rubric`, `score`, `ideal`, `trace_text`
- Both require `node_id` AND `node_name` in each culprit entry.

### Validation

Every method validates culprit references against the input trace. If
the LLM hallucinates a span or cites an ambiguous name without an id,
the relevant `Error` is raised — this is intentional. Severities
outside `VALID_SEVERITIES` also raise. Confidences outside [0, 1] are
silently clamped. Contribution weights that don't sum to 1.0 are
normalized.

## What's been built

### Tests (`tests/`)

Eight test files, no live LLM calls. Each method has its own file;
shared fixtures (linear trace, DAG trace, stub runner, parametric judge)
live at the top of the files that need them.

- **`test_result.py`** — `NodeAttribution` construction, validation, and
  round-trip; `Attribution` helpers and serialization; `by_prompt()` rollup
  (mean confidence, max severity, labeled reasoning, sort order).
- **`test_json.py`** — `extract_json` on raw JSON, fenced JSON, prose
  wrappers, braces-inside-strings, and malformed inputs.
- **`test_critic.py`** — linear and DAG traces; name-only vs id-cited
  culprits; ambiguous name error; unknown id/name error; severity/confidence
  validation; fenced LLM response; name/id mismatch warning.
- **`test_decomposition.py`** — single and multiple failed criteria;
  severity thresholds; weight normalization; DAG aggregation keyed by id;
  all-criteria-passed case.
- **`test_ablation.py`** — culprit ordering, confidence normalization,
  severity thresholds; harmful spans; `_build_placeholder` for all types;
  `candidates` / `budget` / `min_delta`; runner failure isolation;
  total-failure `AblationError`; non-AgentTrace and non-numeric errors;
  ideal placeholder fallback warning.
- **`test_diagnose.py`** — all four methods; `method="all"` with/without
  runner; construction validation; merge logic; `_corroborated_confidence`
  formula; merge keys on `node_id`; `by_prompt()` on merged result;
  top-level `diagnose()` equivalence.
- **`test_pipeline.py`** — `diagnose_pipeline` end-to-end with stubbed
  LLM + judge; `raw` surfacing (`pipeline_output`, `captured_trace`);
  `ideal` and `trace_metadata` propagation to the tracer; `*args/**kwargs`
  forwarded to the pipeline; judge return-value coercion (int, ScoreResult
  duck-type, NaN, garbage, exceptions); uninstrumented pipeline raises
  `PipelineError("empty trace")`; ablation-integration via a rebuild
  runner.
- **`test_judges.py`** — `default_response_from_trace` (string / None /
  non-string / multiple roots / empty trace); `default_messages_from_trace`
  (string input / non-string / empty); `judge_from_verdict` core behaviour
  (passes through response/ideal/messages; trace.ideal fallback vs.
  explicit override; plain-number metric return; non-numeric raises);
  custom `extract_response` / `extract_messages` overrides; a live
  `aevyra_verdict.ExactMatch` integration block gated by
  `pytest.importorskip`.

### CLI (`src/aevyra_origin/cli.py`)

Typer-based, `no_args_is_help=True`, `Annotated` options, version callback.
The `[project.scripts]` entry in `pyproject.toml` is enabled.

```bash
aevyra-origin diagnose TRACE_FILE \
  --score 0.4 \
  --rubric RUBRIC_FILE \
  --llm-model claude-sonnet-4-5 \
  --llm-provider anthropic \
  --method all \
  --output result.json
```

`--rubric -` reads from stdin. `--base-url` is available for non-OpenAI
providers. Default output is `result.render()` to stdout; `--output` additionally
dumps `Attribution.to_json(indent=2)` to a file.

## What's next

- **Heterogeneous optimization targets** — today Reflex can only optimize
  prompts; Witness needs a lightweight schema change (`optimize` from bool
  to a string like `"prompt" | "tool_schema" | "tool_choice"`) so Origin
  can emit attributions Reflex can act on across the stack. See
  the Witness schema for where the flag lives.
- **Bootstrap-style learning from successful traces** — complement failure
  attribution with success attribution: given a high-scoring trace, which
  spans carried the load? Feeds Reflex's few-shot selection and prompt
  library. Largely the same method surface (critic, decomposition,
  ablation) with an inverted rubric framing.
- **`aevyra-origin batch`** — process multiple traces in one invocation.
  Useful once there's enough trace volume to amortize setup cost.

## Development

```bash
pip install -e ".[dev]"       # installs pytest, ruff, typer, anthropic, openai
pytest tests/ -v
ruff check src/ tests/
```

Install modes:

```bash
pip install -e .              # library only (no LLM backend, no CLI)
pip install -e ".[anthropic]" # + Anthropic backend
pip install -e ".[openai]"    # + OpenAI-compat backend
pip install -e ".[all]"       # both backends
pip install -e ".[dev]"       # everything above + typer for the CLI + pytest + ruff
```

The `aevyra-origin` CLI script is always registered (via `[project.scripts]`)
but requires `typer` at runtime; `cli.py` emits a friendly error if it's
missing. Any of `[dev]`, a user-installed `typer`, or adding `typer` to
your own deps makes the CLI available.

## Conventions

- **Apache 2.0 license header** on every `.py` file (copy from `result.py`).
- **`from __future__ import annotations`** at the top of every module.
- **Type hints everywhere.**
- **Logging** via `logging.getLogger(__name__)` — already set up in `critic.py`,
  `decomposition.py`, and `diagnose.py`.
- **No print statements** in library code.
- **CLI** uses typer. Follow Reflex's patterns in `aevyra-reflex`'s `cli.py`.
- **Public vs. private**: names starting with `_` are private to the module
  (e.g., `_json.py`, `_merge()`, `_clamp()`, `_resolve_span()`). Everything
  public is re-exported from `__init__.py`.
- **Error messages** should be specific — include what was expected, what
  was received, and (when relevant) a snippet of the offending input truncated
  to ~500 chars. See `_json.extract_json` and `critic._resolve_span` for examples.
- **Dataclasses** for every structured value. No dicts-as-records in public API.

## Dependencies

Runtime: `aevyra-witness>=0.1.0` only. Everything else is an optional extra
(`anthropic`, `openai`) or a dev dep (`pytest`, `ruff`, `typer`).

Keep it this way. The core library must remain dependency-light — Origin
should be usable with any LLM backend the user wires up, and should never
force a specific SDK on them.

## Out of scope for v0

- **Counterfactual replay** (compare against known-good reference outputs).
  Ablation is Origin's causal method for v0; full counterfactual replay
  against human-authored reference outputs adds the engineering burden of
  sourcing and maintaining those references. Revisit when there's demand.
- **Aevyra dashboard integration**. Later.
- **Subtree attribution** (blame an entire branch of the DAG, not just
  individual spans). The data is there — walking `children_of()` — but the
  product question of when a subtree blame is more useful than its root
  span isn't settled. Deferred until real usage forces the answer.
