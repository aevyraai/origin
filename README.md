# aevyra-origin

**Why did my agent get that wrong?** Point Origin at your pipeline and a
rubric; it runs the pipeline, grades it, and tells you which span(s) in
the pipeline caused the failure — with severity, confidence, and
reasoning grounded in the actual execution.

Origin is the diagnosis stage in the [Aevyra](https://aevyra.ai) stack:

```
Witness  →  captures what happened         (aevyra-witness)
Verdict  →  judges it                      (aevyra-verdict)
Origin   →  finds where it went wrong      (you are here)
Reflex   →  fixes it                       (aevyra-reflex)
```

## Install

```bash
pip install aevyra-origin[anthropic]     # default: Claude
# or
pip install aevyra-origin[openai]        # OpenAI-compat (OpenAI, OpenRouter, Together, local)
```

Python 3.10+.

## Quick start (turnkey)

Instrument your pipeline with `@span`, hand Origin a rubric and a judge,
get back an attribution:

```python
from aevyra_witness.runtime import span
from aevyra_origin import diagnose_pipeline
from aevyra_origin.llm import anthropic_llm
from aevyra_origin.judges import judge_from_verdict
from aevyra_verdict import LLMJudge
from aevyra_verdict.providers import get_provider

@span("classify")
def classify(text): ...

@span("retrieve")
def retrieve(topic): ...

@span("answer", optimize=True, prompt_id="answer_v1")
def answer(q, docs): ...

def my_agent(q):
    topic = classify(q)
    return answer(q, retrieve(topic))

judge = judge_from_verdict(LLMJudge(judge_provider=get_provider("anthropic")))

result = diagnose_pipeline(
    my_agent, "I was charged twice — how do I get a refund?",
    judge=judge,
    rubric="Accurate, grounded in the policy docs, and addresses the user's concern.",
    llm=anthropic_llm(),
)

print(result.render())
```

`diagnose_pipeline` runs your pipeline under a tracer, scores the
captured trace with your judge, and invokes the attribution engine —
all in one call. You get back a ranked list of culprit spans. No
known-good reference output is required.

Don't have a Verdict metric? Pass any `Callable[[AgentTrace], float]`
as `judge=` — including a lambda that wraps your own evaluator.

## Three on-ramps

The turnkey path is the recommended starting point, but Origin's
attribution engine works with any trace you can produce:

1. **Turnkey** — give Origin your pipeline and it handles tracing +
   scoring: `diagnose_pipeline(pipeline, input, judge, rubric, llm)`.
   Your pipeline just needs `@span` decorators from
   `aevyra_witness.runtime`.
2. **Adapter** — if you already emit framework logs (OpenClaw JSONL
   today; LangSmith, OTel, and others are additive), parse them into
   an `AgentTrace` and hand it to Origin:

    ```python
    from aevyra_witness.adapters import from_openclaw_jsonl
    trace = from_openclaw_jsonl(log_lines)
    origin.diagnose(trace=trace, score=0.4, rubric=...)
    ```

3. **Raw** — you already have an `AgentTrace` and a score:

    ```python
    from aevyra_origin import Origin
    origin = Origin(llm=anthropic_llm())
    result = origin.diagnose(trace=my_trace, score=0.4, rubric=...)
    ```

## API

### `diagnose_pipeline(...)` → `Attribution`

```python
result = diagnose_pipeline(
    pipeline, *args,                  # your callable + whatever it takes
    judge=...,                        # Callable[[AgentTrace], float]
    rubric=...,                       # str
    llm=...,                          # Callable[[str], str]
    ideal=None,                       # optional reference output
    trace_metadata=None,              # dict, stored on the trace
    method="all",                     # "critic" | "decomposition" | "ablation" | "all"
    runner=None,                      # ablation replay function (enables ablation)
    ablation_placeholder="null",      # "null" or "ideal"
    ablation_budget=None,             # cap ablation runs
    **kwargs,                         # forwarded to your pipeline
)
```

### `Origin.diagnose(...)` → `Attribution`

```python
result = origin.diagnose(
    trace=...,                        # AgentTrace
    score=...,                        # float, the judge score being explained
    rubric=...,                       # str
    method="all",                     # "critic" | "decomposition" | "ablation" | "all"
    ablation_placeholder="null",
    ablation_budget=None,
)
```

### `Attribution`

```python
result.summary              # str — one-paragraph overview
result.culprits             # list[NodeAttribution], sorted by confidence desc
result.method               # "critic" | "decomposition" | "ablation" | "all"
result.score                # float — the judge score
result.raw                  # dict — pipeline_output, captured_trace, method-level raw outputs

result.top_culprit()        # NodeAttribution | None
result.primary_culprits()   # [NodeAttribution] — severity="primary"
result.by_prompt()          # [PromptAttribution] — roll blame up to prompt_id for Reflex
result.render()             # str — multi-line CLI rendering
result.to_json(indent=2)    # str — JSON serialization
```

### `NodeAttribution`

```python
c.node_name                 # str — matches a name in the trace
c.severity                  # "primary" | "contributing" | "minor"
c.confidence                # float in [0, 1]
c.reasoning                 # str — grounded in the trace
c.node_id                   # str | None — span id (required for DAG traces with repeated names)
c.prompt_id                 # str | None — prompt identity; used by by_prompt() rollup
```

### `Attribution.by_prompt()` → `list[PromptAttribution]`

For DAG traces where the same prompt fires at many call sites (a planner at
step 1, step 2, step 3, ...), Reflex needs to know which *prompt* to update.
`by_prompt()` rolls span-level blame up to the prompt level — mean confidence
across spans sharing a `prompt_id`, max severity, concatenated reasoning.

```python
for pa in result.by_prompt():
    print(pa.prompt_id, pa.severity, pa.confidence)
```

### `judge_from_verdict(metric, *, ...)` → `Judge`

Adapts any Verdict `Metric` (`LLMJudge`, `ExactMatch`, `BleuScore`,
`RougeScore`, custom metrics) to Origin's `Callable[[AgentTrace], float]`
contract. Duck-typed — no hard Verdict dependency.

```python
from aevyra_origin.judges import judge_from_verdict
from aevyra_verdict import LLMJudge

judge = judge_from_verdict(LLMJudge(judge_provider=provider))
```

Customize what gets fed to the metric with `extract_response=...` and
`extract_messages=...` when the defaults (last root span's output, first
root span's input as user message) aren't right for your pipeline.

## CLI

For pre-captured traces (the raw on-ramp):

```bash
aevyra-origin diagnose trace.json \
  --score 0.4 \
  --rubric rubric.txt \
  --llm-model claude-sonnet-4-5 \
  --llm-provider anthropic \
  --method all \
  --output result.json     # optional — writes full Attribution JSON
```

`--rubric -` reads from stdin. `--llm-provider openai` works with any
OpenAI-compatible endpoint; pair it with `--base-url` for OpenRouter,
Ollama, or a local vLLM server. The render (including prompt-level rollup
for Reflex) always goes to stdout.

## Methods

v0 ships with three attribution methods:

- **LLM-as-critic** (`method="critic"`) — one LLM call. The LLM reads the
  rubric, score, and full trace, and returns a ranked list of culprit spans
  with severity, confidence, and reasoning. Fast, general, works for any
  rubric. Best for single-cause failures.
- **Score decomposition** (`method="decomposition"`) — one LLM call. The
  LLM enumerates the rubric's underlying criteria, attributes each criterion
  to the span(s) responsible, and aggregates per-span blame across failed
  criteria. Better at surfacing distributed failures.
- **Ablation** (`method="ablation"`) — causal. For each candidate span,
  replaces its output with a neutral placeholder, re-runs the pipeline via a
  user-supplied `runner`, and re-scores via the `judge`. The only method
  that makes a causal claim — a large score delta means the span is
  genuinely responsible, independent of whether an LLM thinks it looks
  suspicious. Requires a deterministic runner.
- **`method="all"`** — runs all available methods and merges. The two LLM
  methods always run (two LLM calls). Ablation participates when a `runner`
  is supplied; otherwise it's silently skipped. Spans named by multiple
  methods receive a corroboration bonus — merged confidence lies between
  the arithmetic mean and the max, weighted toward the max by how many
  methods agreed.

### Ablation quick start

```python
from aevyra_origin import diagnose_pipeline
from aevyra_witness import AgentTrace

def my_runner(trace: AgentTrace, overrides: dict) -> AgentTrace:
    # Replay the pipeline with overrides[span_id] forced as the output for that span.
    # LLM calls should be cached or mocked for determinism.
    ...

result = diagnose_pipeline(
    my_agent, "how do I refund?",
    judge=judge, rubric=rubric, llm=anthropic_llm(),
    runner=my_runner,
    method="all",
)
```

Ablation cost control: `ablation_budget=N` caps total runs. The raw on-ramp
exposes `candidates=["span_a", "span_b"]` to limit the sweep to specific
span ids.

## Interop with Reflex

Any Reflex `LLM` works directly:

```python
from aevyra_reflex import LLM
from aevyra_origin.llm import LLMFn

reflex_llm = LLM(model="claude-sonnet-4-5")
llm: LLMFn = lambda p: reflex_llm.generate(p, temperature=0.0)
```

## License

Apache-2.0.
