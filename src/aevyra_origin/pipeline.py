# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Turnkey entry point: `diagnose_pipeline(pipeline, input, judge, rubric, llm)`.

This is Origin's answer to "I have an agent, tell me why it's wrong." It
composes the three pieces of the Aevyra stack that Origin needs:

    pipeline   ─runs under─►  aevyra_witness.runtime.trace()
                              │
                              ▼
                          AgentTrace
                              │
                         judge(trace)
                              │
                              ▼
                            score
                              │
                              ▼
               Origin.diagnose(trace, score, rubric)
                              │
                              ▼
                         Attribution

The user hands in three things — a pipeline callable, the input to feed
it, and a judge — plus the rubric and an LLM for the attribution methods
themselves. Everything else (trace capture, composition) is handled
internally.

The pipeline must be instrumented with ``@span`` / ``with span(...):``
from ``aevyra_witness.runtime`` so the tracer can see it. Un-instrumented
code still runs, but produces a trace with no spans — and no spans means
no attribution.

The judge is any ``Callable[[AgentTrace], float]``. To plug in a Verdict
``Metric``, see :func:`aevyra_origin.judges.judge_from_verdict`.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from aevyra_witness import AgentTrace
from aevyra_witness.runtime import trace as _witness_trace

from aevyra_origin.ablation import Judge, Runner
from aevyra_origin.diagnose import Origin
from aevyra_origin.llm import LLMFn
from aevyra_origin.result import Attribution

logger = logging.getLogger(__name__)

#: A pipeline callable. Takes whatever ``input`` the user wants to feed
#: it and returns whatever it wants to return. Must be instrumented with
#: ``aevyra_witness.runtime.span`` so the tracer can observe its spans.
Pipeline = Callable[..., Any]


class PipelineError(RuntimeError):
    """Raised when a pipeline run produces no spans, or the judge misbehaves."""


def diagnose_pipeline(
    pipeline: Pipeline,
    *args: Any,
    judge: Judge,
    rubric: str,
    llm: LLMFn,
    ideal: str | None = None,
    trace_metadata: dict[str, Any] | None = None,
    method: Literal["critic", "decomposition", "ablation", "all"] = "all",
    runner: Runner | None = None,
    score_range: tuple[float, float] = (0.0, 1.0),
    ablation_placeholder: str = "null",
    ablation_budget: int | None = None,
    **kwargs: Any,
) -> Attribution:
    """Run ``pipeline(*args, **kwargs)`` under Witness, score it, attribute failures.

    This is the turnkey on-ramp for users who have a live pipeline and
    just want to know why it's wrong. For users who already have a
    captured trace, use :class:`Origin` directly.

    Args:
        pipeline:       A callable instrumented with ``@span`` /
                        ``with span(...):``. Called once as
                        ``pipeline(*args, **kwargs)``.
        *args, **kwargs: Passed through to the pipeline.
        judge:          ``Callable[[AgentTrace], float]``. Returns the
                        score for the captured trace. Typically wraps a
                        Verdict metric via
                        :func:`aevyra_origin.judges.judge_from_verdict`,
                        or is a user-supplied callable.
        rubric:         The evaluation rubric. Passed to the attribution
                        methods, not the judge — so the judge may use a
                        different internal prompt. Commonly the rubric
                        text and the judge's criteria are the same
                        string.
        llm:            LLM callable for the critic / decomposition
                        attribution methods.
        ideal:          Reference output for the run, stored on the
                        trace. Useful for ablation's ``placeholder="ideal"``
                        strategy and for prompts that reference an
                        expected answer.
        trace_metadata: Trace-level metadata (model name, run id, ...).
        method:         Which attribution method(s) to run. See
                        :class:`Origin` for semantics.
        runner:         Pipeline replay callable for ablation. When
                        given, ``method="ablation"`` and
                        ``method="all"`` include ablation; when omitted,
                        ``method="ablation"`` raises and ``method="all"``
                        silently skips ablation.
        score_range:    ``(min, max)`` range of scores the judge can
                        return. Passed through to ablation for
                        delta normalization.
        ablation_placeholder: ``"null"`` or ``"ideal"`` — placeholder
                        strategy for ablation. Ignored when ablation is
                        not used.
        ablation_budget: Cap on the number of spans ablated. ``None``
                        ablates every eligible span; set a small integer
                        for fast sanity checks on long traces.

    Returns:
        An :class:`Attribution` with the full culprit list, method-level
        raw outputs, and the score the judge produced.

    Raises:
        PipelineError:  The pipeline produced a trace with no spans, or
                        the judge returned a non-finite score.
        ValueError:     Invalid arguments (e.g. bad ``method``).

    Example::

        from aevyra_witness.runtime import trace, span
        from aevyra_origin import diagnose_pipeline
        from aevyra_origin.judges import judge_from_verdict
        from aevyra_origin.llm import anthropic_llm
        from aevyra_verdict import LLMJudge

        @span("classify")
        def classify(text): ...

        @span("answer", optimize=True)
        def answer(q, docs): ...

        def my_agent(q):
            topic = classify(q)
            return answer(q, retrieve(topic))

        judge = judge_from_verdict(LLMJudge(judge_provider=provider))
        result = diagnose_pipeline(
            my_agent, "how do I refund?",
            judge=judge, rubric="Accurate and concise.", llm=anthropic_llm(),
        )
        print(result.render())
    """
    if not callable(pipeline):
        raise ValueError(f"pipeline must be callable, got {type(pipeline).__name__}")
    if not callable(judge):
        raise ValueError(f"judge must be callable, got {type(judge).__name__}")

    # --- 1. Run the pipeline under a Witness tracer ------------------------
    with _witness_trace(ideal=ideal, metadata=trace_metadata) as tracer:
        pipeline_output = pipeline(*args, **kwargs)
    captured_trace: AgentTrace = tracer.finish()

    if not captured_trace.nodes:
        raise PipelineError(
            "pipeline produced an empty trace (no spans captured). "
            "Ensure the pipeline is instrumented with aevyra_witness.runtime.span "
            "— un-instrumented code runs fine but yields nothing to attribute."
        )

    logger.debug("diagnose_pipeline: captured trace with %d spans", len(captured_trace.nodes))

    # --- 2. Ask the judge for a score --------------------------------------
    try:
        raw_score = judge(captured_trace)
    except Exception as e:
        raise PipelineError(f"judge raised an exception: {e}") from e

    score = _coerce_score(raw_score)

    # --- 3. Run Origin's attribution engine --------------------------------
    # Only pass runner/judge to Origin when runner is provided — otherwise
    # Origin.__init__ XOR-validates and rejects judge-without-runner.
    if runner is not None:
        origin = Origin(llm=llm, runner=runner, judge=judge, score_range=score_range)
    else:
        origin = Origin(llm=llm, score_range=score_range)
    result = origin.diagnose(
        trace=captured_trace,
        score=score,
        rubric=rubric,
        method=method,
        ablation_placeholder=ablation_placeholder,
        ablation_budget=ablation_budget,
    )

    # Surface the pipeline output and the captured trace via `raw` so
    # callers can inspect them without rerunning.
    result.raw.setdefault("pipeline_output", pipeline_output)
    result.raw.setdefault("captured_trace", captured_trace.to_dict())
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_score(raw: Any) -> float:
    """Coerce the judge's return value to a finite float.

    Accepts plain floats/ints and Verdict-style ``ScoreResult`` objects
    that expose a ``score`` attribute. Anything else raises.
    """
    # Duck-typing on ScoreResult — avoid a hard Verdict import.
    if hasattr(raw, "score") and not isinstance(raw, (int, float, bool)):
        raw = raw.score
    try:
        score = float(raw)
    except (TypeError, ValueError) as e:
        raise PipelineError(
            f"judge must return a float (or ScoreResult), got {type(raw).__name__}: {raw!r}"
        ) from e
    if score != score:  # NaN
        raise PipelineError("judge returned NaN")
    return score


__all__ = ["PipelineError", "Pipeline", "diagnose_pipeline"]
