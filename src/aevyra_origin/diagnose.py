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

"""The ``Origin`` class — Origin's primary public entry point.

``Origin.diagnose()`` takes a trace, a score, a rubric, and returns an
``Attribution``. Three attribution methods are available:

    - ``"critic"``:        LLM-as-critic.           One LLM call. Opinionated.
    - ``"decomposition"``: score decomposition.     One LLM call. Structured.
    - ``"ablation"``:      causal ablation sweep.   Re-runs the pipeline
                           per span. Requires a ``runner`` and ``judge``.
    - ``"all"``:           run every available method and merge.
                           The LLM methods always run; ablation
                           participates only when ``runner`` and ``judge``
                           were provided to ``Origin``.

Token accounting
----------------

When using :func:`anthropic_llm` or :func:`openai_llm` from
:mod:`aevyra_origin.llm`, the returned LLM callable exposes a
``tokens_used`` attribute that is read before and after each attribution
method call. The per-method deltas are summed into
:attr:`Attribution.llm_tokens`. Ablation's runner+judge invocations are
counted separately in :attr:`Attribution.ablation_calls`.

Plain lambdas or closures passed as ``llm=`` will work correctly but
won't contribute to token accounting (``tokens_used`` will stay 0).

Resume
------

Pass a :class:`~aevyra_origin.run_store.DiagnoseRun` as ``run=`` to enable
checkpointing. After each method completes, Origin writes a checkpoint to
the run directory. If the process is interrupted (e.g. during a long ablation
sweep), restart with the same ``DiagnoseRun`` and Origin will skip the methods
that already finished::

    store = DiagnoseStore()

    # First call — interrupted mid-ablation
    run = store.new_run()
    try:
        result = origin.diagnose(trace=t, score=0.4, rubric=r, run=run)
    except KeyboardInterrupt:
        pass

    # Resume — critic and decomposition are skipped, ablation continues
    run = store.find_incomplete_run()
    result = origin.diagnose(trace=t, score=0.4, rubric=r, run=run)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from aevyra_witness import AgentTrace

from aevyra_origin.ablation import Judge, Runner, run_ablation
from aevyra_origin.critic import run_critic
from aevyra_origin.decomposition import run_decomposition
from aevyra_origin.llm import LLMFn
from aevyra_origin.result import Attribution, NodeAttribution

logger = logging.getLogger(__name__)

Method = Literal["critic", "decomposition", "ablation", "all"]
VALID_METHODS: tuple[str, ...] = ("critic", "decomposition", "ablation", "all")


# ---------------------------------------------------------------------------
# Checkpoint serialization helpers
# ---------------------------------------------------------------------------


def _serialize_out(out: dict[str, Any]) -> dict[str, Any]:
    """Make a method output dict JSON-serializable.

    ``run_critic`` and ``run_decomposition`` embed ``NodeAttribution`` objects
    in their ``"culprits"`` key. This replaces them with plain dicts so the
    checkpoint can be written as JSON.
    """
    result = dict(out)
    if "culprits" in result:
        result["culprits"] = [
            c.to_dict() if isinstance(c, NodeAttribution) else c
            for c in result["culprits"]
        ]
    return result


def _deserialize_out(d: dict[str, Any]) -> dict[str, Any]:
    """Restore ``NodeAttribution`` objects from a serialized method output dict."""
    result = dict(d)
    if "culprits" in result:
        result["culprits"] = [
            NodeAttribution.from_dict(c) if isinstance(c, dict) else c
            for c in result["culprits"]
        ]
    return result


# ---------------------------------------------------------------------------
# Token accounting helper
# ---------------------------------------------------------------------------


def _read_tokens(llm: LLMFn) -> int:
    """Read the cumulative ``tokens_used`` from an LLM callable, or 0."""
    return int(getattr(llm, "tokens_used", 0))


# ---------------------------------------------------------------------------
# Origin class
# ---------------------------------------------------------------------------


class Origin:
    """Failure attribution for agent pipelines.

    ``Origin`` wraps an LLM and optional pipeline-replay machinery, and
    provides a single method, ``diagnose()``, which takes an
    ``AgentTrace``, a judge score, and the rubric used to produce that
    score, and returns an ``Attribution`` identifying the node(s)
    responsible.

    Example::

        from aevyra_origin import Origin
        from aevyra_origin.llm import anthropic_llm

        origin = Origin(llm=anthropic_llm())
        result = origin.diagnose(trace=my_trace, score=0.4, rubric=my_rubric)
        print(result.render())
        print(f"LLM tokens used: {result.llm_tokens}")

    For ablation (the causal second-opinion method), supply a
    ``runner`` and ``judge``::

        origin = Origin(llm=anthropic_llm(), runner=my_runner, judge=my_judge)
        result = origin.diagnose(
            trace=my_trace, score=0.4, rubric=my_rubric, method="all",
        )

    For checkpointing and resume, pass a :class:`~aevyra_origin.run_store.DiagnoseRun`::

        from aevyra_origin.run_store import DiagnoseStore
        store = DiagnoseStore()
        run = store.new_run()
        result = origin.diagnose(trace=my_trace, score=0.4, rubric=my_rubric, run=run)

    Args:
        llm:    A callable ``(prompt: str) -> str`` — any LLM-ish function.
                Use the factories in ``aevyra_origin.llm`` for common
                backends (they also track token usage), or pass your own wrapper.
        runner: Optional pipeline-replay callable for ablation. Signature
                ``(trace, overrides: dict[span_id, forced_output]) -> trace``.
                See :mod:`aevyra_origin.ablation` for the contract.
        judge:  Optional trace-scoring callable for ablation. Signature
                ``(trace) -> float``. Typically wraps Verdict.
        score_range: The ``(lo, hi)`` range the judge scores in. Used by
                ablation to normalize score deltas into confidences.
                Defaults to Verdict's ``(0.0, 1.0)``.
    """

    def __init__(
        self,
        llm: LLMFn,
        *,
        runner: Runner | None = None,
        judge: Judge | None = None,
        score_range: tuple[float, float] = (0.0, 1.0),
    ):
        if not callable(llm):
            raise TypeError(
                f"llm must be a callable (prompt: str) -> str, got {type(llm).__name__}"
            )
        if runner is not None and not callable(runner):
            raise TypeError(
                f"runner must be callable or None, got {type(runner).__name__}"
            )
        if judge is not None and not callable(judge):
            raise TypeError(
                f"judge must be callable or None, got {type(judge).__name__}"
            )
        if bool(runner) ^ bool(judge):
            raise ValueError(
                "runner and judge must be provided together (ablation needs both)"
            )
        self.llm = llm
        self.runner = runner
        self.judge = judge
        self.score_range = score_range

    @property
    def ablation_available(self) -> bool:
        """Whether ablation can run — i.e. both runner and judge are set."""
        return self.runner is not None and self.judge is not None

    def diagnose(
        self,
        *,
        trace: AgentTrace,
        score: float,
        rubric: str,
        method: Method = "all",
        ablation_placeholder: str = "null",
        ablation_budget: int | None = None,
        run: "Any | None" = None,  # DiagnoseRun | None — avoid circular import
    ) -> Attribution:
        """Attribute a trace's failure to specific node(s).

        Args:
            trace:   The execution trace to diagnose.
            score:   Judge score being explained (typically in [0, 1]).
            rubric:  The rubric/criteria the judge used.
            method:  Attribution method. See the class docstring.
                     ``"all"`` runs the two LLM methods and — if a
                     runner and judge were configured — ablation as
                     well, merging every method's culprits.
            ablation_placeholder: ``"null"`` or ``"ideal"``. Only used
                     when the method includes ablation. Defaults to
                     ``"null"``. See :mod:`aevyra_origin.ablation`.
            ablation_budget: Max number of spans to ablate. ``None``
                     means ablate every span; set a small integer for
                     quick sanity checks on long traces.
            run:     Optional :class:`~aevyra_origin.run_store.DiagnoseRun`.
                     When provided, Origin writes a checkpoint after each
                     method completes and saves the final result on
                     completion. If the run has an existing checkpoint,
                     already-completed methods are skipped (resume).

        Returns:
            An ``Attribution`` with ranked culprits, a summary, and token
            accounting fields.
        """
        if method not in VALID_METHODS:
            raise ValueError(
                f"method must be one of {VALID_METHODS}, got {method!r}"
            )
        if not trace.nodes:
            raise ValueError("cannot diagnose a trace with zero nodes")
        if not rubric or not rubric.strip():
            raise ValueError("rubric must be a non-empty string")
        if method == "ablation" and not self.ablation_available:
            raise ValueError(
                "method='ablation' requires Origin to be constructed with a "
                "runner and judge; got runner={} judge={}".format(
                    self.runner, self.judge
                )
            )

        # --- Load checkpoint if resuming -----------------------------------
        checkpoint = run.load_checkpoint() if run is not None else None
        completed: set[str] = set(checkpoint.completed_methods) if checkpoint else set()
        method_outputs: dict[str, Any] = {}
        if checkpoint:
            for m, raw_out in checkpoint.method_outputs.items():
                method_outputs[m] = _deserialize_out(raw_out)
        llm_tokens: int = checkpoint.llm_tokens if checkpoint else 0
        ablation_calls: int = checkpoint.ablation_calls if checkpoint else 0

        # Save config on first run (no checkpoint yet)
        if run is not None and checkpoint is None:
            run.save_config(
                rubric=rubric,
                method=method,
                score=float(score),
                trace_dict=trace.to_dict(),
            )

        def _save_checkpoint() -> None:
            if run is None:
                return
            from aevyra_origin.run_store import CheckpointState
            run.save_checkpoint(
                CheckpointState(
                    run_id=run.run_id,
                    rubric=rubric,
                    method=method,
                    score=float(score),
                    trace_dict=trace.to_dict(),
                    completed_methods=list(completed),
                    method_outputs={m: _serialize_out(o) for m, o in method_outputs.items()},
                    llm_tokens=llm_tokens,
                    ablation_calls=ablation_calls,
                )
            )

        # --- Single-method dispatch ----------------------------------------
        if method == "critic":
            if "critic" not in completed:
                tok_before = _read_tokens(self.llm)
                out = run_critic(trace=trace, score=score, rubric=rubric, llm=self.llm)
                llm_tokens += _read_tokens(self.llm) - tok_before
                method_outputs["critic"] = out
                completed.add("critic")
                _save_checkpoint()
            else:
                out = method_outputs["critic"]
                logger.info("diagnose: skipping critic (already completed in checkpoint)")
            result = Attribution(
                summary=out["summary"],
                culprits=out["culprits"],
                method="critic",
                score=float(score),
                llm_tokens=llm_tokens,
                raw={"critic": out},
            )
            if run is not None:
                run.save_result(result.to_dict())
            return result

        if method == "decomposition":
            if "decomposition" not in completed:
                tok_before = _read_tokens(self.llm)
                out = run_decomposition(trace=trace, score=score, rubric=rubric, llm=self.llm)
                llm_tokens += _read_tokens(self.llm) - tok_before
                method_outputs["decomposition"] = out
                completed.add("decomposition")
                _save_checkpoint()
            else:
                out = method_outputs["decomposition"]
                logger.info("diagnose: skipping decomposition (already completed in checkpoint)")
            result = Attribution(
                summary=out["summary"],
                culprits=out["culprits"],
                method="decomposition",
                score=float(score),
                llm_tokens=llm_tokens,
                raw={"decomposition": out},
            )
            if run is not None:
                run.save_result(result.to_dict())
            return result

        if method == "ablation":
            assert self.runner is not None and self.judge is not None
            if "ablation" not in completed:
                out = run_ablation(
                    trace=trace,
                    score=score,
                    rubric=rubric,
                    runner=self.runner,
                    judge=self.judge,
                    placeholder=ablation_placeholder,  # type: ignore[arg-type]
                    budget=ablation_budget,
                    score_range=self.score_range,
                )
                ablation_calls += out.get("num_effects", 0)
                method_outputs["ablation"] = out
                completed.add("ablation")
                _save_checkpoint()
            else:
                out = method_outputs["ablation"]
                logger.info("diagnose: skipping ablation (already completed in checkpoint)")
            result = Attribution(
                summary=out["summary"],
                culprits=out["culprits"],
                method="ablation",
                score=float(score),
                ablation_calls=ablation_calls,
                raw={"ablation": out},
            )
            if run is not None:
                run.save_result(result.to_dict())
            return result

        # --- method == "all" ------------------------------------------------
        if "critic" not in completed:
            tok_before = _read_tokens(self.llm)
            critic_out = run_critic(trace=trace, score=score, rubric=rubric, llm=self.llm)
            llm_tokens += _read_tokens(self.llm) - tok_before
            method_outputs["critic"] = critic_out
            completed.add("critic")
            _save_checkpoint()
        else:
            critic_out = method_outputs["critic"]
            logger.info("diagnose: skipping critic (already completed in checkpoint)")

        if "decomposition" not in completed:
            tok_before = _read_tokens(self.llm)
            decomp_out = run_decomposition(trace=trace, score=score, rubric=rubric, llm=self.llm)
            llm_tokens += _read_tokens(self.llm) - tok_before
            method_outputs["decomposition"] = decomp_out
            completed.add("decomposition")
            _save_checkpoint()
        else:
            decomp_out = method_outputs["decomposition"]
            logger.info("diagnose: skipping decomposition (already completed in checkpoint)")

        per_method_culprits: dict[str, list[NodeAttribution]] = {
            "critic": critic_out["culprits"],
            "decomposition": decomp_out["culprits"],
        }
        per_method_summaries: dict[str, str] = {
            "critic": critic_out["summary"],
            "decomposition": decomp_out["summary"],
        }

        if self.ablation_available:
            assert self.runner is not None and self.judge is not None
            if "ablation" not in completed:
                ablation_out = run_ablation(
                    trace=trace,
                    score=score,
                    rubric=rubric,
                    runner=self.runner,
                    judge=self.judge,
                    placeholder=ablation_placeholder,  # type: ignore[arg-type]
                    budget=ablation_budget,
                    score_range=self.score_range,
                )
                ablation_calls += ablation_out.get("num_effects", 0)
                method_outputs["ablation"] = ablation_out
                completed.add("ablation")
                _save_checkpoint()
            else:
                ablation_out = method_outputs["ablation"]
                logger.info("diagnose: skipping ablation (already completed in checkpoint)")
            per_method_culprits["ablation"] = ablation_out["culprits"]
            per_method_summaries["ablation"] = ablation_out["summary"]
        else:
            logger.debug(
                "ablation skipped in method='all' (runner/judge not configured)"
            )

        raw: dict[str, Any] = {k: method_outputs[k] for k in method_outputs}
        merged = _merge(per_method_culprits, trace)
        summary = _merge_summaries(per_method_summaries)
        result = Attribution(
            summary=summary,
            culprits=merged,
            method="all",
            score=float(score),
            llm_tokens=llm_tokens,
            ablation_calls=ablation_calls,
            raw=raw,
        )
        if run is not None:
            run.save_result(result.to_dict())
        return result


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

_SEVERITY_RANK = {"primary": 3, "contributing": 2, "minor": 1}


def _merge(
    per_method: dict[str, list[NodeAttribution]],
    trace: AgentTrace,
) -> list[NodeAttribution]:
    """Merge N independent culprit lists into a single ranked list.

    Per span, confidence is averaged across the methods that named it.
    Severity is the max of any method's severity. Reasoning is
    concatenated with method-prefixed labels so the reader can see
    which method said what — and where the methods disagree.

    A span named by more than one method gets a small corroboration
    bonus: its merged confidence is floor-bumped toward the max of the
    per-method confidences rather than strictly averaged. This
    preserves the intuition that "three methods agree at 0.6 each" is
    stronger than "one method alone at 0.6".

    Keying is by ``node_id`` when present (DAG traces), falling back to
    ``node_name`` (linear traces with unique names). This ensures two
    distinct spans with the same name are not falsely merged.
    """
    order = {node.id: idx for idx, node in enumerate(trace.nodes)}

    def key_of(c: NodeAttribution) -> str:
        return c.node_id or c.node_name

    buckets: dict[str, dict[str, Any]] = {}

    for method, culprits in per_method.items():
        for c in culprits:
            k = key_of(c)
            b = buckets.setdefault(
                k,
                {
                    "confidences": [],
                    "methods": [],
                    "sev": "minor",
                    "parts": [],
                    "node_name": c.node_name,
                    "node_id": c.node_id,
                    "prompt_id": c.prompt_id,
                },
            )
            b["confidences"].append(c.confidence)
            b["methods"].append(method)
            b["sev"] = _max_severity(b["sev"], c.severity)
            if c.reasoning:
                b["parts"].append(f"[{method}] {c.reasoning}")
            if c.prompt_id and not b["prompt_id"]:
                b["prompt_id"] = c.prompt_id

    merged: list[NodeAttribution] = []
    for info in buckets.values():
        conf = _corroborated_confidence(info["confidences"])
        merged.append(
            NodeAttribution(
                node_name=info["node_name"],
                severity=info["sev"],  # type: ignore[arg-type]
                confidence=conf,
                reasoning="\n".join(info["parts"]),
                node_id=info["node_id"],
                prompt_id=info["prompt_id"],
            )
        )
    merged.sort(key=lambda n: (-n.confidence, order.get(n.node_id or "", 1e9)))
    return merged


def _corroborated_confidence(confidences: list[float]) -> float:
    """Combine per-method confidences for one span.

    Single-method spans keep their confidence as-is. Multi-method spans
    receive a small corroboration bonus: the result lies between the
    arithmetic mean and the max, weighted toward the max by how many
    methods named the span. Bounded to ``[0.0, 1.0]``.
    """
    if not confidences:
        return 0.0
    if len(confidences) == 1:
        return max(0.0, min(1.0, confidences[0]))
    avg = sum(confidences) / len(confidences)
    peak = max(confidences)
    weight = 1.0 - 1.0 / len(confidences)
    corroborated = avg + (peak - avg) * weight
    return max(0.0, min(1.0, corroborated))


def _max_severity(a: str, b: str) -> str:
    return a if _SEVERITY_RANK.get(a, 0) >= _SEVERITY_RANK.get(b, 0) else b


def _merge_summaries(per_method: dict[str, str]) -> str:
    """Stitch the per-method summaries into one paragraph."""
    critic = (per_method.get("critic") or "").strip()
    decomp = (per_method.get("decomposition") or "").strip()
    ablation = (per_method.get("ablation") or "").strip()

    parts: list[str] = []
    if critic:
        parts.append(critic)
    tail: list[str] = []
    if decomp:
        tail.append(f"Decomposition: {decomp}")
    if ablation:
        tail.append(f"Ablation: {ablation}")
    if tail:
        suffix = "  (" + "  |  ".join(tail) + ")"
        if parts:
            return parts[0] + suffix
        return suffix.strip("  ()")
    return parts[0] if parts else ""


# ---------------------------------------------------------------------------
# Top-level convenience function
# ---------------------------------------------------------------------------


def diagnose(
    *,
    trace: AgentTrace,
    score: float,
    rubric: str,
    llm: LLMFn,
    method: Method = "all",
    runner: Runner | None = None,
    judge: Judge | None = None,
    score_range: tuple[float, float] = (0.0, 1.0),
    ablation_placeholder: str = "null",
    ablation_budget: int | None = None,
) -> Attribution:
    """One-shot equivalent of ``Origin(llm, runner=..., judge=...).diagnose(...)``.

    Useful when you don't want to hold onto an ``Origin`` instance.
    Pass ``runner`` and ``judge`` if you want ablation to participate.
    """
    return Origin(
        llm=llm, runner=runner, judge=judge, score_range=score_range
    ).diagnose(
        trace=trace,
        score=score,
        rubric=rubric,
        method=method,
        ablation_placeholder=ablation_placeholder,
        ablation_budget=ablation_budget,
    )


__all__ = ["Origin", "diagnose", "Method", "VALID_METHODS"]
