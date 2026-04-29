# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a Langfuse trace export into an :class:`AgentTrace`.

Langfuse stores agent runs as a tree:

    Trace
      ├── Observation (type: GENERATION | SPAN | EVENT)
      ├── Observation
      └── Observation

Each observation has ``id``, ``parentObservationId``, ``name``, ``type``,
``input``, ``output``, ``metadata``, optional ``model`` / ``usage`` fields,
and ISO-8601 ``startTime`` / ``endTime``.

This adapter is ~50 lines because the schemas line up almost 1:1. The
only real work is deciding the Witness ``kind`` from Langfuse's ``type``
and reading our convention fields (``prompt_id``, ``optimize``) out of
the observation's ``metadata`` dict.

Use it as a starting template for your own observability stack — the
shape of the function won't change much for LangSmith, Phoenix,
Braintrust, or a home-grown JSONL trace store. The Witness API surface
is the contract.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from aevyra_witness import KIND_OTHER, KIND_REASON, KIND_TOOL, AgentTrace, TraceNode


def _kind_for(obs: dict[str, Any]) -> str:
    """Map Langfuse observation type → Witness kind."""
    obs_type = (obs.get("type") or "").upper()
    if obs_type in ("GENERATION", "AGENT", "CHAIN"):
        return KIND_REASON
    if obs_type in ("SPAN", "TOOL", "RETRIEVER"):
        # SPAN is generic in Langfuse; in practice it's most often a tool call.
        # If you have your own naming convention, branch here on `obs["name"]`.
        return KIND_TOOL
    return KIND_OTHER


def _ts(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def from_langfuse_export(payload: dict[str, Any]) -> AgentTrace:
    """Turn a Langfuse trace export (one trace + its observations) into an AgentTrace.

    Args:
        payload: The decoded JSON of a single Langfuse trace export.
                 Expected top-level keys: ``id``, ``name``, ``input``,
                 ``output``, ``metadata``, ``observations``.

    Returns:
        AgentTrace ready to hand to ``Origin.diagnose``.
    """
    nodes: list[TraceNode] = []
    for obs in payload.get("observations", []):
        meta = dict(obs.get("metadata") or {})
        # Move our convention fields out of metadata so they're first-class.
        prompt_id = meta.pop("prompt_id", None)
        optimize = bool(meta.pop("optimize", False))
        # Token usage from Langfuse → Witness `tokens` (single number).
        usage = obs.get("usage") or {}
        tokens = int(usage.get("totalTokens", 0) or 0)
        # Carry model name and any leftover metadata through.
        if model := obs.get("model"):
            meta.setdefault("model", model)

        nodes.append(
            TraceNode(
                name=obs["name"],
                input=obs.get("input"),
                output=obs.get("output"),
                id=obs["id"],
                parent_id=obs.get("parentObservationId"),
                kind=_kind_for(obs),
                prompt_id=prompt_id,
                optimize=optimize,
                tokens=tokens,
                started_at=_ts(obs.get("startTime")),
                ended_at=_ts(obs.get("endTime")),
                metadata=meta,
            )
        )

    trace_meta = dict(payload.get("metadata") or {})
    ideal = trace_meta.pop("ideal", None)
    return AgentTrace(nodes=nodes, ideal=ideal, metadata=trace_meta)


def from_langfuse_file(path: str | Path) -> AgentTrace:
    """Convenience wrapper — read a Langfuse JSON file and convert it."""
    return from_langfuse_export(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# CLI — print a quick summary so you can sanity-check the conversion
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_langfuse.json")
    trace = from_langfuse_file(src)
    print(f"Loaded {len(trace.nodes)} spans from {src}")
    print(f"  ideal = {trace.ideal!r}")
    print(f"  optimize_prompt_ids = {trace.optimize_prompt_ids}")
    for n in trace.nodes:
        parent = f" (parent={n.parent_id})" if n.parent_id else ""
        print(f"  [{n.kind}] {n.name}  id={n.id}{parent}  prompt={n.prompt_id}")

    out = Path("trace.json")
    out.write_text(json.dumps(trace.to_dict(), indent=2, default=str), encoding="utf-8")
    print(f"\ntrace saved → {out}")
