# Copyright 2026 Aevyra AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Convert a line-delimited JSON trace into an :class:`AgentTrace`.

The on-the-wire schema this adapter expects is the smallest possible
projection of Witness's :class:`TraceNode` — one JSON object per line,
fields named the same as Witness fields. If you control the producer
side (e.g. you're emitting traces from a TypeScript / Go / Rust agent),
emit this and you've got Origin compatibility for free.

Each line should look like::

    {"id": "n0", "parent_id": null, "name": "plan",
     "kind": "reason", "prompt_id": "planner", "optimize": true,
     "input": "...", "output": "..."}

Top-level trace metadata (``ideal``, ``metadata``) is optional and lives
on the FIRST line tagged ``"_trace": true``. Everything else is a span.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from aevyra_witness import AgentTrace, TraceNode


def from_jsonl(lines: list[str]) -> AgentTrace:
    """Parse JSONL lines into an AgentTrace.

    Lines tagged with ``"_trace": true`` set trace-level fields
    (``ideal`` / ``metadata``); all others are spans. Order of lines
    determines sibling ordering (parent_id wires up DAG structure).
    """
    nodes: list[TraceNode] = []
    ideal: str | None = None
    trace_metadata: dict[str, Any] = {}

    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        rec = json.loads(raw)

        if rec.get("_trace") is True:
            ideal = rec.get("ideal", ideal)
            trace_metadata = dict(rec.get("metadata") or trace_metadata)
            continue

        nodes.append(
            TraceNode(
                name=rec["name"],
                input=rec.get("input"),
                output=rec.get("output"),
                id=rec.get("id", ""),
                parent_id=rec.get("parent_id"),
                kind=rec.get("kind", "other"),
                prompt_id=rec.get("prompt_id"),
                optimize=bool(rec.get("optimize", False)),
                tokens=int(rec.get("tokens", 0) or 0),
                started_at=rec.get("started_at"),
                ended_at=rec.get("ended_at"),
                error=rec.get("error"),
                metadata=dict(rec.get("metadata") or {}),
            )
        )

    return AgentTrace(nodes=nodes, ideal=ideal, metadata=trace_metadata)


def from_jsonl_file(path: str | Path) -> AgentTrace:
    return from_jsonl(Path(path).read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1] if len(sys.argv) > 1 else "sample_jsonl.jsonl")
    trace = from_jsonl_file(src)
    print(f"Loaded {len(trace.nodes)} spans from {src}")
    print(f"  optimize_prompt_ids = {trace.optimize_prompt_ids}")
    for n in trace.nodes:
        parent = f" (parent={n.parent_id})" if n.parent_id else ""
        print(f"  [{n.kind}] {n.name}  id={n.id}{parent}  prompt={n.prompt_id}")
