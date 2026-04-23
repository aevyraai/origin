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

"""aevyra-origin CLI — diagnose which span caused an agent trace failure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Optional

try:
    import typer
except ImportError:
    print("typer is required for the CLI. Install it with: pip install typer")
    sys.exit(1)


def _version_callback(value: bool) -> None:
    if value:
        from aevyra_origin import __version__
        typer.echo(f"aevyra-origin {__version__}")
        raise typer.Exit()


app = typer.Typer(
    name="aevyra-origin",
    help="Failure attribution for agent pipelines — find which span caused the failure.",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version and exit.",
            callback=_version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """aevyra-origin — agent pipeline failure attribution."""


@app.command()
def diagnose(
    trace_file: Annotated[
        Path,
        typer.Argument(help="Path to a JSON file containing an AgentTrace (AgentTrace.to_dict())."),
    ],
    score: Annotated[
        float,
        typer.Option("--score", help="Judge score for this trace (typically 0.0–1.0)."),
    ],
    rubric: Annotated[
        Optional[Path],
        typer.Option(
            "--rubric",
            help="Path to a text file containing the evaluation rubric. Use '-' to read from stdin.",
        ),
    ] = None,
    llm_model: Annotated[
        str,
        typer.Option("--llm-model", help="LLM model ID (e.g. 'claude-sonnet-4-5', 'gpt-4o')."),
    ] = "claude-sonnet-4-5",
    llm_provider: Annotated[
        str,
        typer.Option(
            "--llm-provider",
            help="LLM provider: 'anthropic' or 'openai'. For OpenAI-compatible endpoints, use 'openai'.",
        ),
    ] = "anthropic",
    base_url: Annotated[
        Optional[str],
        typer.Option(
            "--base-url",
            help="Custom base URL for openai-compatible endpoints (e.g. OpenRouter, Ollama).",
        ),
    ] = None,
    method: Annotated[
        str,
        typer.Option("--method", help="Attribution method: critic, decomposition, ablation, or all."),
    ] = "all",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", help="Write full Attribution JSON to this file."),
    ] = None,
) -> None:
    """Attribute the failure in TRACE_FILE to specific pipeline spans.

    Loads the trace, runs the requested attribution method(s), and prints
    a human-readable report to stdout. With --output, also writes the full
    structured result as JSON.

    Examples::

        aevyra-origin diagnose trace.json --score 0.4 --rubric rubric.txt

        aevyra-origin diagnose trace.json \\
          --score 0.4 --rubric rubric.txt \\
          --llm-model claude-sonnet-4-5 \\
          --llm-provider anthropic \\
          --method all \\
          --output result.json
    """
    from aevyra_origin import Origin, VALID_METHODS
    from aevyra_origin.llm import anthropic_llm, openai_llm

    # --- Validate method -------------------------------------------------------
    if method not in VALID_METHODS:
        typer.echo(
            f"Error: --method must be one of {VALID_METHODS}, got {method!r}.",
            err=True,
        )
        raise typer.Exit(code=1)

    # --- Load trace -----------------------------------------------------------
    if not trace_file.exists():
        typer.echo(f"Error: trace file not found: {trace_file}", err=True)
        raise typer.Exit(code=1)

    try:
        from aevyra_witness import AgentTrace
        trace_json = trace_file.read_text(encoding="utf-8")
        import json
        trace_dict = json.loads(trace_json)
        trace = AgentTrace.from_dict(trace_dict)
    except Exception as exc:
        typer.echo(f"Error: could not load trace from {trace_file}: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Load rubric -----------------------------------------------------------
    rubric_text: str
    if rubric is None:
        typer.echo("Error: --rubric is required.", err=True)
        raise typer.Exit(code=1)
    elif str(rubric) == "-":
        rubric_text = sys.stdin.read()
    else:
        if not rubric.exists():
            typer.echo(f"Error: rubric file not found: {rubric}", err=True)
            raise typer.Exit(code=1)
        rubric_text = rubric.read_text(encoding="utf-8")

    if not rubric_text.strip():
        typer.echo("Error: rubric is empty.", err=True)
        raise typer.Exit(code=1)

    # --- Build LLM factory ---------------------------------------------------
    try:
        if llm_provider == "anthropic":
            llm = anthropic_llm(model=llm_model)
        elif llm_provider == "openai":
            llm = openai_llm(model=llm_model, base_url=base_url)
        else:
            typer.echo(
                f"Error: --llm-provider must be 'anthropic' or 'openai', got {llm_provider!r}.",
                err=True,
            )
            raise typer.Exit(code=1)
    except ImportError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Run attribution ------------------------------------------------------
    try:
        origin = Origin(llm=llm)
        result = origin.diagnose(
            trace=trace,
            score=score,
            rubric=rubric_text,
            method=method,  # type: ignore[arg-type]
        )
    except Exception as exc:
        typer.echo(f"Error during attribution: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Output ---------------------------------------------------------------
    # render() already appends a prompt-level rollup when prompt_ids are
    # present — see result.Attribution.render().
    typer.echo(result.render())

    if output is not None:
        try:
            output.write_text(result.to_json(indent=2), encoding="utf-8")
            typer.echo(f"\nFull attribution written to {output}")
        except Exception as exc:
            typer.echo(f"Error writing output: {exc}", err=True)
            raise typer.Exit(code=1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
