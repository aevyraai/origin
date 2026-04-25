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
        typer.Option(
            "--method", help="Attribution method: critic, decomposition, ablation, or all."
        ),
    ] = "all",
    output: Annotated[
        Optional[Path],
        typer.Option("--output", help="Write full Attribution JSON to this file."),
    ] = None,
    run_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--run-dir",
            help="Directory for run history and checkpoints. Defaults to .origin/ in cwd.",
        ),
    ] = None,
    resume: Annotated[
        bool,
        typer.Option("--resume", help="Resume the latest interrupted run."),
    ] = False,
    resume_from: Annotated[
        Optional[str],
        typer.Option(
            "--resume-from",
            help="Resume a specific run by ID (e.g. '001').",
        ),
    ] = None,
) -> None:
    """Attribute the failure in TRACE_FILE to specific pipeline spans.

    Loads the trace, runs the requested attribution method(s), and prints
    a human-readable report to stdout. With --output, also writes the full
    structured result as JSON.

    Token usage is shown in the report header when using the built-in LLM
    factories. Use --run-dir to persist run history and enable --resume.

    Examples::

        aevyra-origin diagnose trace.json --score 0.4 --rubric rubric.txt

        aevyra-origin diagnose trace.json \\
          --score 0.4 --rubric rubric.txt \\
          --llm-model claude-sonnet-4-5 \\
          --llm-provider anthropic \\
          --method all \\
          --run-dir .origin \\
          --output result.json

        # Resume the latest interrupted run
        aevyra-origin diagnose trace.json --score 0.4 --rubric rubric.txt --resume

        # Resume a specific run
        aevyra-origin diagnose trace.json --score 0.4 --rubric rubric.txt --resume-from 002
    """
    from aevyra_origin import Origin, VALID_METHODS
    from aevyra_origin.llm import anthropic_llm, openai_llm
    from aevyra_origin.run_store import DiagnoseStore

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
        import json

        trace_dict = json.loads(trace_file.read_text(encoding="utf-8"))
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

    # --- Resolve run (checkpointing / resume) --------------------------------
    run = None
    if run_dir is not None or resume or resume_from:
        store = DiagnoseStore(root=run_dir or ".origin")
        if resume_from:
            run = store.get_run(resume_from)
            if run is None:
                typer.echo(f"Error: run '{resume_from}' not found in {store.runs_dir}", err=True)
                raise typer.Exit(code=1)
            if run.is_complete():
                typer.echo(
                    f"Run {resume_from} is already complete. Use 'aevyra-origin runs' to inspect it."
                )
                raise typer.Exit(code=0)
            typer.echo(f"Resuming run {run.run_id} from {run.path.name} ...")
        elif resume:
            run = store.find_incomplete_run()
            if run is None:
                typer.echo("No interrupted run found. Starting a new run.")
                run = store.new_run()
            else:
                ckpt = run.load_checkpoint()
                done = ckpt.completed_methods if ckpt else []
                typer.echo(f"Resuming run {run.run_id} — already completed: {done or 'none'}")
        else:
            run = store.new_run()

    # --- Run attribution ------------------------------------------------------
    try:
        origin = Origin(llm=llm)
        result = origin.diagnose(
            trace=trace,
            score=score,
            rubric=rubric_text,
            method=method,  # type: ignore[arg-type]
            run=run,
        )
    except Exception as exc:
        typer.echo(f"Error during attribution: {exc}", err=True)
        raise typer.Exit(code=1)

    # --- Output ---------------------------------------------------------------
    typer.echo(result.render())

    if run is not None:
        typer.echo(f"\nRun saved to {run.path}")

    if output is not None:
        try:
            output.write_text(result.to_json(indent=2), encoding="utf-8")
            typer.echo(f"Full attribution written to {output}")
        except Exception as exc:
            typer.echo(f"Error writing output: {exc}", err=True)
            raise typer.Exit(code=1)


@app.command()
def runs(
    run_dir: Annotated[
        Path,
        typer.Option("--run-dir", help="Run history directory. Defaults to .origin/ in cwd."),
    ] = Path(".origin"),
) -> None:
    """List all past diagnostic runs with their status and token usage.

    Example::

        aevyra-origin runs
        aevyra-origin runs --run-dir /path/to/.origin
    """
    from aevyra_origin.run_store import DiagnoseStore

    store = DiagnoseStore(root=run_dir)
    rows = store.list_runs()

    if not rows:
        typer.echo(f"No runs found in {store.runs_dir}")
        return

    # Header
    typer.echo(
        f"{'ID':<6}  {'Status':<12}  {'Method':<14}  {'Score':<7}  "
        f"{'LLM tokens':<12}  {'Abl. calls':<11}  {'Timestamp':<20}  Rubric"
    )
    typer.echo("-" * 110)

    for row in rows:
        status = row["status"]
        status_fmt = (
            typer.style(status, fg=typer.colors.GREEN)
            if status == "completed"
            else typer.style(status, fg=typer.colors.YELLOW)
            if status == "interrupted"
            else status
        )
        score_str = f"{row['score']:.3f}" if row["score"] is not None else "—"
        completed_str = ",".join(row["completed_methods"]) or "—"
        typer.echo(
            f"{row['run_id']:<6}  {status_fmt:<12}  {row['method']:<14}  {score_str:<7}  "
            f"{row['llm_tokens_fmt']:<12}  {row['ablation_calls']:<11}  "
            f"{row['timestamp'][:19]:<20}  {row['rubric_preview']}"
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
