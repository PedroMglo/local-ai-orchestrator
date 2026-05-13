"""CLI — orc command entry point."""

from __future__ import annotations

import json
import sys
import uuid

import click

from orchestrator.config import get_settings


def _open_session(session_id: str | None):
    """Resolve session: open SessionStore and load history if enabled.

    Returns (store, session_id, history) — store is None when sessions
    are disabled or no ``--session`` flag was provided.
    """
    if session_id is None:
        return None, None, None

    cfg = get_settings()
    if not cfg.session.enabled:
        click.echo(click.style("⚠ Sessions disabled in config", fg="yellow"), err=True)
        return None, None, None

    from orchestrator.core.session import SessionStore

    if session_id == "new":
        session_id = str(uuid.uuid4())

    store = SessionStore(
        db_path=cfg.session.db_path or None,
        max_messages=cfg.session.max_messages,
    )
    history = store.get(session_id) or None

    click.echo(
        click.style(f"Session: {session_id}", fg="yellow"),
        err=True,
    )
    return store, session_id, history


def _persist_session(store, session_id: str, user_msg: str, assistant_msg: str):
    """Save the user/assistant pair and close the store."""
    if store is None:
        return
    try:
        store.append(session_id, "user", user_msg)
        store.append(session_id, "assistant", assistant_msg)
    finally:
        store.close()


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx: click.Context):
    """AI Orchestrator — intelligent query routing."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("query", nargs=-1, required=True)
@click.option("-m", "--model", default=None, help="Override model selection")
@click.option("--json-output", "json_out", is_flag=True, help="Output as JSON")
@click.option("--stream", is_flag=True, help="Stream response")
@click.option("--debug", "debug_mode", is_flag=True, help="Show routing decisions")
@click.option(
    "-s", "--session", "session_id",
    default=None,
    help="Session memory. Use -s new for new session or -s <id> to continue.",
)
def ask(
    query: tuple[str, ...],
    model: str | None,
    json_out: bool,
    stream: bool,
    debug_mode: bool,
    session_id: str | None,
):
    """Ask a question."""
    from orchestrator.factory import create_engine

    question = " ".join(query)
    engine = create_engine()

    store, session_id, history = _open_session(session_id)

    if stream:
        if debug_mode:
            # Emit routing decisions to stderr before streaming
            routing = engine.classify(question, history=history)
            from orchestrator.core.model_router import ConfigModelRouter
            chosen_model = model or ConfigModelRouter().select(routing.intent, routing.complexity)
            from orchestrator.core.context_router import ConfigContextRouter
            sources = ConfigContextRouter().route(routing.intent, routing.complexity)
            click.echo(click.style("── Debug ──", fg="cyan"), err=True)
            if session_id:
                click.echo(f"  Session:    {session_id}", err=True)
                click.echo(f"  History:    {len(history or [])} messages", err=True)
            click.echo(f"  Intent:     {routing.intent.value}", err=True)
            click.echo(f"  Complexity: {routing.complexity.value}", err=True)
            click.echo(f"  Model:      {chosen_model}", err=True)
            click.echo(f"  Sources:    {', '.join(sources) or '(none)'}", err=True)
            click.echo(click.style("────────────", fg="cyan"), err=True)

        chunks: list[str] = []
        for chunk in engine.stream(question, history=history, model_override=model):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            chunks.append(chunk)
        sys.stdout.write("\n")

        _persist_session(store, session_id, question, "".join(chunks)) if store else None
        return

    result = engine.run(question, history=history, model_override=model)

    if debug_mode:
        click.echo(click.style("── Debug ──", fg="cyan"), err=True)
        if session_id:
            click.echo(f"  Session:    {session_id}", err=True)
            click.echo(f"  History:    {len(history or [])} messages", err=True)
        click.echo(f"  Intent:     {result.intent.value}", err=True)
        click.echo(f"  Complexity: {result.complexity.value}", err=True)
        click.echo(f"  Model:      {result.model_used}", err=True)
        click.echo(f"  Sources:    {', '.join(result.sources_used) or '(none)'}", err=True)
        click.echo(f"  Context:    {result.context_tokens} tokens", err=True)
        click.echo(f"  Latency:    {result.latency_ms:.0f}ms", err=True)
        click.echo(click.style("────────────", fg="cyan"), err=True)

    _persist_session(store, session_id, question, result.response) if store else None

    if json_out:
        out = {
            "response": result.response,
            "model_used": result.model_used,
            "intent": result.intent.value,
            "complexity": result.complexity.value,
            "sources_used": result.sources_used,
            "context_tokens": result.context_tokens,
            "latency_ms": round(result.latency_ms, 1),
        }
        if session_id:
            out["session_id"] = session_id
        click.echo(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        click.echo(result.response)


@main.command()
@click.argument("query", nargs=-1, required=True)
def classify(query: tuple[str, ...]):
    """Classify intent and complexity (no LLM call)."""
    from orchestrator.factory import create_engine

    question = " ".join(query)
    engine = create_engine()
    routing = engine.classify(question)

    click.echo(f"Intent:     {routing.intent.value}")
    click.echo(f"Complexity: {routing.complexity.value}")


@main.command()
def serve():
    """Start the orchestrator API server."""
    from orchestrator.api.app import run_server
    run_server()


@main.command()
def health():
    """Check health of all components."""
    from orchestrator.factory import create_engine

    engine = create_engine()
    cfg = get_settings()
    report = engine.health_report()

    click.echo(f"Orchestrator: {cfg.orchestrator.host}:{cfg.orchestrator.port}")
    click.echo(f"Ollama:       {'✓' if report['ollama'] else '✗'}")

    for name, ok in report["providers"].items():
        click.echo(f"  {name:12s} {'✓' if ok else '✗'}")


@main.command()
def config():
    """Show current configuration."""
    cfg = get_settings()
    click.echo("Models:")
    click.echo(f"  default:   {cfg.models.default}")
    click.echo(f"  fast:      {cfg.models.fast}")
    click.echo(f"  code:      {cfg.models.code}")
    click.echo(f"  deep:      {cfg.models.deep}")
    click.echo(f"  embedding: {cfg.models.embedding}")
    click.echo("\nServices:")
    click.echo(f"  Ollama: {cfg.ollama.base_url}")
    click.echo(f"  RAG:    {cfg.rag.url}")
    click.echo(f"  API:    {cfg.orchestrator.host}:{cfg.orchestrator.port}")
    click.echo("\nContext:")
    click.echo(f"  token_budget: {cfg.context.token_budget}")
    click.echo(f"  cag_db:       {cfg.context.cag.db_path or '(not set)'}")
    if cfg.repos.paths:
        click.echo(f"\nRepos ({len(cfg.repos.paths)}):")
        for p in cfg.repos.paths:
            click.echo(f"  - {p}")
