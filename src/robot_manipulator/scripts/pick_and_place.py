#!/usr/bin/env python
"""Control the SO101 arm via the OpenClaw (CMDOP) skills system.

The ``so101`` skill teaches the CMDOP agent how to operate the arm through
LeRobot.  The agent receives the user's natural-language task, plans the
steps, and executes shell commands on this machine via ``cmd_execute``.

Prerequisites:
    1. Install + start the CMDOP desktop agent: https://cmdop.com/download
    2. Set OPENCLAW_API_KEY in your .env file
    3. Install the skill:
       cp skills/so101.md ~/.local/share/cmdop/skills/so101.md

Usage:
    uv run pick-and-place                          # default task
    uv run pick-and-place --task "wave the arm"    # custom task
    uv run pick-and-place --dry-run                # print prompt only
"""

from __future__ import annotations

import asyncio
import os

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(help="Run an OpenClaw-orchestrated manipulation task on the SO101 arm.")
console = Console()

SKILL_NAME = "so101"


@app.command()
def main(
    task: str = typer.Option(
        "Pick up the object in front of the arm and place it 15 cm to the right.",
        help="Natural-language description of the manipulation task.",
    ),
    machine: str = typer.Option("beelzebub", help="CMDOP machine hostname to target."),
    timeout: int = typer.Option(300, help="Agent timeout in seconds."),
    dry_run: bool = typer.Option(False, help="Print the task without connecting."),
) -> None:
    """Orchestrate a manipulation task on the SO101 arm via the so101 skill."""
    console.print("[bold cyan]OpenClaw SO101 Task[/bold cyan]")
    console.print(f"  Task:    [yellow]{task}[/yellow]")
    console.print(f"  Skill:   [dim]{SKILL_NAME}[/dim]")
    console.print(f"  Machine: [dim]{machine}[/dim]")

    if dry_run:
        console.print(f"\n[dim]─── dry-run ───[/dim]\nTask: {task}")
        return

    api_key = os.environ.get("OPENCLAW_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENCLAW_API_KEY not set. Add it to your .env file."
        )
        raise typer.Exit(1)

    asyncio.run(_run_skill(api_key, task, machine, timeout))


async def _run_skill(
    api_key: str,
    task: str,
    machine: str,
    timeout: int,
) -> None:
    from cmdop.models.agent import AgentRunOptions
    from openclaw import AsyncOpenClaw

    try:
        async with AsyncOpenClaw.remote(api_key=api_key) as client:
            await client.skills.set_machine(machine)

            opts = AgentRunOptions(timeout_seconds=timeout)
            console.print("[dim]Running skill...[/dim]\n")
            result = await client.skills.run(SKILL_NAME, task, options=opts)

            if result.text:
                console.print(result.text)

            # Show tool execution summary
            tool_results = getattr(result, "tool_results", []) or []
            if tool_results:
                console.print(f"\n[dim]Tools executed: {len(tool_results)}[/dim]")
                for tr in tool_results:
                    status = "[green]OK[/green]" if tr.success else "[red]FAIL[/red]"
                    console.print(f"  {status} {tr.tool_name} ({tr.duration_ms}ms)")
            else:
                console.print("\n[yellow]No tools were executed.[/yellow]")

            if not result.success:
                console.print(
                    f"\n[bold red]Error:[/bold red] {getattr(result, 'error', 'unknown')}"
                )
    except Exception as exc:
        console.print(f"\n[bold red]Connection error:[/bold red] {exc}")
        console.print("[dim]Ensure the CMDOP desktop agent is running and connected.[/dim]")
        raise typer.Exit(1)

    console.print("\n[bold green]Task complete.[/bold green]")


if __name__ == "__main__":
    app()
