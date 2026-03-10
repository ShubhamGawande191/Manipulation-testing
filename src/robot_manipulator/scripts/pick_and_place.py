#!/usr/bin/env python
"""Pick-and-place task using OpenClaw agent orchestration + LeRobot.

This script uses OpenClaw to direct the SO101 arm through a high-level
pick-and-place pipeline described in natural language, while LeRobot
handles low-level servo control.

Usage:
    uv run pick-and-place
    # or
    python -m robot_manipulator.scripts.pick_and_place

Environment variables:
    OPENCLAW_API_KEY — your OpenClaw / CMDOP API key (set in .env)
"""

from __future__ import annotations

import os

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(help="Run an OpenClaw-orchestrated pick-and-place task.")
console = Console()

PICK_AND_PLACE_PROMPT = """
You are controlling an SO101 robotic arm (6-DOF, 12V Feetech servos).
Two cameras are available: 'workspace' (overhead) and 'wrist' (end-effector).

Task: Pick up the target object and place it in the target zone.

Steps:
1. Observe workspace camera — identify the object position.
2. Move the arm above the object using joint-space commands.
3. Lower the gripper and close it to grasp the object.
4. Lift the object and move above the target zone.
5. Lower and release by opening the gripper.
6. Return to home position.

Report the outcome after each step.
"""


@app.command()
def main(
    leader_port: str = typer.Option("/dev/ttyUSB0", help="Leader arm port."),
    follower_port: str = typer.Option("/dev/ttyUSB1", help="Follower arm port."),
    calibration_dir: str = typer.Option("config/calibration"),
    workspace_cam: int = typer.Option(0, help="Workspace camera device index."),
    wrist_cam: int = typer.Option(1, help="Wrist camera device index."),
    dry_run: bool = typer.Option(False, help="Print the plan without moving the arm."),
) -> None:
    """Orchestrate a pick-and-place task via OpenClaw."""
    api_key = os.environ.get("OPENCLAW_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENCLAW_API_KEY not set. "
            "Add it to your .env file."
        )
        raise typer.Exit(1)

    from openclaw2 import OpenClaw  # provided by the 'openclaw' PyPI package

    console.print("[bold cyan]Pick-and-Place Task[/bold cyan]")
    console.print(f"  Workspace cam → index [yellow]{workspace_cam}[/yellow]")
    console.print(f"  Wrist cam     → index [yellow]{wrist_cam}[/yellow]")

    if dry_run:
        console.print("\n[dim]Dry-run mode — no arm movement.[/dim]")
        console.print(f"\n[bold]Prompt:[/bold]\n{PICK_AND_PLACE_PROMPT}")
        return

    client = OpenClaw.remote(api_key=api_key)
    results = client.pipeline([PICK_AND_PLACE_PROMPT])
    console.print("\n[bold green]Task complete.[/bold green]")
    console.print(results[-1])


if __name__ == "__main__":
    app()
