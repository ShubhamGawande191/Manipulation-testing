#!/usr/bin/env python
"""Pick-and-place task using OpenClaw (CMDOP) agent orchestration + LeRobot.

OpenClaw connects to a CMDOP agent running on this machine and lets an AI
execute terminal commands to control the SO101 arm via LeRobot.

Prerequisites:
    1. Install + start the CMDOP desktop agent on this machine:
       https://cmdop.com/download
    2. Set OPENCLAW_API_KEY in your .env file (from https://cmdop.com/dashboard)

Usage:
    uv run pick-and-place [OPTIONS]
    uv run pick-and-place --dry-run        # print prompt only, no connection
    uv run pick-and-place --task "..."     # custom task description
"""

from __future__ import annotations

import os

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(help="Run an OpenClaw-orchestrated manipulation task on the SO101 arm.")
console = Console()

# ── System context sent to the AI agent ────────────────────────────────────────
_SYSTEM_CONTEXT = """\
You are an AI agent controlling an SO101 6-DOF robotic arm (12V, Feetech STS3215 servos) via \
the LeRobot Python library.

Hardware on this machine:
  - Follower arm  → /dev/ttyACM0  (calibrated, id=my_so101_follower)
  - Workspace cam → OpenCV index 0  (eMeet C950, mounted left of the arm)
  - Wrist cam     → OpenCV index 2  (eMeet C950, on end-effector)

Python environment:
  - Activate with: cd ~/Desktop/RobotManipulator && source .venv/bin/activate
  - Import arm config: from robot_manipulator.arm import build_follower_config
  - LeRobot follower: from lerobot.robots.so_follower import SOFollower
  - Cameras: from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig

Workflow for arm control:
  1. Build config:  cfg = build_follower_config("/dev/ttyACM0")
  2. Connect:       robot = SOFollower(cfg); robot.connect()
  3. Read state:    obs = robot.get_observation()
  4. Send action:   robot.send_action({"shoulder_pan.pos": <deg>, ...})
  5. Disconnect:    robot.disconnect()

Joint names: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
All positions in degrees. Gripper: ~0 = open, ~100 = closed.
"""


def _build_task_prompt(task: str) -> str:
    return f"{_SYSTEM_CONTEXT}\n\nTask: {task}\n\nPlan and execute this step-by-step. "\
           "Write and run Python code for each step. Report what you observe and do."


@app.command()
def main(
    task: str = typer.Option(
        "Pick up the object in front of the arm and place it 15 cm to the right.",
        help="Natural-language description of the manipulation task.",
    ),
    follower_port: str = typer.Option("/dev/ttyACM0", help="Follower arm serial port."),
    workspace_cam: int = typer.Option(0, help="Workspace camera OpenCV index."),
    wrist_cam: int = typer.Option(2, help="Wrist camera OpenCV index."),
    timeout: int = typer.Option(300, help="Agent timeout in seconds."),
    stream: bool = typer.Option(True, help="Stream agent output token-by-token."),
    dry_run: bool = typer.Option(False, help="Print the prompt without connecting."),
) -> None:
    """Orchestrate a manipulation task on the SO101 arm via OpenClaw."""
    prompt = _build_task_prompt(task)

    console.print("[bold cyan]OpenClaw Manipulation Task[/bold cyan]")
    console.print(f"  Task:          [yellow]{task}[/yellow]")
    console.print(f"  Follower port: [dim]{follower_port}[/dim]")
    console.print(f"  Cameras:       [dim]index {workspace_cam} (workspace), index {wrist_cam} (wrist)[/dim]")

    if dry_run:
        console.print("\n[dim]─── dry-run: prompt only, no CMDOP connection ───[/dim]")
        console.print(prompt)
        return

    api_key = os.environ.get("OPENCLAW_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENCLAW_API_KEY not set. "
            "Add it to your .env file (copy from .env.example)."
        )
        raise typer.Exit(1)

    from openclaw import OpenClaw
    from cmdop.models.agent import AgentType, AgentRunOptions

    console.print("\n[dim]Connecting to CMDOP agent...[/dim]")
    client = OpenClaw.remote(api_key=api_key)

    options = AgentRunOptions(timeout_seconds=timeout)

    if stream:
        console.print("[dim]─── agent output ───[/dim]\n")
        for event in client.agent.stream(prompt, agent_type=AgentType.TERMINAL, options=options):
            if hasattr(event, "text") and event.text:
                console.print(event.text, end="")
        console.print()
    else:
        result = client.agent.run(prompt, agent_type=AgentType.TERMINAL, options=options)
        console.print(result.text)

    console.print("\n[bold green]Task complete.[/bold green]")


if __name__ == "__main__":
    app()
