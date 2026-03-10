#!/usr/bin/env python
"""Pick-and-place task using OpenClaw (CMDOP) + LeRobot on the SO101 arm.

Two execution modes:
  - **agent** (default): The CMDOP cloud AI plans *and* executes commands via
    the desktop agent's PTY tool.  Requires a running CMDOP desktop agent.
  - **direct**: The CMDOP cloud AI generates a step-by-step plan, then this
    script executes each shell command locally via ``terminal.execute()``.
    More reliable because it doesn't depend on the agent's tool-calling loop.

Prerequisites:
    1. Install + start the CMDOP desktop agent on this machine:
       https://cmdop.com/download
    2. Set OPENCLAW_API_KEY in your .env file (from https://cmdop.com/dashboard)

Usage:
    uv run pick-and-place                         # agent mode (default)
    uv run pick-and-place --mode direct            # direct execution mode
    uv run pick-and-place --dry-run                # print prompt, no connection
    uv run pick-and-place --task "..."             # custom task
"""

from __future__ import annotations

import asyncio
import os
import re
from enum import Enum

import typer
from dotenv import load_dotenv
from rich.console import Console

load_dotenv()

app = typer.Typer(help="Run an OpenClaw-orchestrated manipulation task on the SO101 arm.")
console = Console()


class Mode(str, Enum):
    agent = "agent"
    direct = "direct"


# ── System context ─────────────────────────────────────────────────────────────
_SYSTEM_CONTEXT = """\
You are an AI agent controlling an SO101 6-DOF robotic arm (12 V, Feetech STS3215 servos) \
via the LeRobot Python library.

Hardware on this machine:
  - Follower arm  → /dev/ttyACM0  (calibrated as my_so101_follower)
  - Workspace cam → OpenCV index 0  (eMeet C950)
  - Wrist cam     → OpenCV index 2  (eMeet C950, on end-effector)

Python environment (MUST activate first):
  cd ~/Desktop/RobotManipulator && source .venv/bin/activate

Imports:
  from robot_manipulator.arm import build_follower_config
  from lerobot.robots.so_follower import SOFollower

Workflow:
  1. cfg = build_follower_config("/dev/ttyACM0")
  2. robot = SOFollower(cfg); robot.connect()
  3. obs = robot.get_observation()
  4. robot.send_action({{"shoulder_pan.pos": <deg>, ...}})
  5. robot.disconnect()

Joints: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
Positions in degrees.  Gripper: ~0 = open, ~100 = closed.
"""


def _build_agent_prompt(task: str) -> str:
    """Prompt for agent mode — AI plans AND executes."""
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Task: {task}\n\n"
        "Plan and execute this step-by-step.  "
        "Write and run Python code for each step. Report what you observe and do."
    )


def _build_planner_prompt(task: str) -> str:
    """Prompt for direct mode — AI returns a plan with numbered shell commands."""
    return (
        f"{_SYSTEM_CONTEXT}\n\n"
        f"Task: {task}\n\n"
        "Respond ONLY with a numbered list of shell commands to accomplish this task.  "
        "Each command must be a single-line bash command that can be run directly.\n"
        "Always start with:  cd ~/Desktop/RobotManipulator && source .venv/bin/activate\n"
        "Use `python -c '...'` for inline Python.  "
        "Do NOT include explanations — just the numbered commands."
    )


def _extract_commands(text: str) -> list[str]:
    """Pull numbered commands from the AI response."""
    cmds = []
    for line in text.splitlines():
        # Match lines like "1. command" or "1) command"
        m = re.match(r"^\s*\d+[.)]\s+(.+)$", line)
        if m:
            cmd = m.group(1).strip()
            # Strip markdown backticks if wrapped
            if cmd.startswith("`") and cmd.endswith("`"):
                cmd = cmd[1:-1]
            cmds.append(cmd)
    return cmds


@app.command()
def main(
    task: str = typer.Option(
        "Pick up the object in front of the arm and place it 15 cm to the right.",
        help="Natural-language description of the manipulation task.",
    ),
    mode: Mode = typer.Option(Mode.agent, help="Execution mode: agent or direct."),
    machine: str = typer.Option("beelzebub", help="CMDOP machine hostname to target."),
    timeout: int = typer.Option(300, help="Agent timeout in seconds."),
    dry_run: bool = typer.Option(False, help="Print the prompt without connecting."),
) -> None:
    """Orchestrate a manipulation task on the SO101 arm via OpenClaw."""
    prompt = (
        _build_agent_prompt(task) if mode == Mode.agent else _build_planner_prompt(task)
    )

    console.print("[bold cyan]OpenClaw Manipulation Task[/bold cyan]")
    console.print(f"  Task: [yellow]{task}[/yellow]")
    console.print(f"  Mode: [dim]{mode.value}[/dim]")

    if dry_run:
        console.print("\n[dim]─── dry-run: prompt only ───[/dim]")
        console.print(prompt)
        return

    api_key = os.environ.get("OPENCLAW_API_KEY")
    if not api_key:
        console.print(
            "[bold red]Error:[/bold red] OPENCLAW_API_KEY not set. "
            "Add it to your .env file."
        )
        raise typer.Exit(1)

    if mode == Mode.agent:
        asyncio.run(_run_agent(api_key, prompt, machine, timeout))
    else:
        asyncio.run(_run_direct(api_key, prompt, machine, timeout))


# ── Agent mode: let the AI plan + execute via CMDOP tool-calling ───────────────
async def _run_agent(api_key: str, prompt: str, machine: str, timeout: int) -> None:
    from openclaw import AsyncOpenClaw
    from cmdop.models.agent import AgentType, AgentRunOptions

    async with AsyncOpenClaw.remote(api_key=api_key) as client:
        await client.agent.set_machine(machine)
        opts = AgentRunOptions(timeout_seconds=timeout)
        result = await client.agent.run(
            prompt, agent_type=AgentType.TERMINAL, options=opts,
        )

        if result.text:
            console.print(result.text)

        tool_count = getattr(result, "tool_call_count", 0)
        if tool_count:
            console.print(f"\n[dim]Tools executed: {tool_count}[/dim]")
        else:
            console.print(
                "\n[yellow]Warning:[/yellow] The agent returned a plan but "
                "executed 0 tools.  Try [bold]--mode direct[/bold] instead."
            )

        if not result.success and result.error:
            console.print(f"\n[bold red]Error:[/bold red] {result.error}")

    console.print("\n[bold green]Task complete.[/bold green]")


# ── Direct mode: AI plans, we execute each command via terminal.execute() ──────
async def _run_direct(api_key: str, prompt: str, machine: str, timeout: int) -> None:
    from openclaw import AsyncOpenClaw
    from cmdop.models.agent import AgentType, AgentRunOptions

    async with AsyncOpenClaw.remote(api_key=api_key) as client:
        # Step 1 — ask the AI for a plan (CHAT type = no tool execution)
        console.print("[dim]Asking AI for a plan...[/dim]")
        opts = AgentRunOptions(timeout_seconds=timeout)
        result = await client.agent.run(
            prompt, agent_type=AgentType.CHAT, options=opts,
        )
        if not result.success:
            console.print(f"[bold red]Planning failed:[/bold red] {result.error}")
            return

        plan_text = result.text or ""
        console.print("\n[bold]AI Plan:[/bold]")
        console.print(plan_text)

        commands = _extract_commands(plan_text)
        if not commands:
            console.print("[yellow]No executable commands found in the plan.[/yellow]")
            return

        # Step 2 — execute each command on the machine
        console.print(f"\n[bold cyan]Executing {len(commands)} commands...[/bold cyan]")
        await client.terminal.set_machine(machine)

        for i, cmd in enumerate(commands, 1):
            console.print(f"\n[bold][{i}/{len(commands)}][/bold] {cmd}")
            try:
                output, exit_code = await client.terminal.execute(
                    cmd, timeout=float(timeout),
                )
                decoded = output.decode() if isinstance(output, bytes) else str(output)
                if decoded.strip():
                    console.print(f"[dim]{decoded.strip()}[/dim]")
                if exit_code != 0:
                    console.print(
                        f"[bold red]Exit code {exit_code} — stopping.[/bold red]"
                    )
                    break
            except Exception as exc:
                console.print(f"[bold red]Execution error:[/bold red] {exc}")
                break

    console.print("\n[bold green]Task complete.[/bold green]")


if __name__ == "__main__":
    app()
