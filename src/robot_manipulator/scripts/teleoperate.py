#!/usr/bin/env python
"""Teleoperate the SO101 follower arm using the leader arm.

LeRobot 0.5+ ships ``lerobot-teleoperate`` as a standalone CLI.  This script
wraps it so you can run ``uv run teleoperate`` without memorising every flag.

Basic usage (no cameras):
    uv run teleoperate \
        --follower-port /dev/ttyUSB1 \
        --leader-port /dev/ttyUSB0

With cameras + live visualisation in Rerun:
    uv run teleoperate \
        --follower-port /dev/ttyUSB1 \
        --leader-port /dev/ttyUSB0 \
        --workspace-cam 0 \
        --wrist-cam 1 \
        --display

Press Ctrl+C to stop.
"""

import subprocess
import sys

import typer
from rich.console import Console

app = typer.Typer(help="Teleoperate SO101 follower from leader arm (wraps lerobot-teleoperate).")
console = Console()


@app.command()
def main(
    follower_port: str = typer.Option("/dev/ttyUSB1", help="Serial port for the follower arm."),
    follower_id: str = typer.Option("so101_follower", help="Robot ID (must match calibration)."),
    leader_port: str = typer.Option("/dev/ttyUSB0", help="Serial port for the leader arm."),
    leader_id: str = typer.Option("so101_leader", help="Teleop ID (must match calibration)."),
    workspace_cam: int = typer.Option(-1, help="Workspace camera device index (-1 to skip)."),
    wrist_cam: int = typer.Option(-1, help="Wrist camera device index (-1 to skip)."),
    display: bool = typer.Option(False, help="Stream data to Rerun visualiser."),
) -> None:
    """Mirror leader arm movements to the follower in real-time."""
    console.print("[bold cyan]SO101 Teleoperation[/bold cyan]")
    console.print(f"  Follower → [yellow]{follower_port}[/yellow] (id={follower_id})")
    console.print(f"  Leader   → [yellow]{leader_port}[/yellow] (id={leader_id})")
    console.print("Press [bold red]Ctrl+C[/bold red] to stop.\n")

    # Build the inline camera dict that lerobot-teleoperate accepts
    cam_parts: list[str] = []
    if workspace_cam >= 0:
        cam_parts.append(
            f"workspace: {{type: opencv, index_or_path: {workspace_cam}, width: 1920, height: 1080, fps: 30}}"
        )
    if wrist_cam >= 0:
        cam_parts.append(
            f"wrist: {{type: opencv, index_or_path: {wrist_cam}, width: 1920, height: 1080, fps: 30}}"
        )

    cmd = [
        sys.executable,
        "-m",
        "lerobot.scripts.lerobot_teleoperate",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_id}",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_id}",
        f"--display_data={'true' if display else 'false'}",
    ]
    if cam_parts:
        cmd.append(f"--robot.cameras={{{{ {', '.join(cam_parts)} }}}}")

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    app()
