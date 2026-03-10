#!/usr/bin/env python
"""Calibrate the SO101 leader and follower arms.

LeRobot 0.5+ ships ``lerobot-calibrate`` as a standalone CLI.  This script
is a thin wrapper that builds the right arguments and delegates to it, so
you can run it directly via ``uv run calibrate`` without having to remember
the exact flags.

Calibrate follower only (default):
    uv run calibrate --follower-port /dev/ttyUSB1

Calibrate leader only:
    uv run calibrate --device leader --leader-port /dev/ttyUSB0

Calibrate both in sequence:
    uv run calibrate --device both --follower-port /dev/ttyUSB1 --leader-port /dev/ttyUSB0

Calibration files are saved by LeRobot in
~/.cache/huggingface/lerobot/calibration/ keyed by --robot.id / --teleop.id.
"""

import subprocess
import sys

import typer
from rich.console import Console

app = typer.Typer(help="Calibrate SO101 leader / follower arms (wraps lerobot-calibrate).")
console = Console()


@app.command()
def main(
    device: str = typer.Option(
        "follower",
        help="Which device to calibrate: 'follower', 'leader', or 'both'.",
    ),
    follower_port: str = typer.Option("/dev/ttyUSB1", help="Serial port for the follower arm."),
    follower_id: str = typer.Option("so101_follower", help="Unique ID saved with calibration."),
    leader_port: str = typer.Option("/dev/ttyUSB0", help="Serial port for the leader arm."),
    leader_id: str = typer.Option("so101_leader", help="Unique ID saved with calibration."),
) -> None:
    """Run LeRobot's built-in calibration wizard for the SO101 arms."""
    calibrate_cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_calibrate"]

    def _run_follower() -> None:
        console.print("[bold cyan]Calibrating follower arm…[/bold cyan]")
        console.print(f"  Port → [yellow]{follower_port}[/yellow]  ID → [yellow]{follower_id}[/yellow]\n")
        subprocess.run(
            calibrate_cmd
            + [
                "--robot.type=so101_follower",
                f"--robot.port={follower_port}",
                f"--robot.id={follower_id}",
            ],
            check=True,
        )

    def _run_leader() -> None:
        console.print("[bold cyan]Calibrating leader arm…[/bold cyan]")
        console.print(f"  Port → [yellow]{leader_port}[/yellow]  ID → [yellow]{leader_id}[/yellow]\n")
        subprocess.run(
            calibrate_cmd
            + [
                "--teleop.type=so101_leader",
                f"--teleop.port={leader_port}",
                f"--teleop.id={leader_id}",
            ],
            check=True,
        )

    if device == "follower":
        _run_follower()
    elif device == "leader":
        _run_leader()
    elif device == "both":
        _run_follower()
        _run_leader()
    else:
        console.print(f"[red]Unknown device '{device}'. Choose follower, leader, or both.[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
