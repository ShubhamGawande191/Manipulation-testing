"""Bench check — verify every hardware component after a reboot or replug.

Camera indices and tty numbers shift across reboots; this is the one command
to run before anything else touches the bench:

    uv run bench-check

Checks USB enumeration (arm CH343, MiniLab 3, eMeet cams), tty nodes + udev
symlinks, camera capture on each /dev/video node, and MIDI input ports.
Writes machine-readable results to config/bench_state.json and exits nonzero
if a critical component is missing.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(add_completion=False)
console = Console()

# What this bench is supposed to have plugged in.
FOLLOWER = ("1a86", "55d3", "5AA9017934")  # WCH CH343 on the SO101 follower board
LEADER = ("1a86", "55d3", "5AA9017169")  # leader board (motor 2 broken, optional)
MINILAB = ("1c75", "020b")  # Arturia MiniLab 3
EMEET = ("328f", "0073")  # eMeet C950, expected x2
FOLLOWER_SYMLINK = "/dev/so101_follower"


@dataclass
class BenchReport:
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    usb: list[dict] = field(default_factory=list)
    serial_ports: list[dict] = field(default_factory=list)
    cameras: list[dict] = field(default_factory=list)
    midi_inputs: list[str] = field(default_factory=list)
    follower_symlink: str | None = None
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def scan_usb() -> list[dict]:
    """Enumerate USB devices from sysfs (no lsusb dependency)."""
    devices = []
    for dev in sorted(Path("/sys/bus/usb/devices").glob("*")):
        vid_f, pid_f = dev / "idVendor", dev / "idProduct"
        if vid_f.exists() and pid_f.exists():
            serial_f = dev / "serial"
            devices.append(
                {
                    "vid": vid_f.read_text().strip(),
                    "pid": pid_f.read_text().strip(),
                    "serial": serial_f.read_text().strip() if serial_f.exists() else None,
                    "product": (dev / "product").read_text().strip()
                    if (dev / "product").exists()
                    else None,
                }
            )
    return devices


def scan_serial_ports() -> list[dict]:
    from serial.tools import list_ports

    return [
        {
            "device": p.device,
            "serial": p.serial_number,
            "vid": f"{p.vid:04x}" if p.vid else None,
            "pid": f"{p.pid:04x}" if p.pid else None,
        }
        for p in list_ports.comports()
    ]


def scan_cameras(max_index: int = 8) -> list[dict]:
    """Probe even /dev/video indices (odd nodes are UVC metadata)."""
    import cv2

    cams = []
    for idx in range(0, max_index, 2):
        if not Path(f"/dev/video{idx}").exists():
            continue
        name_f = Path(f"/sys/class/video4linux/video{idx}/name")
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        ok, frame = cap.read()
        cams.append(
            {
                "index": idx,
                "name": name_f.read_text().strip() if name_f.exists() else "?",
                "captures": bool(ok),
                "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if ok else None,
            }
        )
        cap.release()
    return cams


def scan_midi() -> list[str]:
    try:
        import mido

        return mido.get_input_names()
    except Exception as e:  # rtmidi backend missing or ALSA trouble
        return [f"<error: {e}>"]


def build_report(
    usb: list[dict],
    serial_ports: list[dict],
    cameras: list[dict],
    midi_inputs: list[str],
    follower_symlink_exists: bool,
) -> BenchReport:
    """Pure assembly + diagnosis, unit-testable without hardware."""
    r = BenchReport(
        usb=usb,
        serial_ports=serial_ports,
        cameras=cameras,
        midi_inputs=midi_inputs,
        follower_symlink=FOLLOWER_SYMLINK if follower_symlink_exists else None,
    )

    def usb_has(vid: str, pid: str, serial: str | None = None) -> bool:
        return any(
            d["vid"] == vid and d["pid"] == pid and (serial is None or d["serial"] == serial)
            for d in usb
        )

    if not usb_has(*FOLLOWER):
        r.problems.append(
            f"follower arm board (CH343 {FOLLOWER[0]}:{FOLLOWER[1]} serial {FOLLOWER[2]}) is NOT "
            "enumerating on USB — not a tty-number problem. Check the USB cable, try a direct "
            "port on the machine (not the dock/hub), and check board power."
        )
    elif not follower_symlink_exists:
        r.problems.append(
            f"CH343 enumerated but {FOLLOWER_SYMLINK} missing — udev rule didn't fire; "
            "try `sudo udevadm trigger` or replug."
        )
    if not usb_has(*MINILAB):
        r.problems.append("MiniLab 3 not on USB.")
    elif not any("minilab" in m.lower() for m in midi_inputs):
        r.problems.append("MiniLab 3 on USB but no MIDI input port (rtmidi backend installed?).")
    working_cams = [c for c in cameras if c["captures"]]
    if len(working_cams) < 2:
        r.problems.append(
            f"expected 2 capturing cameras, found {len(working_cams)} "
            f"({[c['index'] for c in working_cams]}). Indices shift on reboot — update "
            "config/so101.yaml if the working indices changed."
        )
    return r


@app.command()
def run(
    write: bool = typer.Option(True, help="write config/bench_state.json"),
    max_camera_index: int = 8,
) -> None:
    report = build_report(
        usb=scan_usb(),
        serial_ports=scan_serial_ports(),
        cameras=scan_cameras(max_camera_index),
        midi_inputs=scan_midi(),
        follower_symlink_exists=Path(FOLLOWER_SYMLINK).exists(),
    )

    table = Table(title="Bench check")
    table.add_column("component")
    table.add_column("status")
    table.add_row(
        "follower arm",
        f"[green]{report.follower_symlink}[/]" if report.follower_symlink else "[red]MISSING[/]",
    )
    for c in report.cameras:
        status = f"[green]{c['resolution']}[/]" if c["captures"] else "[red]no frame[/]"
        table.add_row(f"camera video{c['index']} ({c['name']})", status)
    minilab = [m for m in report.midi_inputs if "minilab" in m.lower()]
    table.add_row("MiniLab 3 MIDI", f"[green]{minilab[0]}[/]" if minilab else "[red]MISSING[/]")
    console.print(table)

    for p in report.problems:
        console.print(f"[red]problem:[/] {p}")

    if write:
        out = Path("config/bench_state.json")
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2) + "\n")
        console.print(f"wrote {out}")

    raise typer.Exit(0 if report.ok else 1)


if __name__ == "__main__":
    app()
