"""Clock protocol shared by backends and the runner.

`exec.scheduler` supplies the WallClock and SimClock implementations.
"""

from __future__ import annotations

from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...

    def sleep_until(self, t: float) -> None: ...
