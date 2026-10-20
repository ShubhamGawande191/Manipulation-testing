"""Learned IK seed: an MLP from (fingertip xyz, pitch) to joints, refined by DLS (the hybrid).

Inverse kinematics is multi-valued (elbow up/down, facing/reaching back), and a plain regression
on random (q -> task) pairs averages the branches into nonsense. The labels are therefore the
analytic solution picked with one fixed preference (closest to HOME), which makes the target a
function. The network alone is a fast but approximate guess; a few DLS steps finish it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from robot_manipulator.ik.analytic import AnalyticIK
from robot_manipulator.sim.envs.reach import HOME_ARM

HOME = np.r_[HOME_ARM, 0.0]


def make_dataset(ik: AnalyticIK, n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(X: n x 4 [x, y, z, pitch], Y: n x 4 [pan, lift, elbow, wflex]) with consistent branches."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    while len(X) < n:
        q = np.r_[rng.uniform(ik.lo[:5] + 0.05, ik.hi[:5] - 0.05), 0.0]
        t = ik.task(q)
        lab = ik.solve(t[:3], t[3], seed=HOME)  # the branch the preference picks, not q itself
        if lab is None:
            continue
        X.append(t)
        Y.append(lab[:4])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)


class LearnedSeed:
    def __init__(self, path: Path | None = None) -> None:
        import torch

        self.torch = torch
        self.net = torch.nn.Sequential(
            torch.nn.Linear(4, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 4),
        )
        self.mu = np.zeros(4, np.float32)
        self.sd = np.ones(4, np.float32)
        if path is not None:
            state = torch.load(path, weights_only=False)
            self.net.load_state_dict(state["net"])
            self.mu, self.sd = state["mu"], state["sd"]
        self.net.eval()

    def fit(self, X, Y, epochs: int = 60, batch: int = 1024, lr: float = 2e-3, log=print):
        torch = self.torch
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        x = torch.from_numpy((X - self.mu) / self.sd)
        y = torch.from_numpy(Y)
        opt = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        self.net.train()
        t0 = time.monotonic()
        for ep in range(epochs):
            perm = torch.randperm(len(x))
            total = 0.0
            for i in range(0, len(x), batch):
                idx = perm[i : i + batch]
                loss = torch.nn.functional.smooth_l1_loss(self.net(x[idx]), y[idx], beta=0.01)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item() * len(idx)
            sched.step()
            if ep % 10 == 0 or ep == epochs - 1:
                log(f"epoch {ep:3d} loss {total / len(x):.5f} ({time.monotonic() - t0:.0f}s)")
        self.net.eval()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({"net": self.net.state_dict(), "mu": self.mu, "sd": self.sd}, path)

    def __call__(self, p, phi: float, roll: float = 0.0) -> np.ndarray:
        x = (np.r_[p, phi].astype(np.float32) - self.mu) / self.sd
        with self.torch.no_grad():
            y = self.net(self.torch.from_numpy(x[None]))[0].numpy()
        return np.r_[y.astype(float), roll, 0.0]
