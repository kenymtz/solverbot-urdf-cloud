"""Compromiso de rama de muñeca en tiempo de ejecución (EXPERIMENTAL, desactivado por defecto).

La muñeca del Piper es redundante y el experto elige entre dos ramas de joint4 de forma arbitraria; la regresión por
MSE del BC promedia ramas y la política puede cambiar de rama a mitad del alcance (joint4 de −1 a +1), inclinando la
pinza hasta la horizontal. Este guardián NO usa información privilegiada ni reentrena: observa solo la articulación
joint4 (propiocepción) y, cuando la política ya se comprometió con una rama (joint4 se alejó `commit` rad de su valor
inicial), limita la acción de joint4 que empuja hacia la otra rama a `max_reverse`. Registra cada intervención.
"""
from __future__ import annotations

import numpy as np

J4 = 3                                   # índice de joint4 en q / acciones


class BranchCommitGuard:
    def __init__(self, lo, hi, commit: float = 0.15, max_reverse: float = 0.2):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)
        self.commit, self.max_reverse = commit, max_reverse
        self.reset()

    def reset(self):
        self.q4_start = None
        self.branch = 0
        self.commit_step = None
        self.interventions = 0
        self.steps = 0

    def q4(self, obs):
        return float((obs[J4] + 1) / 2 * (self.hi[J4] - self.lo[J4]) + self.lo[J4])

    def __call__(self, obs, action):
        a = np.array(action, np.float32, copy=True)
        q4 = self.q4(obs)
        if self.q4_start is None:
            self.q4_start = q4
        if self.branch == 0 and abs(q4 - self.q4_start) > self.commit:
            self.branch = int(np.sign(q4 - self.q4_start))
            self.commit_step = self.steps
        if self.branch and a[J4] * self.branch < -self.max_reverse:
            a[J4] = -self.max_reverse * self.branch
            self.interventions += 1
        self.steps += 1
        return a

    def summary(self):
        return {'branch': self.branch, 'commit_step': self.commit_step, 'interventions': self.interventions}
