"""Curriculum por TASA DE ÉXITO real (no por recompensa) y registro de episodios en JSONL."""
from __future__ import annotations

import json
import os
import time
from collections import deque

import gymnasium as gym


class EpisodeLogger(gym.Wrapper):
    """Escribe un JSON por episodio (retorno, longitud, éxito, nivel, términos de recompensa, semilla)."""

    def __init__(self, env, path: str, run_id: str = ''):
        super().__init__(env)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.run_id, self.episode, self.seed = path, run_id, 0, None

    def reset(self, **kw):
        self.seed = kw.get('seed')
        return self.env.reset(**kw)

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        if 'episode_summary' in info:
            rec = {'run': self.run_id, 'episode': self.episode, 'seed': self.seed, 'time': time.time(),
                   'object_source': info.get('object_source'), **info['episode_summary']}
            with open(self.path, 'a') as fh:
                fh.write(json.dumps(rec) + '\n')
            self.episode += 1
        return obs, rew, term, trunc, info


class Curriculum:
    """Sube de nivel cuando la tasa de éxito de las últimas `window` episodios del nivel actual >= promote_at."""

    def __init__(self, n_levels: int, window: int, promote_at: float, level: int = 0):
        self.n_levels, self.window, self.promote_at, self.level = n_levels, window, promote_at, level
        self.hist: deque = deque(maxlen=window)

    def record(self, success: bool) -> bool:
        """Registra un episodio; devuelve True si se promovió de nivel."""
        self.hist.append(bool(success))
        if (len(self.hist) == self.window and sum(self.hist) / self.window >= self.promote_at
                and self.level < self.n_levels - 1):
            self.level += 1
            self.hist.clear()
            return True
        return False

    @property
    def success_rate(self) -> float:
        return sum(self.hist) / len(self.hist) if self.hist else 0.0
