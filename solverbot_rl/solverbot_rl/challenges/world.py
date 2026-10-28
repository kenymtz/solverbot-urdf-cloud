"""Mundo de los desafíos: mapa, entidades (personas, puerta de entrada, objetos, prendas) y lugares con nombre."""
from __future__ import annotations

import functools
import math
import os

import numpy as np
import yaml

from ..config import REPO
from .arena import SPEC_PATH, ArenaMap

CHALLENGES_PATH = os.path.join(REPO, 'solverbot_sim', 'config', 'challenges.yaml')


def load_challenges(path: str | None = None) -> dict:
    with open(path or CHALLENGES_PATH) as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=2)
def challenge_arena(path: str | None = None) -> ArenaMap:
    """Mapa con los extras estáticos del mundo home_challenges (cesto). Se construye una vez por proceso (~5 s)."""
    ch = load_challenges(path)
    lb = ch['laundry']['basket']
    extra = [{'name': lb['name'], 'kind': 'basket', 'xy': lb['xy'], 'size': lb['size']}]
    extra += [{'name': d['name'], 'kind': 'door_leaf', 'xy': d['xy'], 'size': d['size']} for d in ch.get('closed_doors', [])]
    return ArenaMap(extra_static=extra)


class ChallengeWorld:
    def __init__(self, path: str | None = None):
        self.ch = load_challenges(path)
        self.arena = challenge_arena(path)
        with open(SPEC_PATH) as fh:
            self.spec = yaml.safe_load(fh)
        self.furniture = {f['name']: f for f in self.spec['furniture']}
        self.objects = {o['name']: o for o in self.spec['objects']}
        self.person_names = [p['name'] for p in self.ch['persons']]
        self.person_rgb = {p['name']: p['shirt_rgb'] for p in self.ch['persons']}
        self.parking = {p: tuple(xy) for p, xy in zip(self.person_names, self.ch['person_parking'])}
        self.seats = self.ch['hri']['seats']
        d = self.ch['entrance_door']
        # la hoja de la puerta en el modelo 2D: cadena de círculos sobre el vano (la colisión exacta la da Gazebo)
        self.door_leaf_circles = [(d['closed_xy'][0], d['closed_xy'][1] + t, 0.06) for t in np.linspace(-0.3, 0.3, 6)]

    # -------------------------------------------------------------------------------------------------- superficies
    def surface_height(self, furniture: str) -> float:
        fu = self.furniture[furniture]
        z = fu['size'][2]
        return 0.03 + (z - 0.04) / 3 if fu['kind'] in ('shelf', 'cabinet_open') else z

    def sample_on_surface(self, furniture: str, rng, margin: float = 0.08) -> tuple[float, float, float]:
        fu = self.furniture[furniture]
        sx, sy = fu['size'][0], fu['size'][1]
        if fu['kind'] == 'round_table':
            r = rng.uniform(0, sx / 2 - margin)
            a = rng.uniform(0, 2 * math.pi)
            return fu['xy'][0] + r * math.cos(a), fu['xy'][1] + r * math.sin(a), self.surface_height(furniture)
        return (fu['xy'][0] + rng.uniform(-sx / 2 + margin, sx / 2 - margin),
                fu['xy'][1] + rng.uniform(-sy / 2 + margin, sy / 2 - margin), self.surface_height(furniture))

    def placeable(self) -> list[str]:
        return [f['name'] for f in self.spec['furniture'] if f.get('placeable') and f['name'] in self.arena.locations]

    def location_xy(self, name: str) -> tuple[float, float, float]:
        loc = self.arena.locations[name]
        return loc.x, loc.y, loc.yaw
