"""Personas simuladas: modelo de comportamiento SCRIPTADO (no aprendido) que mueve entidades del mundo.

Cada persona tiene una pose real (que el simulador fija en Gazebo con set_pose) y un estado:
  parked     fuera de la escena
  standing   de pie, quieta (opcionalmente mirando a un punto)
  walking    caminando por una ruta planificada en la grilla (respeta paredes, puertas y muebles)
  following  sigue a un objetivo (el robot) manteniendo distancia, solo si lo tiene cerca y a la vista
  seated     sentada en un asiento (el modelo baja; ver challenges.yaml)
  leading    camina hacia un destino guiando al robot (anfitrión en la etapa de la bolsa)
Las respuestas de diálogo son ESTRUCTURADAS (nombre, bebida); no hay audio ni reconocimiento de voz.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

from .arena import ArenaMap

WALK_SPEED = 0.45          # m/s (supuesto)
LEAD_WAIT = 1.8           # quien guía espera si el robot queda más lejos (supuesto del escenario)
ROBOT_KEEP = 0.74          # distancia mínima persona-eje del robot al caminar (radio de giro 0.50 + persona 0.22 + 0.02)


@dataclass
class SimPerson:
    name: str
    pose: np.ndarray                     # x, y, yaw (verdad)
    state: str = 'parked'
    path: list = field(default_factory=list)
    target: object = None                # asiento / destino / nombre de lo que sigue
    seat: str | None = None
    persona: dict = field(default_factory=dict)   # nombre y bebida (datos del escenario)
    z: float = 0.0
    waited: float = 0.0                  # tiempo cediendo el paso al robot

    @property
    def xy(self):
        return self.pose[:2]


class PersonPlanner:
    """A* para personas (radio 0.25 m) en la grilla de planificación: pasan por las puertas sin maniobras."""

    def __init__(self, arena: ArenaMap, radius: float = 0.24):
        self.arena = arena
        r = arena.plan_res
        ny, nx = arena.plan_free.shape
        iy, ix = np.mgrid[0:ny, 0:nx]
        X, Y = arena.x0 + (ix + 0.5) * r, arena.y0 + (iy + 0.5) * r
        I = ((Y - arena.y0) // arena.res).astype(int).clip(0, arena.nav_dist.shape[0] - 1)
        J = ((X - arena.x0) // arena.res).astype(int).clip(0, arena.nav_dist.shape[1] - 1)
        self.free = arena.nav_dist[I, J] >= radius

    def plan(self, a, b, blocked=None) -> list | None:
        """blocked: círculos (x, y, r) a evitar (p. ej. el robot)."""
        ar = self.arena
        r = ar.plan_res
        s, g = ar.to_ij(a[0], a[1], r), ar.to_ij(b[0], b[1], r)
        free = self.free
        if blocked:
            free = free.copy()
            iy, ix = np.mgrid[0:free.shape[0], 0:free.shape[1]]
            X, Y = ar.x0 + (ix + 0.5) * r, ar.y0 + (iy + 0.5) * r
            for ox, oy, rad in blocked:
                free &= (X - ox) ** 2 + (Y - oy) ** 2 >= rad * rad
        ny, nx = free.shape
        tail = []
        if not free[g]:
            # objetivo dentro de un mueble (p. ej. el asiento): planificar a la celda libre más cercana y terminar recto
            ii, jj = np.nonzero(free)
            k = int(np.argmin((ii - g[0]) ** 2 + (jj - g[1]) ** 2))
            g = (int(ii[k]), int(jj[k]))
            tail = [tuple(b)]
        h = lambda c: math.hypot(c[0] - g[0], c[1] - g[1])
        openq, cost, parent = [(h(s), 0.0, s)], {s: 0.0}, {s: None}
        while openq:
            _, c, cur = heapq.heappop(openq)
            if cur == g:
                out = []
                while cur is not None:
                    out.append(ar.to_xy(cur[0], cur[1], r))
                    cur = parent[cur]
                pts = out[::-1][::3] + [out[0]]
                return pts + (tail or [tuple(b)])
            if c > cost[cur]:
                continue
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nb = (cur[0] + di, cur[1] + dj)
                if 0 <= nb[0] < ny and 0 <= nb[1] < nx and (free[nb] or nb == s):
                    nc = c + math.hypot(di, dj)
                    if nc < cost.get(nb, 1e18):
                        cost[nb], parent[nb] = nc, cur
                        heapq.heappush(openq, (nc + h(nb), nc, nb))
        return None


class HumanSim:
    def __init__(self, arena: ArenaMap, names: list[str], parking: dict, seats: list[dict], seated_dz: float):
        self.arena = arena
        self.planner = PersonPlanner(arena)
        self.parking = parking
        self.seats = {s['name']: s for s in seats}
        self.seated_dz = seated_dz
        self.people = {n: SimPerson(n, np.array([*parking[n], 0.0])) for n in names}

    # ---------------------------------------------------------------------------------------- órdenes del escenario
    def place(self, name, x, y, yaw=0.0, state='standing'):
        p = self.people[name]
        p.pose, p.state, p.path, p.seat, p.z = np.array([x, y, yaw]), state, [], None, 0.0

    def park(self, name):
        self.place(name, *self.parking[name], state='parked')

    def walk_to(self, name, xy, then='standing', blocked=None):
        p = self.people[name]
        path = self.planner.plan(p.xy, xy, blocked) if blocked else None
        path = path or self.planner.plan(p.xy, xy)     # sin ruta alrededor del robot: ruta libre y cede el paso
        p.path, p.state, p.target, p.seat, p.z = (path or [tuple(xy)]), 'walking', then, None, 0.0

    def follow(self, name, who='robot'):
        p = self.people[name]
        p.state, p.target, p.path, p.seat, p.z = 'following', who, [], None, 0.0

    def walk_and_sit(self, name, seat, blocked=None):
        s = self.seats[seat]
        front = (s['xy'][0] + 0.65 * math.cos(s['yaw']), s['xy'][1] + 0.65 * math.sin(s['yaw']))
        self.walk_to(name, front, blocked=blocked)
        p = self.people[name]
        p.path.append(tuple(s['xy']))
        p.target = ('sit', seat)

    def sit(self, name, seat):
        s = self.seats[seat]
        p = self.people[name]
        p.pose = np.array([s['xy'][0], s['xy'][1], s['yaw']])
        p.state, p.seat, p.z, p.path = 'seated', seat, self.seated_dz, []

    def face(self, name, xy):
        p = self.people[name]
        d = np.asarray(xy) - p.xy
        if np.linalg.norm(d) > 1e-3:
            p.pose[2] = math.atan2(d[1], d[0])

    def lead_to(self, name, xy, blocked=None):
        self.walk_to(name, xy, then='standing', blocked=blocked)
        self.people[name].state = 'leading'

    def occupant(self, seat) -> str | None:
        for p in self.people.values():
            if p.state == 'seated' and p.seat == seat:
                return p.name
        return None

    def free_seats(self) -> list[str]:
        return [s for s in self.seats if self.occupant(s) is None]

    # ---------------------------------------------------------------------------------------- dinámica
    def update(self, dt, robot_ref_xy, robot_visible_to: dict | None = None, follow_dist=1.0, robot_speed=0.0):
        """Avanza dt. robot_ref_xy: posición real del robot. Las personas que siguen al robot lo hacen si está a
        < 3 m; las que guían (leading) esperan si el robot queda a > LEAD_WAIT."""
        for p in self.people.values():
            if p.state in ('walking', 'leading') and p.path:
                if p.state == 'leading' and np.linalg.norm(p.xy - robot_ref_xy) > LEAD_WAIT:
                    continue
                if self._would_approach_robot(p, dt, robot_ref_xy):
                    # cede el paso: espera y, si el robot no se aparta, replanifica rodeándolo
                    p.waited += dt
                    if p.waited > 2.0:
                        p.waited = 0.0
                        alt = self.planner.plan(p.xy, p.path[-1], [(*robot_ref_xy, ROBOT_KEEP + 0.1)])
                        if alt:
                            p.path = alt
                    continue
                p.waited = 0.0
                self._advance_along(p, dt)
                if not p.path:
                    if isinstance(p.target, tuple) and p.target[0] == 'sit':
                        self.sit(p.name, p.target[1])
                    else:
                        p.state = 'standing'
            elif p.state == 'following':
                d = robot_ref_xy - p.xy
                dist = float(np.linalg.norm(d))
                if 3.0 > dist > follow_dist:
                    p.path = [tuple(robot_ref_xy)]
                    self._advance_along(p, dt, stop_at=follow_dist, speed=min(WALK_SPEED, max(robot_speed, 0.25) + 0.1))
                if dist > 0.05:
                    p.pose[2] = math.atan2(d[1], d[0])

    def _would_approach_robot(self, p, dt, robot_xy):
        tgt = np.array(p.path[0])
        d = tgt - p.xy
        n = float(np.linalg.norm(d))
        if n < 1e-6:
            return False
        nxt = p.xy + d / n * min(WALK_SPEED * dt, n)
        now, then = np.linalg.norm(p.xy - robot_xy), np.linalg.norm(nxt - robot_xy)
        return then < ROBOT_KEEP and then < now

    def _advance_along(self, p, dt, stop_at=0.0, speed=WALK_SPEED):
        budget = speed * dt
        while budget > 1e-6 and p.path:
            tgt = np.array(p.path[0])
            d = tgt - p.xy
            dist = float(np.linalg.norm(d))
            if len(p.path) == 1 and dist <= stop_at + 1e-4:
                p.path = [] if stop_at == 0.0 else p.path
                return
            step = min(budget, dist - (stop_at if len(p.path) == 1 else 0.0))
            if step < 1e-6 and dist > 1e-3:
                return                       # nada que avanzar (evita iterar con pasos ínfimos)
            if dist > 1e-6:
                p.pose[:2] = p.xy + d / dist * step
                p.pose[2] = math.atan2(d[1], d[0])
            budget -= step
            if dist - step < 1e-3:
                p.path.pop(0)

    # ---------------------------------------------------------------------------------------- consultas (verdad)
    def head_xyz(self, name):
        p = self.people[name]
        return np.array([p.pose[0], p.pose[1], 1.61 + p.z])

    def hand_xyz(self, name):
        p = self.people[name]
        c, s = math.cos(p.pose[2]), math.sin(p.pose[2])
        return np.array([p.pose[0] + c * 0.28 + s * 0.12, p.pose[1] + s * 0.28 - c * 0.12, 1.00 + p.z])

    def obstacles(self, radius):
        return [(p.pose[0], p.pose[1], radius) for p in self.people.values() if p.state != 'parked']
