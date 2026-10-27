"""Mapa 2D de la arena doméstica derivado de la MISMA geometría que genera el mundo de Gazebo.

Fuente: solverbot_sim/config/arena_spec.yaml + solverbot_sim/tools/make_worlds.py (primitivas de colisión). Así la
grilla de navegación, el LiDAR simulado del backend cinemático y el SDF de Gazebo no pueden divergir.

* nav_occ:   celdas ocupadas por cualquier primitiva con colisión entre z = 0.02 y 1.40 m (altura del robot).
* lidar_occ: celdas ocupadas en el plano del LiDAR (z = 0.364 m): patas de mesa, columnas, bases de muebles.
* Planificador A* sobre una grilla inflada por la semi-anchura del robot; las puertas (vano de 0.70 m contra un robot
  de 0.669 m con el Piper plegado) se cruzan con una maniobra explícita: punto previo alineado con el eje de la
  puerta, giro en el lugar y avance recto hasta el punto posterior.
"""
from __future__ import annotations

import heapq
import importlib.util
import math
import os
from dataclasses import dataclass, field

import numpy as np
import yaml

from ..config import REPO

SPEC_PATH = os.path.join(REPO, 'solverbot_sim', 'config', 'arena_spec.yaml')
CHALLENGES_PATH = os.path.join(REPO, 'solverbot_sim', 'config', 'challenges.yaml')

# Huella del robot en el frame base_footprint con el Piper en DEFAULT_HOME y elevador en 0 (medida sobre las mallas de
# colisión del URDF): x [-0.262, 0.283], y [-0.306, 0.363]. Ancho 0.669 m.
FOOTPRINT = (-0.262, 0.283, -0.306, 0.363)
# Punto de referencia del planificador y del controlador: el CENTRO DEL EJE DE LAS RUEDAS (x 0.084, URDF), que es el
# centro de giro real de la base diferencial y no se desplaza al girar en el lugar (medido en Gazebo: girar 90°
# desplaza base_footprint 129 mm, radio ~9 cm).
AXLE = np.array([0.084, 0.0025])
FOOT_CENTER = AXLE                     # nombre histórico: el punto que se planifica y controla
LEFT_EXT = FOOTPRINT[3] - AXLE[1]      # 0.3605: extensión lateral izquierda desde el eje
RIGHT_EXT = AXLE[1] - FOOTPRINT[2]     # 0.3085
HALF_WIDTH = max(LEFT_EXT, RIGHT_EXT)  # holgura lateral mínima del punto de referencia
LANE_OFFSET = (RIGHT_EXT - LEFT_EXT) / 2   # desplazamiento lateral del eje en una puerta para centrar la huella
# Radio barrido al girar en el lugar (alrededor del eje de las ruedas): esquina más lejana de la huella
CIRCUMRADIUS = max(math.hypot(x - AXLE[0], y - AXLE[1]) for x in FOOTPRINT[:2] for y in FOOTPRINT[2:])
DOOR_STANDOFF = 0.62          # distancia del eje de la pared a los puntos previo/posterior de un cruce de puerta
LASER_XY = np.array([0.2429, 0.0082])
LASER_Z = 0.364
ROBOT_HEIGHT = 1.40


def _load_make_worlds():
    path = os.path.join(REPO, 'solverbot_sim', 'tools', 'make_worlds.py')
    spec = importlib.util.spec_from_file_location('solverbot_make_worlds', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Prim:
    """Primitiva de colisión en el mundo: caja (sx, sy, yaw) o cilindro (radio)."""
    name: str
    shape: str
    cx: float
    cy: float
    z0: float
    z1: float
    sx: float = 0.0
    sy: float = 0.0
    yaw: float = 0.0
    r: float = 0.0


@dataclass
class Door:
    name: str
    center: np.ndarray            # punto medio del vano (sobre el eje de la pared)
    normal: np.ndarray            # dirección de cruce (unitaria)
    width: float
    rooms: tuple = ()
    usable: bool = True
    standoff: tuple = (0.62, 0.62)       # distancia del eje de la pared a los puntos previo/posterior (lado -n, lado +n)


@dataclass
class Waypoint:
    x: float
    y: float
    kind: str = 'free'            # free | door_pre | door_post | goal
    heading: float | None = None
    door: str | None = None


@dataclass
class Location:
    name: str
    x: float
    y: float
    yaw: float
    room: str | None
    kind: str = 'room'            # room | furniture | special
    target: str | None = None     # mueble asociado


def collision_prims(spec: dict, extra_static: list[dict] | None = None) -> list[Prim]:
    """Primitivas con colisión del mundo (paredes, mobiliario, sillas y extras estáticos de challenges.yaml)."""
    mw = _load_make_worlds()
    prims: list[Prim] = []

    def add_parts(owner, ox, oy, oyaw, parts):
        c, s = math.cos(oyaw), math.sin(oyaw)
        for p in parts:
            if not p.collide:
                continue
            px, py, pz = p.xyz
            wx, wy = ox + c * px - s * py, oy + s * px + c * py
            if p.shape == 'box':
                prims.append(Prim(f'{owner}/{p.name}', 'box', wx, wy, pz - p.size[2] / 2, pz + p.size[2] / 2,
                                  sx=p.size[0], sy=p.size[1], yaw=oyaw + p.rpy[2]))
            elif p.shape == 'cylinder' and abs(p.rpy[0]) < 1e-6 and abs(p.rpy[1]) < 1e-6:
                prims.append(Prim(f'{owner}/{p.name}', 'cyl', wx, wy, pz - p.size[2] / 2, pz + p.size[2] / 2,
                                  r=p.size[0] / 2))
            elif p.shape == 'sphere':
                prims.append(Prim(f'{owner}/{p.name}', 'cyl', wx, wy, pz - p.size[0] / 2, pz + p.size[0] / 2,
                                  r=p.size[0] / 2))

    # paredes: reconstruidas con el mismo generador (el modelo house_walls)
    walls_sdf = mw.walls(spec)
    del walls_sdf
    h = spec['house']
    W, L = h['size']
    t, hh, dh = h['wall_thickness'], h['wall_height'], h['door_height']
    ix, iy = h['interior_wall_x'], h['interior_wall_y']
    defs = {'south': ('y', t / 2, 0, W), 'north': ('y', L - t / 2, 0, W), 'west': ('x', t / 2, 0, L),
            'east': ('x', W - t / 2, 0, L), 'interior_x': ('x', ix, t, L - t), 'interior_y': ('y', iy, t, W - t)}
    for wname, (axis, c0, a, b) in defs.items():
        gaps = sorted((o['center'] - o['width'] / 2, o['center'] + o['width'] / 2)
                      for o in spec['openings'] if o['wall'] == wname)
        cur = a
        for k, (g0, g1) in enumerate(gaps + [(b, b)]):
            if g0 - cur > 1e-3:
                mid, ln = (cur + g0) / 2, g0 - cur
                if axis == 'x':
                    prims.append(Prim(f'wall/{wname}_{k}', 'box', c0, mid, 0.0, hh, sx=t, sy=ln))
                else:
                    prims.append(Prim(f'wall/{wname}_{k}', 'box', mid, c0, 0.0, hh, sx=ln, sy=t))
            cur = max(cur, g1)
    for fu in spec['furniture']:
        add_parts(fu['name'], fu['xy'][0], fu['xy'][1], fu.get('yaw', 0.0), mw.furniture_parts(fu['kind'], fu['size']))
        if fu['kind'] == 'round_table':
            for name, xy, yaw, parts in mw.table_chairs(fu):
                add_parts(name, xy[0], xy[1], yaw, parts)
    for ex in extra_static or []:
        add_parts(ex['name'], ex['xy'][0], ex['xy'][1], ex.get('yaw', 0.0), mw.furniture_parts(ex['kind'], ex['size']))
    return prims


class ArenaMap:
    def __init__(self, spec: dict | None = None, res: float = 0.02, plan_res: float = 0.05,
                 extra_static: list[dict] | None = None, bounds=(-2.5, 7.9, -0.8, 7.9)):
        if spec is None:
            with open(SPEC_PATH) as fh:
                spec = yaml.safe_load(fh)
        self.spec = spec
        self.res, self.plan_res = res, plan_res
        self.x0, self.x1, self.y0, self.y1 = bounds
        self.prims = collision_prims(spec, extra_static)
        self.nav_occ = self._raster(res, lambda p: p.z1 > 0.02 and p.z0 < ROBOT_HEIGHT)
        self.lidar_occ = self._raster(res, lambda p: p.z0 <= LASER_Z <= p.z1)
        from scipy.ndimage import distance_transform_edt
        self.nav_dist = distance_transform_edt(~self.nav_occ) * res
        self.lidar_dist = distance_transform_edt(~self.lidar_occ) * res
        self.doors = self._doors()
        self.rooms = {k: (np.array(v['min']), np.array(v['max'])) for k, v in spec['rooms'].items()}
        self._plan_grid()
        self.locations = self._locations()

    # ------------------------------------------------------------------------------------------- grillas
    def _shape(self, res):
        return int(math.ceil((self.y1 - self.y0) / res)), int(math.ceil((self.x1 - self.x0) / res))

    def _raster(self, res, keep) -> np.ndarray:
        ny, nx = self._shape(res)
        occ = np.zeros((ny, nx), bool)
        for p in self.prims:
            if not keep(p):
                continue
            ext = (max(p.sx, p.sy) * 0.75 + 0.05) if p.shape == 'box' else p.r + 0.05
            i0, i1 = self.to_ij(p.cx - ext, p.cy - ext, res), self.to_ij(p.cx + ext, p.cy + ext, res)
            js = np.arange(max(0, i0[1]), min(nx, i1[1] + 1))
            iis = np.arange(max(0, i0[0]), min(ny, i1[0] + 1))
            if js.size == 0 or iis.size == 0:
                continue
            X, Y = np.meshgrid(self.x0 + (js + 0.5) * res, self.y0 + (iis + 0.5) * res)
            if p.shape == 'box':
                c, s = math.cos(-p.yaw), math.sin(-p.yaw)
                lx, ly = c * (X - p.cx) - s * (Y - p.cy), s * (X - p.cx) + c * (Y - p.cy)
                inside = (np.abs(lx) <= p.sx / 2) & (np.abs(ly) <= p.sy / 2)
            else:
                inside = (X - p.cx) ** 2 + (Y - p.cy) ** 2 <= p.r ** 2
            occ[iis[0]:iis[-1] + 1, js[0]:js[-1] + 1] |= inside
        return occ

    def to_ij(self, x, y, res=None):
        res = res or self.res
        return int((y - self.y0) // res), int((x - self.x0) // res)

    def to_xy(self, i, j, res=None):
        res = res or self.res
        return self.x0 + (j + 0.5) * res, self.y0 + (i + 0.5) * res

    def inside_bounds(self, x, y):
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1

    def clearance(self, x, y, lidar=False) -> float:
        """Distancia (m) al obstáculo más cercano en la grilla de navegación (o del plano del LiDAR)."""
        if not self.inside_bounds(x, y):
            return 0.0
        i, j = self.to_ij(x, y)
        return float((self.lidar_dist if lidar else self.nav_dist)[i, j])

    # ------------------------------------------------------------------------------------------- huella del robot
    def footprint_points(self, x, y, yaw, step=0.04, margin=0.0):
        xs = np.arange(FOOTPRINT[0] - margin, FOOTPRINT[1] + margin + 1e-9, step)
        ys = np.arange(FOOTPRINT[2] - margin, FOOTPRINT[3] + margin + 1e-9, step)
        X, Y = np.meshgrid(xs, ys)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([x + c * X - s * Y, y + s * X + c * Y], -1).reshape(-1, 2)

    def footprint_collides(self, x, y, yaw, margin=0.0, dynamic: list | None = None) -> bool:
        """Colisión de la huella REAL (sin inflar) con la geometría estática (y obstáculos dinámicos (x, y, r))."""
        pts = self.footprint_points(x, y, yaw, margin=margin)
        ii = ((pts[:, 1] - self.y0) // self.res).astype(int)
        jj = ((pts[:, 0] - self.x0) // self.res).astype(int)
        ny, nx = self.nav_occ.shape
        if np.any((ii < 0) | (jj < 0) | (ii >= ny) | (jj >= nx)):
            return True
        if np.any(self.nav_occ[ii, jj]):
            return True
        for ox, oy, r in dynamic or []:
            if np.any((pts[:, 0] - ox) ** 2 + (pts[:, 1] - oy) ** 2 < r * r):
                return True
        return False

    # ------------------------------------------------------------------------------------------- puertas y habitaciones
    def _doors(self) -> list[Door]:
        h = self.spec['house']
        t = h['wall_thickness']
        W, L = h['size']
        axes = {'west': ('x', t / 2), 'east': ('x', W - t / 2), 'south': ('y', t / 2), 'north': ('y', L - t / 2),
                'interior_x': ('x', h['interior_wall_x']), 'interior_y': ('y', h['interior_wall_y'])}
        out = []
        for o in self.spec['openings']:
            axis, c = axes[o['wall']]
            if axis == 'x':
                center, normal = np.array([c, o['center']]), np.array([1.0, 0.0])
            else:
                center, normal = np.array([o['center'], c]), np.array([0.0, 1.0])
            rooms = (self.room_of(*(center - normal * 0.4)), self.room_of(*(center + normal * 0.4)))
            out.append(Door(o['name'], center, normal, o['width'], rooms))
        return out

    def room_of(self, x, y) -> str | None:
        for name, (lo, hi) in getattr(self, 'rooms', {}).items() or {
                k: (np.array(v['min']), np.array(v['max'])) for k, v in self.spec['rooms'].items()}.items():
            if lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]:
                return name
        return None if (0 <= x <= self.spec['house']['size'][0] and 0 <= y <= self.spec['house']['size'][1]) \
            else 'outside'

    # ------------------------------------------------------------------------------------------- planificación
    def _plan_grid(self):
        """Grilla de planificación para el CENTRO de la huella: libre si la holgura >= semi-anchura + 1 cm.
        Las puertas se tallan como un pasillo de una celda sobre su eje (el cruce lo resuelve la maniobra)."""
        r = self.plan_res
        ny, nx = self._shape(r)
        free = np.zeros((ny, nx), bool)
        iy, ix = np.mgrid[0:ny, 0:nx]
        X, Y = self.x0 + (ix + 0.5) * r, self.y0 + (iy + 0.5) * r
        I = ((Y - self.y0) // self.res).astype(int).clip(0, self.nav_dist.shape[0] - 1)
        J = ((X - self.x0) // self.res).astype(int).clip(0, self.nav_dist.shape[1] - 1)
        clear = self.nav_dist[I, J]
        free = clear >= HALF_WIDTH + 0.03
        # costo extra por celda cerca de obstáculos: rutas por el centro de los pasillos, con espacio para girar
        self.clear_cost = np.clip(CIRCUMRADIUS + 0.15 - clear, 0.0, None) * 8.0
        self.corridor = np.zeros_like(free)      # solo informativo: el cruce es una arista explícita
        for d in self.doors:
            for t_ in np.arange(-(DOOR_STANDOFF - r), DOOR_STANDOFF - r + 1e-9, r / 2):
                p = d.center + d.normal * t_
                self.corridor[self.to_ij(p[0], p[1], r)] = True
        self.plan_free = free & ~self.corridor
        self.rot_free = self.nav_dist[I, J] >= CIRCUMRADIUS + 0.01    # donde se puede girar en el lugar
        # Aristas de puerta: punto previo <-> punto posterior sobre el eje de la puerta. Cada uno es el punto girable más
        # cercano (holgura >= radio circunscrito) desde el que el carril recto, con la huella REAL orientada al eje,
        # está libre hasta el otro lado. El cruce es un tramo recto entre ambos.
        self.door_edges = {}
        for d in self.doors:
            ends = []
            for sgn in (-1.0, 1.0):
                found = None
                for st in np.arange(DOOR_STANDOFF, 2.0, 0.05):
                    p = d.center + d.normal * sgn * st
                    if not self.inside_bounds(*p):
                        break
                    if self._sweep_free(p, math.atan2(-sgn * d.normal[1], -sgn * d.normal[0])):
                        found = st
                        break
                ends.append(found)
            d.standoff = tuple(ends)
            d.usable = None not in ends and self._lane_free(d)
            if d.usable:
                a = d.center - d.normal * ends[0]
                b = d.center + d.normal * ends[1]
                ia, ib = self.to_ij(a[0], a[1], r), self.to_ij(b[0], b[1], r)
                self.door_edges.setdefault(ia, []).append((ib, d))
                self.door_edges.setdefault(ib, []).append((ia, d))

    @staticmethod
    def base_from_ref(ref, yaw):
        """Origen de base_footprint dado el punto de referencia (eje de ruedas) y el rumbo."""
        c, s_ = math.cos(yaw), math.sin(yaw)
        return ref[0] - (c * AXLE[0] - s_ * AXLE[1]), ref[1] - (s_ * AXLE[0] + c * AXLE[1])

    def _sweep_free(self, point, heading, span=math.radians(100), step=math.radians(5)) -> bool:
        """Giro en el lugar (alrededor del eje de las ruedas, fijo) entre heading±span en un extremo de un cruce, para
        ambos sentidos de cruce (heading y heading+pi), con el eje desplazado lateralmente como durante el cruce."""
        for h0 in (heading, heading + math.pi):
            n = np.array([math.cos(h0), math.sin(h0)])
            ref = point + np.array([-n[1], n[0]]) * LANE_OFFSET
            for yaw in np.arange(h0 - span, h0 + span + 1e-9, step):
                bx, by = self.base_from_ref(ref, yaw)
                if self.footprint_collides(bx, by, yaw, margin=0.01):
                    return False
        return True

    def _lane_free(self, d) -> bool:
        """Carril recto del cruce libre para la huella real orientada al eje (en ambos sentidos)."""
        for sgn in (1.0, -1.0):
            n = d.normal * sgn
            yaw = math.atan2(n[1], n[0])
            s0, s1 = (d.standoff[0], d.standoff[1]) if sgn > 0 else (d.standoff[1], d.standoff[0])
            for t_ in np.arange(-s0, s1 + 1e-9, 0.04):
                ref = d.center + n * t_ + np.array([-n[1], n[0]]) * LANE_OFFSET
                bx, by = self.base_from_ref(ref, yaw)
                if self.footprint_collides(bx, by, yaw):
                    return False
        return True

    def _nearest_free(self, x, y, need_rot=False, max_r=1.0):
        r = self.plan_res
        grid = self.rot_free if need_rot else (self.plan_free & ~self.corridor)
        i0, j0 = self.to_ij(x, y, r)
        best = None
        rr = int(max_r / r)
        for di in range(-rr, rr + 1):
            for dj in range(-rr, rr + 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j]:
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, i, j)
        return None if best is None else self.to_xy(best[1], best[2], r)

    def plan(self, start, goal, dynamic: list | None = None) -> list[Waypoint] | None:
        """Ruta para el CENTRO de la huella desde start (x, y) hasta goal (x, y). dynamic: obstáculos (x, y, r).
        Devuelve waypoints simplificados con las maniobras de puerta explícitas; None si no hay ruta."""
        r = self.plan_res
        grid = self.plan_free.copy()
        if dynamic:
            iy, ix = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
            X, Y = self.x0 + (ix + 0.5) * r, self.y0 + (iy + 0.5) * r
            for ox, oy, rad in dynamic:
                grid &= (X - ox) ** 2 + (Y - oy) ** 2 >= (rad + HALF_WIDTH + 0.05) ** 2
        s = self._nearest_free(*start) if not self._free_cell(grid, *start) else start
        g = self._nearest_free(*goal) if not self._free_cell(grid, *goal) else goal
        if s is None or g is None:
            return None
        si, sj = self.to_ij(s[0], s[1], r)
        gi, gj = self.to_ij(g[0], g[1], r)
        came = self._astar(grid, (si, sj), (gi, gj))
        if came is None:
            return None
        return self._to_waypoints(came, grid, goal)

    def _free_cell(self, grid, x, y):
        i, j = self.to_ij(x, y, self.plan_res)
        return 0 <= i < grid.shape[0] and 0 <= j < grid.shape[1] and grid[i, j] and not self.corridor[i, j]

    def _astar(self, grid, s, g):
        ny, nx = grid.shape
        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]
        h = lambda a: math.hypot(a[0] - g[0], a[1] - g[1])
        openq = [(h(s), 0.0, s)]
        cost = {s: 0.0}
        parent = {s: None}
        while openq:
            _, c, cur = heapq.heappop(openq)
            if cur == g:
                path = []
                while cur is not None:
                    path.append(cur)
                    cur = parent[cur]
                return path[::-1]
            if c > cost[cur]:
                continue
            nbs = [((cur[0] + di, cur[1] + dj), w, di, dj) for di, dj, w in moves]
            nbs += [(other, sum(d.standoff) / self.plan_res, 0, 0) for other, d in self.door_edges.get(cur, [])]
            for nb, w, di, dj in nbs:
                if not (0 <= nb[0] < ny and 0 <= nb[1] < nx) or not grid[nb]:
                    continue
                if di and dj and not (grid[cur[0] + di, cur[1]] and grid[cur[0], cur[1] + dj]):
                    continue
                nc = c + w + self.clear_cost[nb]
                if nc < cost.get(nb, 1e18):
                    cost[nb] = nc
                    parent[nb] = cur
                    heapq.heappush(openq, (nc + h(nb), nc, nb))
        return None

    def _los(self, grid, a, b, min_clear=None):
        """Línea de visión para simplificar la ruta: todas las muestras con holgura >= min_clear (por defecto la de
        los extremos menos 2 cm), para no recortar esquinas que la ruta A* evitaba."""
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / (self.res)) + 2
        if min_clear is None:
            min_clear = max(HALF_WIDTH + 0.03, min(self.clearance(*a), self.clearance(*b), CIRCUMRADIUS + 0.05) - 0.02)
        for t in np.linspace(0, 1, n):
            x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            i, j = self.to_ij(x, y, self.plan_res)
            if not (0 <= i < grid.shape[0] and 0 <= j < grid.shape[1]) or not grid[i, j] or self.corridor[i, j]:
                return False
            if self.clearance(x, y) < min_clear:
                return False
        return True

    def _to_waypoints(self, cells, grid, goal) -> list[Waypoint]:
        """Tramos libres (simplificados por línea de visión) y cruces de puerta (pre -> post, alineados)."""
        r = self.plan_res
        wps: list[Waypoint] = []
        seg = [self.to_xy(*cells[0], r)]

        def flush(seg):
            anchor, k = seg[0], 1
            keep = [anchor]
            while k < len(seg):
                j = len(seg) - 1
                while j > k and not self._los(grid, anchor, seg[j]):
                    j -= 1
                anchor = seg[j]
                keep.append(anchor)
                k = j + 1
            for p in keep[1:] if wps else keep:
                wps.append(Waypoint(p[0], p[1]))

        for a, b in zip(cells[:-1], cells[1:]):
            door = next((d for other, d in self.door_edges.get(a, []) if other == b), None)
            if door is None:
                seg.append(self.to_xy(*b, r))
                continue
            flush(seg)
            pa, pb = np.array(self.to_xy(*a, r)), np.array(self.to_xy(*b, r))
            fwd = (pb - pa) @ door.normal > 0
            n = door.normal if fwd else -door.normal
            head = math.atan2(n[1], n[0])
            s_pre, s_post = door.standoff if fwd else door.standoff[::-1]
            left = np.array([-n[1], n[0]])       # el eje se desplaza para centrar la huella (asimétrica) en el vano
            pre = door.center - n * s_pre + left * LANE_OFFSET
            post = door.center + n * s_post + left * LANE_OFFSET
            wps.append(Waypoint(float(pre[0]), float(pre[1]), 'door_pre', head, door.name))
            wps.append(Waypoint(float(post[0]), float(post[1]), 'door_post', head, door.name))
            seg = [self.to_xy(*b, r)]
        if len(seg) > 1:
            flush(seg)
        out = []
        for w in wps:
            if out and w.kind == 'free' and math.hypot(w.x - out[-1].x, w.y - out[-1].y) < 0.08:
                continue
            out.append(w)
        if out and out[-1].kind == 'free':
            out[-1] = Waypoint(goal[0], goal[1], 'goal')
        else:
            out.append(Waypoint(goal[0], goal[1], 'goal'))
        return out

    # ------------------------------------------------------------------------------------------- ubicaciones
    def _locations(self) -> dict[str, Location]:
        locs: dict[str, Location] = {}
        for name, (lo, hi) in self.rooms.items():
            c = (lo + hi) / 2
            p = self._nearest_free(c[0], c[1], need_rot=True, max_r=1.5)
            locs[name] = Location(name, p[0], p[1], 0.0, name, 'room')
        for fu in self.spec['furniture']:
            loc = self._approach(fu)
            if loc is not None:
                locs[fu['name']] = loc
        return locs

    def _approach(self, fu, standoff=0.55) -> Location | None:
        """Pose frente al mueble (centro de la huella a `standoff` del borde), girable y dentro de su habitación."""
        cx, cy = fu['xy']
        sx, sy = fu['size'][0], fu['size'][1]
        room = fu.get('room')
        cands = []
        for nx_, ny_ in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            half = sx / 2 if nx_ else sy / 2
            for lat in (0.0, -0.3, 0.3, -0.6, 0.6):
                px = cx + nx_ * (half + standoff) + (lat if ny_ else 0.0)
                py = cy + ny_ * (half + standoff) + (lat if nx_ else 0.0)
                if not self.inside_bounds(px, py):
                    continue
                i, j = self.to_ij(px, py, self.plan_res)
                if self.rot_free[i, j] and self.room_of(px, py) == room:
                    cands.append((abs(lat), px, py, math.atan2(-ny_, -nx_)))
        if not cands:
            return None
        _, px, py, yaw = min(cands)
        return Location(fu['name'], px, py, yaw, room, 'furniture', fu['name'])
