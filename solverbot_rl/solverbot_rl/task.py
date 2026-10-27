"""Tarea de Pick and Place: estado, criterios verificables de éxito y recompensa.

Las recompensas y las métricas usan información INTERNA del simulador (pose real del objeto); eso es legítimo
porque no forma parte de la observación de la política (ver envs/pick_place.py).

Diseño de la recompensa:
  r = [gamma*Phi(s') - Phi(s)]                      shaping basado en potencial (Ng et al. 1999): no cambia la
                                                    política óptima y los ciclos no acumulan recompensa.
    + bonos de evento (agarre, levantamiento)       como mucho UNA vez por episodio.
    + bono de éxito                                 solo con el criterio verificable completo.
    - penalizaciones (tiempo, brusquedad, límites, colisión, caída, falta de progreso).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GOALS = ('reach', 'grasp', 'lift', 'place')


@dataclass
class TaskState:
    """Instantánea usada para recompensa/métricas (frame del robot = frame del mundo de entrenamiento)."""
    tcp_pos: np.ndarray
    tcp_z: np.ndarray                 # eje de aproximación de la pinza (z de tcp_link)
    obj_pos: np.ndarray
    gripper_q: float                  # joint7 medido (apertura de cada dedo, m)
    gripper_cmd_closed: bool          # la orden actual es cerrar
    arm_points: np.ndarray            # (N,3) puntos del brazo/pinza para el chequeo de colisión con la mesa
    joints_at_limit: int = 0
    obj_vel: float = 0.0              # |v| del objeto (m/s)


@dataclass
class EpisodeFlags:
    grasped_once: bool = False
    lifted_once: bool = False
    best_phi: float = -np.inf
    steps_since_progress: int = 0
    lift_streak: int = 0


@dataclass
class TaskParams:
    goal: str
    table_height: float
    table_rect: tuple                 # (xmin, xmax, ymin, ymax)
    target_xy: np.ndarray
    obj_half_height: float
    obj_half_width: float
    pregrasp_height: float = 0.08
    grasp_radius: float = 0.03
    lift_height: float = 0.06
    place_tolerance: float = 0.05
    release_clearance: float = 0.05
    drop_height_margin: float = 0.05
    no_progress_steps: int = 40
    w: dict = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict, goal: str, target_xy=None) -> 'TaskParams':
        wd, tk = cfg['world'], cfg['task']
        (tx, ty), (sx, sy, sz) = wd['table']['xy'], wd['table']['size']
        osz = wd['object']['size']
        return cls(goal=goal, table_height=tk['table_height'],
                   table_rect=(tx - sx / 2, tx + sx / 2, ty - sy / 2, ty + sy / 2),
                   target_xy=np.asarray(target_xy if target_xy is not None else wd['target_zone']['xy'], float),
                   obj_half_height=osz[2] / 2, obj_half_width=min(osz[0], osz[1]) / 2,
                   pregrasp_height=tk['pregrasp_height'], grasp_radius=tk['grasp_radius'],
                   lift_height=tk['lift_height'], place_tolerance=tk['place_tolerance'],
                   release_clearance=tk['release_clearance'], drop_height_margin=tk['drop_height_margin'],
                   no_progress_steps=tk['no_progress_steps'], w=dict(cfg['reward']))


# ------------------------------------------------------------------------------------------- predicados
def on_table_height(p: TaskParams) -> float:
    return p.table_height + p.obj_half_height


def lift_amount(s: TaskState, p: TaskParams) -> float:
    return float(s.obj_pos[2] - on_table_height(p))


def fingers_blocked_by_object(s: TaskState, p: TaskParams) -> bool:
    """Evidencia física de agarre: se ordenó cerrar pero los dedos NO llegaron a cerrarse del todo (algo los
    detiene) y el objeto está junto al TCP. No se exige apertura = media anchura: en Gazebo el objeto puede rotar
    dentro de la pinza (agarre por la arista) y la apertura bajar a ~7 mm manteniéndolo sujeto."""
    near = np.linalg.norm(s.tcp_pos - s.obj_pos) < p.grasp_radius
    blocked = 0.003 < s.gripper_q < p.obj_half_width + 0.008
    return bool(s.gripper_cmd_closed and near and blocked)


def is_held(s: TaskState, p: TaskParams) -> bool:
    """Agarrado Y separado de la mesa: no lo cumple un objeto empujado ni una pinza cerrada en el aire."""
    return fingers_blocked_by_object(s, p) and lift_amount(s, p) > 0.01


def arm_collides_table(s: TaskState, p: TaskParams) -> bool:
    x0, x1, y0, y1 = p.table_rect
    pts = s.arm_points
    inside = (pts[:, 0] > x0) & (pts[:, 0] < x1) & (pts[:, 1] > y0) & (pts[:, 1] < y1)
    return bool(np.any(inside & (pts[:, 2] < p.table_height + 0.005)))


def object_dropped(s: TaskState, p: TaskParams) -> bool:
    return bool(s.obj_pos[2] < p.table_height - p.drop_height_margin)


def placed_in_zone(s: TaskState, p: TaskParams, f: EpisodeFlags) -> bool:
    in_zone = np.linalg.norm(s.obj_pos[:2] - p.target_xy) < p.place_tolerance
    resting = abs(lift_amount(s, p)) < 0.01 and s.obj_vel < 0.05
    released = (not s.gripper_cmd_closed) and np.linalg.norm(s.tcp_pos - s.obj_pos) > p.release_clearance
    return bool(f.lifted_once and in_zone and resting and released)


def approach_point(s: TaskState, p: TaskParams) -> np.ndarray:
    off = p.pregrasp_height if p.goal == 'reach' else 0.0
    return s.obj_pos + np.array([0.0, 0.0, off])


def success(s: TaskState, p: TaskParams, f: EpisodeFlags) -> bool:
    if p.goal == 'reach':
        return bool(np.linalg.norm(s.tcp_pos - approach_point(s, p)) < 0.03 and s.tcp_z[2] < -0.8)
    if p.goal == 'grasp':
        return fingers_blocked_by_object(s, p)
    if p.goal == 'lift':
        return f.lift_streak >= 5
    return placed_in_zone(s, p, f)


# ------------------------------------------------------------------------------------------- recompensa
def potential(s: TaskState, p: TaskParams) -> float:
    w = p.w
    align = float(np.clip(-s.tcp_z[2], 0.0, 1.0))           # 1 si la pinza apunta hacia abajo
    if p.goal in ('lift', 'place') and is_held(s, p):
        # 1.0 > máximo del potencial sin agarre (w_align): agarrar nunca baja el potencial
        phi = 1.0 + w['w_lift'] * min(lift_amount(s, p), p.lift_height) / p.lift_height
        if p.goal == 'place':
            phi -= w['w_transport'] * float(np.linalg.norm(s.obj_pos[:2] - p.target_xy))
        return phi
    d = float(np.linalg.norm(s.tcp_pos - approach_point(s, p)))
    return -w['w_reach'] * d + w['w_align'] * align


def step_reward(prev: TaskState, cur: TaskState, action: np.ndarray, prev_action: np.ndarray, p: TaskParams,
                f: EpisodeFlags) -> tuple[float, dict, bool, bool]:
    """Devuelve (recompensa, términos, terminated, truncated_por_no_progreso). Actualiza `f`."""
    w = p.w
    terms: dict[str, float] = {}
    phi0, phi1 = potential(prev, p), potential(cur, p)
    terms['shaping'] = w['gamma'] * phi1 - phi0

    held = is_held(cur, p)
    if p.goal in ('grasp', 'lift', 'place') and fingers_blocked_by_object(cur, p) and not f.grasped_once:
        f.grasped_once = True
        terms['bonus_grasp'] = w['bonus_grasp']
    if held and lift_amount(cur, p) >= p.lift_height:
        f.lift_streak += 1
        if not f.lifted_once:
            f.lifted_once = True
            if p.goal in ('lift', 'place'):
                terms['bonus_lift'] = w['bonus_lift']
    else:
        f.lift_streak = 0

    terms['pen_time'] = -w['pen_time']
    terms['pen_action_rate'] = -w['pen_action_rate'] * float(np.mean((action - prev_action) ** 2))
    terms['pen_limit'] = -w['pen_limit'] * cur.joints_at_limit

    terminated = False
    if arm_collides_table(cur, p):
        terms['pen_collision'] = -w['pen_collision']
        terminated = True
    if object_dropped(cur, p):
        terms['pen_drop'] = -w['pen_drop']
        terminated = True
    ok = success(cur, p, f)
    if ok:
        terms['bonus_success'] = w['bonus_success']
        terminated = True

    # Falta de progreso: el potencial no mejora su máximo del episodio durante N pasos
    if phi1 > f.best_phi + 1e-3:
        f.best_phi = phi1
        f.steps_since_progress = 0
    else:
        f.steps_since_progress += 1
    stalled = (not terminated) and f.steps_since_progress >= p.no_progress_steps
    if stalled:
        terms['pen_no_progress'] = -w['pen_no_progress']
    terms['success'] = float(ok)
    return float(sum(v for k, v in terms.items() if k != 'success')), terms, terminated, stalled
