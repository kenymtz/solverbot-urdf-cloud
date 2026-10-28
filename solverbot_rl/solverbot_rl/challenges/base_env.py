"""Base común de los entornos Gymnasium de HRI, GPSR y Doing Laundry.

Arquitectura jerárquica (decisión de diseño, ver DESAFIOS_SPEC.md):
  * La POLÍTICA (entrenable con RL) elige en cada paso del entorno una HABILIDAD con un argumento (MultiDiscrete).
  * Cada habilidad se ejecuta físicamente durante muchos pasos de simulación de 0.1 s con controladores de referencia
    (planificador + seguimiento de ruta, apuntado de cabeza, brazo por IK). La navegación usa la pose ESTIMADA por
    LiDAR + odometría (no la verdad), salvo que se configure nav_pose_source='truth' (diagnóstico, queda en `info`).
  * La recompensa y el éxito se calculan con la VERDAD del simulador (posiciones reales de robot y personas, ángulo
    real de la mirada por cinemática directa de las articulaciones medidas). La verdad nunca entra en la observación.
  * La habilidad de navegación de bajo nivel también tiene su propio entorno RL (nav_env.py).
"""
from __future__ import annotations

import math
import os

import numpy as np

import gymnasium as gym

from ..kinematics import UrdfChain, solve_ik
from ..config import load_urdf
from .arena import AXLE, CIRCUMRADIUS, DOOR_STANDOFF, FOOT_RADIUS, FOOTPRINT, HALF_WIDTH, LASER_XY, ArenaMap
from .backend import PAN_LIM, TILT_LIM, KinematicMobileBackend, wrap
from .humans import HumanSim
from .nav import NavController, ScanLocalizer, foot_center
from .sensing import ColorHeadDetector, SimulatedHeadDetector, head_ray_angles, look_at_angles
from .world import ChallengeWorld

DT = 0.1
HEAD_Z_STANDING = 1.61      # altura supuesta de la cabeza de una persona de pie (modelo de persona del mundo)
PERSON_SOCIAL = 0.15
ARM_JOINTS = [f'joint{i}' for i in range(1, 7)]
STOW = [math.radians(v) for v in (86.28, 173.91, -164.0, 66.98, 0.69, -146.41)]
ROOMS = ['outside', 'living_room', 'kitchen', 'laundry', 'bedroom']


class SkillResult(dict):
    """Resultado de una habilidad: ok, motivo, tiempo simulado y datos medidos."""

    @property
    def ok(self):
        return bool(self.get('ok'))


class ChallengeEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, backend: str = 'kinematic', level: int = 0, max_skills: int = 60, max_sim_time: float = 900.0,
                 nav_pose_source: str = 'estimate', challenges_path: str | None = None, gz_world: str = 'home_challenges'):
        super().__init__()
        self.world = ChallengeWorld(challenges_path)
        self.arena: ArenaMap = self.world.arena
        self.ch = self.world.ch
        self.level = int(level)
        self.max_skills, self.max_sim_time = max_skills, max_sim_time
        self.nav_pose_source = nav_pose_source
        self.backend_name = backend
        colors = dict(self.world.person_rgb)
        colors.update({g['name']: g['rgb'] for g in self.ch['laundry']['garments']})
        colors.update({n: o['rgb'] for n, o in self.world.objects.items()})
        if backend == 'kinematic':
            self.backend = KinematicMobileBackend(self.arena)
            self.detector = SimulatedHeadDetector(self.arena)
            self.chain = UrdfChain(load_urdf())
        elif backend == 'gazebo':
            from .gazebo_mobile import GazeboMobileBackend
            self.backend = GazeboMobileBackend(self.arena, world=gz_world, camera=True)
            self.detector = ColorHeadDetector(colors, persons=self.world.person_names,
                                              static_reject=self.arena.on_static_surface)
            self.chain = self.backend.chain
        else:
            raise ValueError(backend)
        self.colors = colors
        self.localizer = ScanLocalizer(self.arena, self.backend.lidar_angles)
        self.humans = HumanSim(self.arena, self.world.person_names, self.world.parking, self.world.seats,
                               self.ch['seated_z_offset'])
        self.person_radius = self.ch['person_radius']
        self._pushed: dict[str, tuple] = {}

    # ================================================================================================ ciclo de vida
    def set_level(self, level: int):
        self.level = int(level)

    def _start(self, robot_ref_pose, door_open: bool, objects: dict | None = None):
        """Reinicio físico común. robot_ref_pose = (x, y, yaw) del punto de referencia (eje de ruedas)."""
        for n in self.humans.people:
            self.humans.park(n)
        self.door_open = door_open
        self.objects_truth = dict(objects or {})
        bx, by = ArenaMap.base_from_ref(robot_ref_pose[:2], robot_ref_pose[2])
        base = np.array([bx, by, robot_ref_pose[2]])
        self._pushed = {}
        ents = self._entity_poses()
        # radios para la verificación de contacto del backend: personas como círculos; la hoja de la puerta se modela
        # con su cadena de círculos ('door#k'); objetos y prendas no cuentan (son pequeños y van sobre muebles)
        self.backend.entity_radius = {n: (0.06 if n.startswith('door#') else self.person_radius
                                          if n in self.humans.people else 0.0) for n in ents}
        if isinstance(self.detector, SimulatedHeadDetector):
            self.detector.rng = np.random.default_rng(int(self.np_random.integers(2 ** 31)))   # reset reproducible
        self.backend.reset(self.np_random, base, {n: e[:3] for n, e in ents.items()})
        for n, e in ents.items():
            if not isinstance(self.backend, KinematicMobileBackend) and len(e) > 3:
                self.backend.set_entity(n, e[0], e[1], e[2], e[3])
            self._pushed[n] = tuple(np.round(e, 4))
        self.localizer.reset(base, self.backend.odom())
        self.est = base.copy()
        self.arm_q = np.array(STOW)
        self.sim_time = 0.0
        self.contact = False
        self.tipped = False
        self.skills_done = 0
        self.tracks: dict[str, tuple[np.ndarray, float]] = {}
        self.events: list[dict] = []
        self.fault: str | None = None
        self.loc_errors: list[float] = []
        self.halts = {'person': 0, 'scan': 0, 'map': 0}
        self.min_clearance, self.contact_with = 9.0, ''

    def _entity_poses(self) -> dict:
        ents = {}
        for p in self.humans.people.values():
            ents[p.name] = (p.pose[0], p.pose[1], p.pose[2], p.z)
        d = self.ch['entrance_door']
        if isinstance(self.backend, KinematicMobileBackend):
            leaf = self.world.door_leaf_circles if not self.door_open else self.world.door_leaf_open_circles
            for k, (x, y, r) in enumerate(leaf):
                ents[f'door#{k}'] = (x, y, 0.0)
        else:
            ents[d['name']] = ((*d['open_xy'], d['open_yaw'], 0.0) if self.door_open else
                               (*d['closed_xy'], 0.0, 0.0))
        if not isinstance(self.backend, KinematicMobileBackend):
            objs = getattr(self, 'objects_truth', {})
            for k, name in enumerate(self.world.objects):
                if name not in objs:          # objetos del mundo que el desafío no usa: fuera de la casa
                    ents[name] = (-8.0, -3.0 - 0.4 * k, 0.0, 0.05)
            for name, xyz in objs.items():
                ents[name] = (xyz[0], xyz[1], 0.0, xyz[2])
        return ents

    def _push_entities(self):
        for n, e in self._entity_poses().items():
            key = tuple(np.round(e, 4))
            if self._pushed.get(n) == key:
                continue
            self._pushed[n] = key
            if isinstance(self.backend, KinematicMobileBackend):
                self.backend.set_entity(n, e[0], e[1], e[2])
            else:
                self.backend.set_entity(n, e[0], e[1], e[2], e[3] if len(e) > 3 else 0.0)

    def open_entrance_door(self):
        self.door_open = True
        self._push_entities()

    # ================================================================================================ paso físico
    def truth(self) -> np.ndarray:
        return self.backend.pose_truth()

    def ref_truth(self) -> np.ndarray:
        return foot_center(self.truth())

    def person_plan_radius(self) -> float:
        """Radio con el que el planificador infla a una persona (sumado a la media huella): persona + margen social +
        lo que sobresalen las esquinas al girar (radio de giro - media huella). Así la ruta no entra en la zona en la
        que el gobernador de seguridad frena."""
        return self.person_radius + PERSON_SOCIAL + (CIRCUMRADIUS - HALF_WIDTH)

    def _robot_block(self):
        """Círculo que las personas simuladas esquivan al planificar: radio de giro del robot + persona + margen."""
        return [(*self.ref_truth(), CIRCUMRADIUS + self.person_radius + 0.13)]

    def obstacles_truth(self):
        obs = self.humans.obstacles(self.person_radius)
        obs += self.world.door_leaf_circles if not self.door_open else self.world.door_leaf_open_circles
        return obs

    def sim_step(self, v=0.0, w=0.0, head=None, perceive_every=3):
        if self.fault:
            return
        tr = self.truth()
        self.humans.update(DT, foot_center(tr), robot_speed=abs(v))
        self._push_entities()
        self.backend.cmd_vel(v, w)
        if head is not None:
            self.backend.head(float(np.clip(head[0], -PAN_LIM, PAN_LIM)), float(np.clip(head[1], -TILT_LIM, TILT_LIM)))
        self.backend.advance(DT)
        self.sim_time += DT
        odom, scan = self.backend.odom(), self.backend.scan()
        if not np.all(np.isfinite(odom)) or np.all(~np.isfinite(scan)):
            self.fault = 'sensor_invalido'
            return
        self.est = self.localizer.update(odom, scan)
        if not np.all(np.isfinite(self.est)):
            self.fault = 'localizacion_nan'
            return
        tr = self.truth()
        self.loc_errors.append(float(np.linalg.norm(self.est[:2] - tr[:2])))
        hit, clr, who = self.arena.contact_truth(tr[0], tr[1], tr[2], dynamic=self.obstacles_truth())
        self.min_clearance = min(self.min_clearance, clr)
        if self.backend.collided_truth() or hit:
            if not self.contact:
                self.contact_with = who or getattr(self.backend, 'contact_with', '') or 'backend'
            self.contact = True
        if self.backend.tipped_truth():
            self.tipped = True
        if int(round(self.sim_time / DT)) % perceive_every == 0:
            self.perceive()

    def perceive(self):
        pan, tilt = self.backend.head_joints()
        if isinstance(self.detector, SimulatedHeadDetector):
            targets = {n: self.humans.head_xyz(n) - np.array([0, 0, 0.45]) for n, p in self.humans.people.items()
                       if p.state != 'parked'}
            targets.update(self.perceivable_objects_truth())
            dets = self.detector.detect(self.truth(), self.est, pan, tilt, targets)
        else:
            dets = self.detector.detect(self.backend.head_rgbd(), self.est, names=self.detect_names(),
                                        gates=self.detection_gates())
        for n, xyz in dets.items():
            self.tracks[n] = (np.asarray(xyz, float), self.sim_time)

    def perceivable_objects_truth(self) -> dict:
        """Objetos que el detector simulado puede ver (los entornos lo amplían)."""
        return {}

    def detection_gates(self) -> dict:
        """Compuertas espaciales por nombre para el detector de color de Gazebo (los entornos las amplían)."""
        return {}

    def detect_names(self):
        return [n for n in self.world.person_names if self.humans.people[n].state != 'parked'] + \
            list(self.perceivable_objects_truth())

    def track(self, name, max_age=6.0):
        t = self.tracks.get(name)
        if t is None or self.sim_time - t[1] > max_age:
            return None
        return t[0]

    # ================================================================================================ habilidades
    def failed(self):
        return self.contact or self.tipped or self.fault is not None

    def skill_wait(self, seconds=1.0, head=None) -> SkillResult:
        t0 = self.sim_time
        for _ in range(max(1, int(round(seconds / DT)))):
            self.sim_step(0.0, 0.0, head)
            if self.failed():
                break
        return SkillResult(ok=not self.failed(), time=self.sim_time - t0)

    def stow_arm(self):
        if np.max(np.abs(self.arm_q - STOW)) < 0.02:
            return
        self._command_arm(STOW)

    def _command_arm(self, q, max_steps=60):
        if isinstance(self.backend, KinematicMobileBackend):
            self.arm_q = np.array(q)
            self.sim_step()
            return True
        self.backend._pub(self.backend.pub_arm, q)
        for _ in range(max_steps):
            self.sim_step()
            js = self.backend.joint_positions()
            cur = np.array([js.get(j, 0.0) for j in ARM_JOINTS])
            if np.max(np.abs(cur - np.asarray(q))) < 0.02:
                break
        js = self.backend.joint_positions()
        self.arm_q = np.array([js.get(j, 0.0) for j in ARM_JOINTS])
        return np.max(np.abs(self.arm_q - np.asarray(q))) < 0.05

    def tcp_truth(self) -> np.ndarray:
        """Punta de la pinza en el mapa (FK de las articulaciones MEDIDAS y pose real de la base)."""
        js = {**dict(zip(ARM_JOINTS, self.arm_q)), 'torso_lift_joint': 0.0}
        if not isinstance(self.backend, KinematicMobileBackend):
            js = self.backend.joint_positions()
        t = self.chain.link_pose('tcp_link', js)[:3, 3]
        tr = self.truth()
        c, s = math.cos(tr[2]), math.sin(tr[2])
        return np.array([tr[0] + c * t[0] - s * t[1], tr[1] + s * t[0] + c * t[1], t[2]])

    def skill_reach(self, target_map_xyz, approach_axis=None) -> SkillResult:
        """Lleva la punta de la pinza a un punto (estimado) del mapa con IK; usa la pose ESTIMADA de la base.
        approach_axis: dirección del eje z de la pinza en el frame base (p. ej. (0, 0, -1) = apuntando hacia abajo)."""
        est = self.est
        c, s = math.cos(est[2]), math.sin(est[2])
        d = np.asarray(target_map_xyz[:2]) - est[:2]
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], target_map_xyz[2]])
        kw = {} if approach_axis is None else {'approach_axis': np.asarray(approach_axis, float)}
        q, err = solve_ik(self.chain, 'tcp_link', ARM_JOINTS, np.array(STOW), local, None, {'torso_lift_joint': 0.0},
                          restarts=4, reg=0.002, **kw)
        if err > 0.03:
            return SkillResult(ok=False, reason='fuera_de_alcance', ik_err=err)
        ok = self._command_arm(q)
        return SkillResult(ok=ok and not self.failed(), ik_err=err)

    def final_approach_plan(self, base_xy, yaw, dynamic=(), back=0.6, margin=0.05):
        """Aproximación final RECTA a una pose junto a obstáculos o personas (donde no se puede girar en el lugar):
        devuelve (ref, pre) si la pose final y todo el tramo recto desde una pose previa `back` m detrás (en la que SÍ
        se puede girar) tienen holgura geométrica exacta >= margin contra el mapa y los círculos `dynamic`; si no, None."""
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        fwd = R[:, 0]
        base = np.asarray(base_xy, float)
        for sgo in np.linspace(0.0, back, 7):
            b = base - fwd * sgo
            if self.arena.footprint_clearance(b[0], b[1], yaw, dynamic=list(dynamic))[0] < margin:
                return None
        ref = base + R @ np.array([AXLE[0], AXLE[1]])
        pre = ref - fwd * back
        i, j = self.arena.to_ij(*pre, self.arena.plan_res)
        rf = self.arena.rot_free
        if not (0 <= i < rf.shape[0] and 0 <= j < rf.shape[1] and rf[i, j]):
            return None
        if any(math.hypot(pre[0] - x, pre[1] - y) < CIRCUMRADIUS + r + 0.05 for x, y, r in dynamic):
            return None                      # en la pose previa se gira: el barrido no puede tocar a nadie
        return ref, pre

    def skill_approach(self, ref, pre, yaw) -> SkillResult:
        """Navega a la pose previa (girable) con el rumbo final y entra en línea recta hasta ref."""
        res = self.skill_navigate(pre, yaw)
        if not res.ok:
            return res
        return self.skill_straight(ref, yaw)

    def skill_straight(self, goal_ref, yaw, v=0.10, tol=0.03, max_time=20.0, person_margin=0.03) -> SkillResult:
        """Avance en línea recta (rumbo `yaw`, corrección lateral suave) hasta que el eje de ruedas llega a goal_ref.
        Para aproximaciones finales junto a muebles o personas, donde no se puede girar en el lugar. El freno por
        personas usa `person_margin` (< margen con que final_approach_plan validó el tramo; HRI usa 0.08 por el ruido de percepción): con el margen
        de navegación (0.08) el robot se detenía para siempre antes de una pose de entrega válida (HRI N5, semilla 1000).
        Si la persona se mueve hacia el tramo, el freno sigue actuando."""
        t0 = self.sim_time
        n = np.array([math.cos(yaw), math.sin(yaw)])
        lat_n = np.array([-n[1], n[0]])
        while self.sim_time - t0 < max_time and not self.failed():
            c = foot_center(self.est)
            along = float((np.asarray(goal_ref) - c) @ n)
            if along < tol:
                self.sim_step(0.0, 0.0, (0.0, 0.2))
                ok = float(np.linalg.norm(self.ref_truth() - np.asarray(goal_ref))) < 0.08
                return SkillResult(ok=ok, reason=None if ok else 'llegada_imprecisa', time=self.sim_time - t0)
            e_lat = float((c - np.asarray(goal_ref)) @ lat_n)
            desired = yaw - float(np.clip(math.atan2(2.0 * e_lat, v), -math.radians(4), math.radians(4)))
            w = float(np.clip(2.5 * wrap(desired - self.est[2]), -0.2, 0.2))
            vv = min(v, max(0.04, along))
            # predicción acotada a lo que falta recorrer: más allá de la pose final validada no se va a ir
            hz = tuple(sorted({min(t, along / vv) for t in (0.3, 0.6, 0.9)}))
            if (self._person_in_sweep(self.est, vv, None, w, margin=person_margin, horizon=hz)
                    or self._scan_blocked(vv, w, horizon=hz)):
                vv, w = 0.0, 0.0
            self.sim_step(vv, w, (0.0, 0.2))
        return SkillResult(ok=False, reason='tiempo_agotado' if not self.failed() else
                           ('colision' if self.contact else 'falla'), time=self.sim_time - t0)

    def skill_reverse(self, dist, v=0.10, max_time=20.0) -> SkillResult:
        """Retroceso recto `dist` m (deshace una aproximación final recta junto a un mueble)."""
        t0, start = self.sim_time, foot_center(self.est).copy()
        while self.sim_time - t0 < max_time and not self.failed():
            if np.linalg.norm(foot_center(self.est) - start) >= dist:
                self.sim_step(0.0, 0.0, (0.0, 0.2))
                return SkillResult(ok=True, time=self.sim_time - t0)
            self.sim_step(-v, 0.0, (0.0, 0.2))
        return SkillResult(ok=False, reason='tiempo_agotado' if not self.failed() else 'falla')

    def skill_turn(self, yaw, w_max=0.5, tol=math.radians(3), max_time=20.0) -> SkillResult:
        """Giro en el lugar (alrededor del eje de ruedas) hasta el rumbo `yaw` (pose estimada), si el barrido está libre."""
        t0 = self.sim_time
        while self.sim_time - t0 < max_time and not self.failed():
            err = wrap(yaw - self.est[2])
            if abs(err) < tol:
                self.sim_step(0.0, 0.0, (0.0, 0.2))
                return SkillResult(ok=True, time=self.sim_time - t0)
            w = float(np.clip(1.5 * err, -w_max, w_max))
            nxt = self.est[2] + w * DT * 3
            base = foot_center(self.est) - np.array([math.cos(nxt), math.sin(nxt)]) * AXLE[0]
            if self.arena.footprint_collides(base[0], base[1], nxt, margin=0.02):
                return SkillResult(ok=False, reason='giro_bloqueado', time=self.sim_time - t0)
            self.sim_step(0.0, w, (0.0, 0.2))
        return SkillResult(ok=False, reason='tiempo_agotado' if not self.failed() else 'falla')

    def look_at(self, xyz, settle=4, turn=False):
        """Apunta la cabeza a xyz (pose estimada). turn=True: si queda fuera del rango del pan, gira la base antes."""
        if turn:
            pan, _ = look_at_angles(self.est, xyz)
            if abs(pan) > PAN_LIM - 0.15:
                self.skill_turn(math.atan2(xyz[1] - self.est[1], xyz[0] - self.est[0]))
        pan, tilt = look_at_angles(self.est, xyz)
        pan, tilt = float(np.clip(pan, -PAN_LIM, PAN_LIM)), float(np.clip(tilt, -TILT_LIM, TILT_LIM))
        for k in range(settle + 30):
            self.sim_step(0.0, 0.0, (pan, tilt), perceive_every=1)
            mp, mt = self.backend.head_joints()       # esperar a que los servos lleguen (en Gazebo son más lentos)
            if k + 1 >= settle and abs(mp - pan) < 0.02 and abs(mt - tilt) < 0.02:
                break
        return pan, tilt

    def gaze_error_truth(self, xyz) -> float:
        """Ángulo real (grados) entre el eje de la cabeza (pan/tilt medidos) y el punto (verdad)."""
        pan, tilt = self.backend.head_joints()
        ang, _ = head_ray_angles(self.truth(), pan, tilt, xyz)
        return math.degrees(ang)

    def skill_navigate(self, goal_xy, final_yaw=None, head_fn=None, max_time=150.0, follower: str | None = None,
                       max_replans=3, stop_near=None, ignore: str | None = None) -> SkillResult:
        """Navegación física: planifica para el eje de ruedas con personas percibidas como obstáculos y sigue la ruta
        con la pose estimada (o la verdad si nav_pose_source='truth')."""
        self.stow_arm()
        t0 = self.sim_time
        start_ref = self.ref_truth()
        replans, moving_steps, forward_gaze = 0, 0, 0
        while replans <= max_replans:
            self._exit_corridor()
            self._exit_unrotatable()
            pose = self.est if self.nav_pose_source == 'estimate' else self.truth()
            # margen social de 0.15 m: la ruta pasa fuera de la zona en la que el gobernador de seguridad frena
            dyn = [(*self.track(n)[:2], self.person_plan_radius()) for n in self.humans.people
                   if n not in (follower, ignore) and self.track(n, 4.0) is not None
                   and np.linalg.norm(self.track(n)[:2] - foot_center(pose)) > 0.6]
            # hoja de la puerta de entrada abierta: obstáculo de posición conocida (barrido de la puerta) que no está en
            # el mapa estático; la verdad de contacto ya la incluye (obstacles_truth)
            if self.door_open:
                dyn += [(x, y, r) for x, y, r in self.world.door_leaf_open_circles]
            wps = self.arena.plan(foot_center(pose), goal_xy, dynamic=dyn) or self.arena.plan(foot_center(pose), goal_xy)
            if wps is None:
                return SkillResult(ok=False, reason='sin_ruta', time=self.sim_time - t0)
            nav = NavController(wps, final_yaw=final_yaw)
            last_prog, best, halted, settled = self.sim_time, 1e9, 0.0, set()
            while not nav.done:
                pose = self.est if self.nav_pose_source == 'estimate' else self.truth()
                if nav.mode == 'align' and nav.k not in settled:
                    # en el punto previo de una puerta: 2 s quieto para que el scan-matching converja (holgura 15.5 mm)
                    settled.add(nav.k)
                    for _ in range(20):
                        self.sim_step(0.0, 0.0, (0.0, 0.15))
                    pose = self.est if self.nav_pose_source == 'estimate' else self.truth()
                v, w = nav.act(pose)
                why = self._halt_reason(pose, v, w, follower, nav.in_door)
                if why and abs(w) > 1e-3 and v > 0.02 and not self._halt_reason(pose, v, 0.0, follower, nav.in_door):
                    # el giro haría barrer la huella sobre el obstáculo pero avanzar recto es seguro: alejarse primero
                    w, why = 0.0, None
                if why:
                    self.halts[why] += 1
                    v, w = 0.0, 0.0                  # gobernador de seguridad: persona percibida en la zona de barrido
                    halted += DT
                    if halted > 4.0 and not nav.in_door:
                        break                        # replanificar con las posiciones actuales
                else:
                    halted = 0.0
                head = head_fn(pose) if head_fn else (0.0, 0.15)
                self.sim_step(v, w, head)
                if abs(v) > 0.05:
                    moving_steps += 1
                    forward_gaze += int(abs(self.backend.head_joints()[0]) < 0.35)
                if self.failed():
                    return SkillResult(ok=False, reason='colision' if self.contact else ('vuelco' if self.tipped
                                       else self.fault), time=self.sim_time - t0)
                if stop_near is not None and not nav.in_door and np.linalg.norm(foot_center(pose) - np.asarray(stop_near[0])) < stop_near[1]:
                    self.sim_step(0.0, 0.0, head)
                    return SkillResult(ok=True, time=self.sim_time - t0, traveled=float(np.linalg.norm(
                        self.ref_truth() - start_ref)))
                dist = float(np.linalg.norm(foot_center(pose) - np.asarray(goal_xy)))
                if dist < best - 0.05:
                    best, last_prog = dist, self.sim_time
                if nav.in_door and self.sim_time - t0 < max_time + 30.0:
                    continue                         # nunca abandonar la ruta en medio de una puerta
                if self.sim_time - last_prog > 12.0 or self.sim_time - t0 > max_time:
                    break
            else:
                ok = float(np.linalg.norm(self.ref_truth() - np.asarray(goal_xy))) < 0.20
                return SkillResult(ok=ok, reason=None if ok else 'llegada_imprecisa', time=self.sim_time - t0,
                                   forward_gaze=forward_gaze / max(1, moving_steps),
                                   traveled=float(np.linalg.norm(self.ref_truth() - start_ref)))
            if self.sim_time - t0 > max_time:
                break
            replans += 1
            self.skill_wait(1.0)
        return SkillResult(ok=False, reason='bloqueado', time=self.sim_time - t0)

    def _in_corridor(self) -> bool:
        c = foot_center(self.est)
        for d in self.arena.doors:
            rel = c - d.center
            along = float(rel @ d.normal)
            lat = float(rel @ np.array([-d.normal[1], d.normal[0]]))
            if abs(lat) < 0.15 and abs(along) < DOOR_STANDOFF - 0.05:
                return True
        return False

    def _exit_corridor(self, max_steps=80):
        """Si una ruta se interrumpe dentro del vano de una puerta (no se puede girar allí), retroceder en línea recta
        por donde se entró hasta salir del corredor antes de replanificar."""
        n = 0
        while self._in_corridor() and n < max_steps and not self.failed():
            self.sim_step(-0.10, 0.0, (0.0, 0.15))
            n += 1
        if n:
            self.sim_step(0.0, 0.0, (0.0, 0.15))

    def _exit_unrotatable(self, margin=0.01, max_dist=0.6):
        """Si la navegación arranca donde NO se puede girar en el lugar (holgura exacta del giro < margin; p. ej. tras
        una aproximación a una persona junto a una mesa), salir en línea recta (atrás o adelante, el sentido cuyo tramo
        tiene holgura exacta) hasta un punto girable. GPSR N3 semilla 1019: el primer giro de la navegación barría la
        tapa de la mesa."""
        est = self.est
        ax = foot_center(est)
        if self.arena.rotation_clearance(*ax) >= margin:
            return
        fwd = np.array([math.cos(est[2]), math.sin(est[2])])
        people = [(*t[:2], self.person_radius + 0.05) for t in (self.track(n, 4.0) for n in self.humans.people)
                  if t is not None]
        for sgn in (-1.0, 1.0):
            for d in np.arange(0.05, max_dist + 1e-9, 0.05):
                b = est[:2] + sgn * fwd * d
                if self.arena.footprint_clearance(b[0], b[1], est[2], dynamic=people)[0] < 0.02:
                    break
                if self.arena.rotation_clearance(*(ax + sgn * fwd * d)) >= margin:
                    start, n = foot_center(self.est).copy(), 0
                    while np.linalg.norm(foot_center(self.est) - start) < d and n < 200 and not self.failed():
                        self.sim_step(sgn * 0.10, 0.0, (0.0, 0.15))
                        n += 1
                    self.sim_step(0.0, 0.0, (0.0, 0.15))
                    return

    def door_scan_open(self) -> bool:
        """¿El LiDAR ve a través del vano de la puerta de entrada? (rayos hacia el centro del vano > 1 m más allá)."""
        est = self.est
        d = np.array([0.08, 2.59]) - np.array([est[0], est[1]])
        dist = float(np.linalg.norm(d))
        if dist > 4.0:
            return getattr(self, 'mem', {}).get('door_open', False)
        bearing = math.atan2(d[1], d[0]) - est[2]
        ang = np.asarray(self.backend.lidar_angles)
        sel = np.abs((ang - bearing + math.pi) % (2 * math.pi) - math.pi) < math.radians(4)
        if not sel.any():
            return False
        r = np.asarray(self.backend.scan())[sel]
        r = np.where(np.isfinite(r), r, 10.0)
        return bool(np.median(r) > dist + 1.0)

    def skill_wait_door(self, terms, reward):
        """Espera (hasta 30 s) a que el LiDAR vea abierta la puerta de entrada."""
        t0 = self.sim_time
        while self.sim_time - t0 < 30.0 and not self.failed():
            self.sim_step(0.0, 0.0, (0.0, 0.1))
            if self.door_scan_open():
                self.mem['door_open'] = True
                if self.door_open and not self.done['door']:
                    self.done['door'] = True
                    terms['door'] = reward
                self.skill_wait(0.5)
                return SkillResult(ok=True, time=self.sim_time - t0)
        return SkillResult(ok=False, reason='puerta_cerrada')

    def head_from_track(self, t) -> np.ndarray:
        """Cabeza estimada a partir de la detección del torso: la altura sale de la POSTURA percibida (torso alto = de
        pie, bajo = sentado) y de la altura supuesta de las personas, no del centroide de la camiseta (que sube cuando
        la cámara mira hacia arriba y solo ve la parte superior)."""
        seated = t[2] < 0.95
        return np.array([t[0], t[1], HEAD_Z_STANDING + (self.ch['seated_z_offset'] if seated else 0.0)])

    def talk(self, p, seconds=2.0, gaze_max_deg=12.0):
        """Habla mirando a la persona p durante `seconds` siguiendo su posición PERCIBIDA (cabeza ~0.45 m sobre el
        torso detectado); devuelve la fracción del tiempo con contacto visual REAL (< gaze_max_deg) y el error final."""
        t = self.track(p, 3.0)
        if t is not None:                         # fuera del rango del pan: girar la base hacia la persona
            pan, _ = look_at_angles(self.est, self.head_from_track(t))
            if abs(pan) > PAN_LIM - 0.15:
                self.skill_turn(math.atan2(t[1] - self.est[1], t[0] - self.est[0]))
        ok_steps, belief_steps, n, err = 0, 0, 0, 180.0
        for _ in range(int(seconds / DT)):
            t = self.track(p, 3.0)
            if t is not None:
                head = self.head_from_track(t)
                self.sim_step(0, 0, look_at_angles(self.est, head), perceive_every=1)
                pan, tilt = self.backend.head_joints()
                belief_steps += int(math.degrees(head_ray_angles(self.est, pan, tilt, head)[0]) < gaze_max_deg)
            else:
                self.sim_step(0, 0)
            err = self.gaze_error_truth(self.humans.head_xyz(p))
            ok_steps += int(err < gaze_max_deg)
            n += 1
        self.last_talk_belief = belief_steps / max(1, n)      # lo que el robot cree (pose estimada + percepción)
        return ok_steps / max(1, n), err

    def skill_follow(self, p, done_room=None, max_time=240.0) -> SkillResult:
        """Sigue a la persona p a ~1 m usando solo sus detecciones. Termina por PERCEPCIÓN cuando la persona queda
        quieta > 3 s (y, si se indica, dentro de done_room)."""
        t0 = self.sim_time
        last, hist, seen = self.track(p, 30.0), [], []
        self.skill_wait(1.0, head=look_at_angles(self.est, (*last[:2], 1.2)) if last is not None else (0.0, 0.2))
        while self.sim_time - t0 < max_time and not self.failed():
            t = self.track(p, 1.0)               # solo detecciones recientes: la persona se mueve
            if t is None:
                last = self.track(p, 60.0)
                goal = None
                if last is not None and len(seen) >= 2 and np.linalg.norm(seen[-1][1] - seen[0][1]) > 0.3:
                    # perdida de vista mientras caminaba: extrapolar 1.5 m en su dirección de marcha (p. ej. cruzó
                    # una puerta y quedó detrás de una pared); el planificador resuelve las puertas
                    d = seen[-1][1] - seen[0][1]
                    goal = last[:2] + d / np.linalg.norm(d) * 1.5
                    if self.arena.plan(foot_center(self.est), goal) is None:
                        goal = None
                if goal is None and last is not None and np.linalg.norm(last[:2] - foot_center(self.est)) > 1.2:
                    goal = last[:2]
                if goal is not None and np.linalg.norm(goal - foot_center(self.est)) > 0.5:
                    self.skill_navigate(goal, None, max_time=8.0, max_replans=0, stop_near=(goal, 0.4), ignore=p,
                                        head_fn=lambda pose: look_at_angles(pose, (*goal, 1.2)))
                    seen = seen[-1:]
                elif not self.skill_search(p).ok:
                    # búsqueda fallida (p. ej. junto a una puerta, donde el giro en el lugar está bloqueado): ir a la
                    # habitación de destino anunciada en la instrucción, o al último punto visto, y buscar desde ahí
                    # (HRI N5 semilla 1000: el anfitrión ya estaba en la cocina y el robot giraba en vano en el vano)
                    loc = self.arena.locations.get(done_room) if done_room is not None else None
                    goal = np.array([loc.x, loc.y]) if loc is not None else (last[:2] if last is not None else None)
                    if goal is not None and np.linalg.norm(goal - foot_center(self.est)) > 0.5:
                        self.skill_navigate(goal, None, max_time=30.0, stop_near=(goal, 0.6), ignore=p,
                                            head_fn=lambda pose: look_at_angles(pose, (*goal, 1.2)))
                continue
            seen.append((self.sim_time, t[:2].copy()))
            seen = [x for x in seen if self.sim_time - x[0] <= 3.0]
            if np.linalg.norm(t[:2] - foot_center(self.est)) > 1.4:
                # ir hacia la persona por el planificador (respeta puertas) y detenerse a ~1 m
                self.skill_navigate(t[:2], None, max_time=6.0, max_replans=0, stop_near=(t[:2], 1.0), ignore=p,
                                    head_fn=lambda pose: look_at_angles(pose, (*t[:2], 1.3)))
            else:
                self.skill_wait(0.5, head=look_at_angles(self.est, (*t[:2], 1.3)))
            hist.append((self.sim_time, t[:2].copy()))
            hist = [h for h in hist if self.sim_time - h[0] <= 3.5]
            if ((done_room is None or self.arena.room_of(*t[:2]) == done_room) and self.sim_time - hist[0][0] >= 3.0
                    and max(np.linalg.norm(h[1] - t[:2]) for h in hist) < 0.12):
                return SkillResult(ok=not self.failed(), time=self.sim_time - t0)
        return SkillResult(ok=False, reason='tiempo_agotado' if not self.failed() else 'falla', time=self.sim_time - t0)

    def skill_search(self, name, max_turn=2 * math.pi, w=0.6, restore=False) -> SkillResult:
        """Busca a `name` girando en el lugar (si el barrido del giro está libre en el mapa) y paneando la cabeza."""
        t0, turned = self.sim_time, 0.0
        for pan in (-1.0, 0.0, 1.0):
            self.skill_wait(0.3, head=(pan, 0.2))
            if self.track(name, 0.4) is not None:
                return SkillResult(ok=True, time=self.sim_time - t0)
        while turned < max_turn and not self.failed():
            est = self.est
            nxt = est[2] + w * DT * 3
            ax = foot_center(est)
            base = ax - np.array([math.cos(nxt), math.sin(nxt)]) * AXLE[0]
            if self.arena.footprint_collides(base[0], base[1], nxt, margin=0.02,
                                             dynamic=[(*t[:2], self.person_radius + 0.08) for t in
                                                      (self.track(n, 8.0) for n in self.humans.people) if t is not None]):
                break
            self.sim_step(0.0, w, (0.0, 0.2))
            turned += w * DT
            if self.track(name, 0.4) is not None:
                break
        self.sim_step(0.0, 0.0, (0.0, 0.2))
        found = self.track(name, 0.6) is not None
        if restore and turned > 0.05 and not self.failed():
            # volver al rumbo inicial (el giro de ida ya verificó el barrido)
            back = 0.0
            while back < turned - 1e-3 and not self.failed():
                self.sim_step(0.0, -w, (0.0, 0.2))
                back += w * DT
            self.sim_step(0.0, 0.0, (0.0, 0.2))
        return SkillResult(ok=found and not self.failed(), time=self.sim_time - t0)

    def _scan_blocked(self, v, w, horizon=(0.3, 0.6, 0.9), margin=0.02) -> bool:
        """Gobernador por LiDAR: True si algún punto del último scan cae dentro de la huella (+margen) en las poses
        predichas con (v, w) en los próximos 0.3-0.9 s (personas fuera de la vista de la cámara, obstáculos no
        mapeados). No se usa durante las maniobras de puerta (holgura de 1.5 cm por lado, validada por el plan)."""
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            return False
        r = np.asarray(self.backend.scan(), float)
        ang = np.asarray(self.backend.lidar_angles, float)
        ok = np.isfinite(r) & (r > 0.05) & (r < 1.5)
        if not ok.any():
            return False
        P = np.stack([LASER_XY[0] + r[ok] * np.cos(ang[ok]), LASER_XY[1] + r[ok] * np.sin(ang[ok])], 1)
        x0, x1, y0, y1 = FOOTPRINT
        # auto-filtro: retornos dentro de la propia huella son el cuerpo del robot (brazo plegado, en Gazebo)
        P = P[~((P[:, 0] > x0 - 0.02) & (P[:, 0] < x1 + 0.02) & (P[:, 1] > y0 - 0.02) & (P[:, 1] < y1 + 0.02))]
        # solo obstáculos NO mapeados: se descartan los puntos que el mapa estático explica (pose estimada)
        est = self.est
        c0, s0 = math.cos(est[2]), math.sin(est[2])
        X, Y = est[0] + c0 * P[:, 0] - s0 * P[:, 1], est[1] + s0 * P[:, 0] + c0 * P[:, 1]
        a = self.arena
        ii = ((Y - a.y0) // a.res).astype(int).clip(0, a.lidar_dist.shape[0] - 1)
        jj = ((X - a.x0) // a.res).astype(int).clip(0, a.lidar_dist.shape[1] - 1)
        um = a.lidar_dist[ii, jj] > 0.08
        P = P[um]
        if not len(P):
            return False
        if os.environ.get('SOLVERBOT_DEBUG_SCAN'):
            self._last_unmapped = np.stack([X[um], Y[um]], 1)
        ax0 = np.array(AXLE)

        def rect_dist(Q):                            # distancia mínima de los puntos al rectángulo de la huella
            dx = np.maximum(np.maximum(x0 - Q[:, 0], Q[:, 0] - x1), 0.0)
            dy = np.maximum(np.maximum(y0 - Q[:, 1], Q[:, 1] - y1), 0.0)
            return float(np.min(np.hypot(dx, dy)))
        now = rect_dist(P)
        for t in horizon:
            th = w * t
            if abs(w) > 1e-6:
                adv = np.array([math.sin(th), 1 - math.cos(th)]) * (v / w)
            else:
                adv = np.array([v * t, 0.0])
            c, s = math.cos(th), math.sin(th)
            Rt = np.array([[c, -s], [s, c]])
            base_t = ax0 + adv - Rt @ ax0
            Q = (P - base_t) @ Rt                    # puntos en el frame de la base predicha
            d = rect_dist(Q)
            # como el freno por personas: solo frena lo que ACERCA la huella al obstáculo; alejarse sigue permitido
            # (antes congelaba al robot junto a un invitado sentado: HRI N5 semillas 1006 y 1007)
            if d < margin and d < now - 1e-3:
                return True
        return False

    def _halt_reason(self, pose, v, w, follower, in_door):
        if self._person_in_sweep(pose, v, follower, w):
            return 'person'
        if not in_door and self._scan_blocked(v, w):
            return 'scan'
        if self._map_blocked(pose, v, w, margin=0.0 if in_door else 0.01):
            return 'map'
        return None

    def _map_blocked(self, pose, v, w, margin=0.01, horizon=(0.3, 0.6)) -> bool:
        """Gobernador contra el MAPA ESTÁTICO: True si la huella real, en las poses predichas con (v, w), queda a menos
        de `margin` (holgura geométrica exacta) de una primitiva de colisión y además se acerca. La grilla del
        planificador supone la huella orientada a la ruta (semi-ancho + 1 cm); con rumbo oblicuo las esquinas pueden
        tocar el extremo de una pared (GPSR N3 semilla 1002). Usa la pose estimada: el error de localización se cubre
        con el margen."""
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            return False
        a = self.arena
        i, j = a.to_ij(*foot_center(pose))
        reach = max(abs(v), 0.05) * horizon[-1] + 0.05
        if 0 <= i < a.nav_dist.shape[0] and 0 <= j < a.nav_dist.shape[1] and a.nav_dist[i, j] > FOOT_RADIUS + reach:
            return False
        now = a.footprint_clearance(pose[0], pose[1], pose[2])[0]
        for b in self._predicted_bases(pose, v, w, horizon):
            c = a.footprint_clearance(b[0], b[1], b[2])[0]
            if c < margin and c < now - 1e-4:
                return True
        return False

    def _predicted_bases(self, pose, v, w, horizon=(0.3, 0.6, 0.9, 1.2)):
        """Poses de base_footprint predichas (mapa) aplicando (v, w) alrededor del eje de ruedas durante `horizon` s."""
        ax = foot_center(pose)
        out = []
        for t in horizon:
            th = pose[2] + w * t
            if abs(w) > 1e-6:
                adv = np.array([math.sin(th) - math.sin(pose[2]), math.cos(pose[2]) - math.cos(th)]) * (v / w)
            else:
                adv = np.array([math.cos(th), math.sin(th)]) * v * t
            a_t = ax + adv
            out.append((a_t[0] - AXLE[0] * math.cos(th) + AXLE[1] * math.sin(th),
                        a_t[1] - AXLE[0] * math.sin(th) - AXLE[1] * math.cos(th), th))
        return out

    def _person_in_sweep(self, pose, v, follower=None, w=0.0, margin=0.08, horizon=(0.3, 0.6, 0.9, 1.2)) -> bool:
        """True si la huella, en las poses predichas con (v, w), queda a menos de `margin` de una persona detectada hace
        < 1 s (círculo de radio person_radius). Solo frena los movimientos que la acercarían: alejarse sigue permitido."""
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            return False
        circles = []
        for n in self.humans.people:
            if n == follower:
                continue
            t = self.track(n, 1.0)
            if t is not None:
                circles.append((t[0], t[1], self.person_radius + margin))
        if not circles:
            return False
        now = self._foot_min_dist(pose, circles)
        for b in self._predicted_bases(pose, v, w, horizon):
            d = self._foot_min_dist(b, circles)
            if d < 0.0 and d < now - 1e-3:
                return True
        return False

    def _foot_min_dist(self, base_pose, circles) -> float:
        pts = self.arena.footprint_points(base_pose[0], base_pose[1], base_pose[2], step=0.05)
        return min(float(np.min(np.hypot(pts[:, 0] - x, pts[:, 1] - y))) - r for x, y, r in circles)

    # ================================================================================================ observación común
    def robot_obs(self) -> np.ndarray:
        est = self.est
        room = self.arena.room_of(*foot_center(est)) or 'outside'
        r = np.zeros(len(ROOMS))
        r[ROOMS.index(room) if room in ROOMS else 0] = 1.0
        pan, tilt = self.backend.head_joints()
        return np.concatenate([[est[0] / 8.0, est[1] / 8.0, math.cos(est[2]), math.sin(est[2])], r,
                               [pan / PAN_LIM, tilt / TILT_LIM, min(self.sim_time / self.max_sim_time, 1.0)]])

    def rel(self, xy):
        est = self.est
        c, s = math.cos(est[2]), math.sin(est[2])
        d = np.asarray(xy[:2]) - est[:2]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])

    def common_info(self) -> dict:
        le = self.loc_errors or [0.0]
        return {'sim_time': round(self.sim_time, 2), 'contact': self.contact, 'tipped': self.tipped,
                'fault': self.fault, 'loc_err_mean': round(float(np.mean(le)), 4),
                'loc_err_max': round(float(np.max(le)), 4), 'nav_pose_source': self.nav_pose_source,
                'backend': self.backend_name, 'physical': self.backend.physical,
                'stale': dict(getattr(self.backend, 'stale', {})), 'safety_halts': dict(self.halts),
                'min_clearance_m': round(float(self.min_clearance), 4), 'contact_with': self.contact_with}

    def close(self):
        self.backend.close()
