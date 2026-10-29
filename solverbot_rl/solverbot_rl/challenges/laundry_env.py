"""Desafío Doing Laundry (imagen 01) como entorno Gymnasium jerárquico — ETAPA INTERMEDIA.

Implementado y verificado con la verdad del simulador:
  inicio        el robot empieza afuera; detecta con el LiDAR que se abrió la puerta de entrada y navega al lavadero.
  identificar   lavarropas, mesa de doblado y cesto: mirar cada uno (< 15° a su centro) y estimar su posición con la
                percepción (error < 0.25 m). Es VERIFICACIÓN visual de muebles del mapa semántico conocido.
  localizar     prendas en el cesto: el robot se ubica junto al cesto (lado del brazo), lleva la CÁMARA DE MUÑECA
                sobre el cesto apuntando hacia abajo y estima la posición de cada prenda (error < 6 cm). La cámara de
                cabeza NO puede ver el fondo del cesto (HALLAZGO: cámara a 1.37 m centrada en la base, tilt máximo 45°:
                el piso visible más cercano está a ~0.6 m y el borde de 0.30 m tapa el fondo salvo a < ~0.5 m).
                Las prendas son PROXIES RÍGIDOS (placas delgadas de color, challenges.yaml: model: rigid_proxy).
                NO modelan tela.
  pre-agarre    punta de la pinza apuntando hacia abajo sobre la prenda, por encima del borde (5 cm en xy).

NO implementado (y por qué):
  recoger / transportar / depositar  dependen del agarre (validación local de Pick and Place) y, para una prenda
                real, de un modelo de tela. Además, HALLAZGO cinemático: con el elevador en 0 la punta de la pinza no baja
                de ~0.32 m (IK sobre el URDF), así que una prenda dentro de un cesto de piso (borde a 0.30 m, altura
                supuesta) o del tambor del lavarropas NO es alcanzable con la configuración actual.
  doblar / apilar                    requiere simulación de tela (Gazebo Harmonic no la tiene en este stack).
  abrir la puerta del lavarropas, extraer prendas del tambor (opcionales en las reglas).

Acción: MultiDiscrete([N_SKILLS, 3]); el argumento elige el mueble a identificar (0 lavarropas, 1 mesa, 2 cesto).
"""
from __future__ import annotations

import math

import numpy as np

from gymnasium import spaces

from .arena import FOOTPRINT
from .base_env import ARM_JOINTS, ChallengeEnv, SkillResult
from .nav import foot_center
from .sensing import CAM_HEIGHT, head_ray_angles

SKILLS = ['wait', 'wait_door', 'go_laundry', 'identify', 'approach_basket', 'look_into_basket', 'pregrasp']
TARGETS = ['washing_machine', 'laundry_table', 'basket']
LEVELS = [
    {'name': 'llegar', 'need': ['laundry']},
    {'name': 'identificar', 'need': ['laundry', 'identify']},
    {'name': 'localizar_prenda', 'need': ['laundry', 'identify', 'locate']},
    {'name': 'pre_agarre', 'need': ['laundry', 'identify', 'locate', 'pregrasp']},
]
REWARD = {'door': 0.5, 'laundry': 1.0, 'identify': 0.5, 'locate': 1.5, 'pregrasp': 2.0, 'success': 5.0,
          'skill_fail': -0.2, 'time_per_s': -0.005, 'collision': -5.0}
IDENTIFY_DEG = 15.0
IDENTIFY_POS_M = 0.25
LOCATE_M = 0.06
PREGRASP_XY_M = 0.05
PREGRASP_ABOVE_RIM = (0.02, 0.15)
PREGRASP_HEIGHT = 0.10      # punta de la pinza sobre el borde del cesto en el pre-agarre
# centro del cesto en el frame base al manipular: >= 0.58 m al costado (medio ancho del cesto 0.18 + huella izquierda
# 0.363 + holgura); la factibilidad del IK se verifica para cada candidato
REACH_CANDIDATES = [np.array([x, y]) for y in (0.58, 0.62) for x in (0.10, 0.20, 0.0, 0.30)]
# para una prenda puntual (re-aproximación de pre-agarre): puede quedar más cerca del costado que el centro del cesto
GARMENT_CANDIDATES = [np.array([x, y]) for y in (0.45, 0.50, 0.55, 0.60) for x in (0.05, 0.15, 0.0)]
DOWN = (0.0, 0.0, -1.0)
LOOK_HEIGHT = 0.60          # altura de la punta de la pinza al mirar dentro del cesto con la cámara de muñeca
WRIST_HALF_FOV = 0.55       # rad (modelo cinemático de la cámara de muñeca)


def hue_mask(rgb, target_rgb, hue_tol_deg=15.0, min_sat=0.4, min_val=0.2):
    """Segmentación por TONO y SATURACIÓN. Con la iluminación de Gazebo los colores se desaturan: el amarillo de la
    prenda queda en cromaticidad más cerca del cesto (marrón claro) que de su valor nominal, pero su tono (60°) y
    saturación (~0.55, contra ~0.25 del cesto y ~0.1 del piso) los separan."""
    img = rgb.astype(np.float32) / 255.0
    mx, mn = img.max(axis=2), img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    hue = np.degrees(np.arctan2(math.sqrt(3) * (g - b), 2 * r - g - b)) % 360
    t = np.asarray(target_rgb, float)
    th = math.degrees(math.atan2(math.sqrt(3) * (t[1] - t[2]), 2 * t[0] - t[1] - t[2])) % 360
    dh = np.abs((hue - th + 180) % 360 - 180)
    return (dh < hue_tol_deg) & (sat > min_sat) & (mx > min_val)


class LaundryEnv(ChallengeEnv):
    def __init__(self, backend='kinematic', level=0, **kw):
        super().__init__(backend=backend, level=level, max_skills=kw.pop('max_skills', 40), **kw)
        self.cfg = self.ch['laundry']
        b = self.cfg['basket']
        self.basket_xy = np.array(b['xy'], float)
        self.basket_size = np.array(b['size'], float)
        self.garments = [g['name'] for g in self.cfg['garments']]
        self.n_obs = 12 + 6 + 3 * 4 + len(self.garments) * 4 + 3
        self.observation_space = spaces.Box(-10.0, 10.0, (self.n_obs,), np.float32)
        self.action_space = spaces.MultiDiscrete([len(SKILLS), len(TARGETS)])
        # colores de los muebles en make_worlds.py (WHITE, WOOD y el del cesto) para el detector de Gazebo
        self.colors.update({'furn:washing_machine': (0.92, 0.92, 0.92), 'furn:laundry_table': (0.55, 0.36, 0.20),
                            'furn:basket': (0.55, 0.45, 0.30)})

    # ================================================================================================ escenario
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'level' in options:
            self.level = int(options['level'])
        rng = self.np_random
        self.L = LEVELS[min(self.level, len(LEVELS) - 1)]
        # prendas (proxies) dentro del cesto: posición y orientación aleatorias, apiladas sin interpenetrar
        hx, hy = self.basket_size[0] / 2 - 0.12, self.basket_size[1] / 2 - 0.10
        objs = {}
        for k, g in enumerate(self.garments):
            objs[g] = (self.basket_xy[0] + rng.uniform(-hx, hx), self.basket_xy[1] + rng.uniform(-hy, hy),
                       0.015 + 0.01 + 0.021 * k)
        self.garment_yaw = {g: float(rng.uniform(-math.pi, math.pi)) for g in self.garments}
        sp = self.ch['spawn']
        self._start((sp['xy'][0] + 0.084 * math.cos(sp['yaw']), sp['xy'][1] + 0.084 * math.sin(sp['yaw']), sp['yaw']),
                    door_open=False, objects=objs)
        self.door_opens_at = float(rng.uniform(2.0, 6.0))
        self.mem = {'door_open': False, 'identified': {}, 'garments': {}, 'at_basket': False, 'looked': False,
                    'pregrasp': False, 'last_ok': 1.0}
        self.done = {'door': False, 'laundry': False, 'identify': set(), 'locate': set(), 'pregrasp': False}
        self._gates = None
        return self._obs(), {'level': self.level, 'level_name': self.L['name'], 'garment_model': 'rigid_proxy',
                             **self.common_info()}

    def _entity_poses(self) -> dict:
        ents = super()._entity_poses()
        for g, yaw in getattr(self, 'garment_yaw', {}).items():
            if g in ents and len(ents[g]) == 4:
                # el modelo SDF de la prenda ya lleva su altura dentro del cesto (make_worlds: 0.025 + 0.021 k): el
                # origen del modelo va en z = 0
                ents[g] = (ents[g][0], ents[g][1], yaw, 0.0)
        return ents

    def sim_step(self, v=0.0, w=0.0, head=None, perceive_every=3):
        super().sim_step(v, w, head, perceive_every)
        if not self.door_open and self.sim_time >= self.door_opens_at:
            self.open_entrance_door()

    # ------------------------------------------------------------------------------------------------ percepción
    def _target_truth(self, name) -> np.ndarray:
        """Centro de la cara superior (verdad) del mueble / cesto."""
        if name == 'basket':
            return np.array([*self.basket_xy, self.basket_size[2]])
        fu = self.world.furniture[name]
        return np.array([fu['xy'][0], fu['xy'][1], fu['size'][2]])

    def _target_box(self, name, pad=0.12):
        if name == 'basket':
            c, s = self.basket_xy, self.basket_size
        else:
            fu = self.world.furniture[name]
            c, s = np.array(fu['xy']), np.array(fu['size'])
        return (c[0] - s[0] / 2 - pad, c[0] + s[0] / 2 + pad, c[1] - s[1] / 2 - pad, c[1] + s[1] / 2 + pad), s[2]

    def _visible_over_rim(self, xyz, cam=None) -> bool:
        """Modelo cinemático: una prenda dentro del cesto solo se ve si el rayo cámara->prenda pasa sobre el borde."""
        if cam is None:
            tr = self.truth()
            cam = np.array([tr[0], tr[1], CAM_HEIGHT])
        d = np.asarray(xyz) - cam
        (x0, x1, y0, y1), rim = self._target_box('basket', pad=0.0)
        tmin, tmax = 0.0, 1.0
        for k, (lo, hi) in enumerate(((x0, x1), (y0, y1))):
            if abs(d[k]) < 1e-9:
                continue
            t1, t2 = (lo - cam[k]) / d[k], (hi - cam[k]) / d[k]
            tmin = max(tmin, min(t1, t2))
        z_entry = cam[2] + d[2] * tmin
        return z_entry > rim

    def perceivable_objects_truth(self) -> dict:
        out = {}       # cámara de cabeza (las prendas se miden con la de muñeca: _wrist_detect)
        for g in self.garments:
            if g in self.objects_truth and self._visible_over_rim(self.objects_truth[g]):
                out[g] = np.array(self.objects_truth[g])
        for t in TARGETS:
            out['furn:' + t] = self._target_truth(t)
        return out

    def detection_gates(self) -> dict:
        return dict(self._gates or {})

    def detect_names(self):
        return list(self._gates or {})

    # ================================================================================================ observación
    def _obs(self):
        status = [float(self.mem['door_open']), float(self.door_scan_open()),
                  float(self.arena.room_of(*foot_center(self.est)) == 'laundry'), float(self.mem['at_basket']),
                  float(self.mem['looked']), float(self.mem['pregrasp'])]
        ident = []
        for t in TARGETS:
            e = self.mem['identified'].get(t)
            if e is None:
                ident += [0.0, 0.0, 0.0, 0.0]
            else:
                r = self.rel(e)
                ident += [1.0, r[0] / 5, r[1] / 5, e[2] / 2]
        garm = []
        for g in self.garments:
            e = self.mem['garments'].get(g)
            if e is None:
                garm += [0.0, 0.0, 0.0, 0.0]
            else:
                r = self.rel(e)
                garm += [1.0, r[0] / 5, r[1] / 5, e[2]]
        lvl = np.array([self.mem['last_ok'], 0.0, 0.0])
        o = np.concatenate([self.robot_obs(), status, ident, garm, lvl])
        return np.clip(np.nan_to_num(o.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)

    # ================================================================================================ paso
    def step(self, action):
        skill, arg = SKILLS[int(action[0])], int(action[1])
        t0 = self.sim_time
        terms = {}
        res = self._run(skill, arg, terms)
        self.mem['last_ok'] = float(res.get('belief', res.ok))
        if not res.ok and not self.failed():
            terms['skill_fail'] = REWARD['skill_fail']
        terms['time'] = REWARD['time_per_s'] * (self.sim_time - t0)
        if self.failed():
            terms['collision'] = REWARD['collision']
        self.skills_done += 1
        success = self._success()
        if success:
            terms['success'] = REWARD['success']
        terminated = success or self.failed()
        truncated = not terminated and (self.skills_done >= self.max_skills or self.sim_time >= self.max_sim_time)
        info = {'skill': skill, 'arg': arg, 'skill_ok': res.ok, 'skill_reason': res.get('reason'), 'terms': terms,
                'is_success': success, 'level': self.level, 'progress': self._progress(),
                'garment_model': 'rigid_proxy', **self.common_info()}
        if terminated or truncated:
            info['episode_summary'] = {'success': success, 'level': self.level, 'skills': self.skills_done,
                                       'progress': self._progress(), 'garment_model': 'rigid_proxy',
                                       **self.common_info()}
        return self._obs(), float(sum(terms.values())), terminated, truncated, info

    def _progress(self):
        return {'door': self.done['door'], 'laundry': self.done['laundry'], 'identify': sorted(self.done['identify']),
                'locate': sorted(self.done['locate']), 'pregrasp': self.done['pregrasp']}

    def _success(self):
        need = self.L['need']
        ok = 'laundry' not in need or self.done['laundry']
        ok &= 'identify' not in need or len(self.done['identify']) == len(TARGETS)
        ok &= 'locate' not in need or len(self.done['locate']) >= 1
        ok &= 'pregrasp' not in need or self.done['pregrasp']
        return bool(ok) and not self.failed()

    # ================================================================================================ habilidades
    def _run(self, skill, arg, terms):
        if skill == 'wait':
            return self.skill_wait(1.0)
        if skill == 'wait_door':
            return self.skill_wait_door(terms, REWARD['door'])
        if skill == 'go_laundry':
            return self._go_laundry(terms)
        if skill == 'identify':
            return self._identify(TARGETS[arg], terms)
        if skill == 'look_into_basket':
            return self._look_into_basket(terms)
        if skill == 'approach_basket':
            return self._approach_basket()
        if skill == 'pregrasp':
            return self._pregrasp(terms)
        return SkillResult(ok=False, reason='desconocida')

    def _go_laundry(self, terms):
        loc = self.arena.locations['laundry']
        res = self.skill_navigate((loc.x, loc.y), None)
        if res.ok and self.arena.room_of(*self.ref_truth()) == 'laundry' and not self.done['laundry']:
            self.done['laundry'] = True
            terms['laundry'] = REWARD['laundry']
        return res

    def _identify(self, name, terms):
        # el apuntado usa el mapa semántico conocido a priori (plano 10 / challenges.yaml); el error se mide contra la
        # verdad del simulador (en este mundo coinciden: los muebles son estáticos)
        c = expected = self._target_truth(name)
        box, h = self._target_box(name)
        key = 'furn:' + name
        self._gates = {key: ((h - 0.10, h + 0.05), box)}
        try:
            self.tracks.pop(key, None)
            self.look_at(expected, settle=6, turn=True)
            for _ in range(4):
                self.sim_step(0, 0, self.backend.head_joints(), perceive_every=1)
        finally:
            self._gates = None
        t = self.track(key, 1.0)
        if t is None:
            return SkillResult(ok=False, reason='no_detectado')
        pan, tilt = self.backend.head_joints()
        gaze = math.degrees(head_ray_angles(self.truth(), pan, tilt, c)[0])
        err = float(np.linalg.norm(t[:2] - c[:2]))
        self.mem['identified'][name] = t
        ok = gaze < IDENTIFY_DEG and err < IDENTIFY_POS_M
        if ok and name not in self.done['identify']:
            self.done['identify'].add(name)
            terms['identify'] = REWARD['identify']
        self.events.append({'t': self.sim_time, 'ev': 'identify', 'target': name, 'ok': ok,
                            'gaze_deg': round(gaze, 1), 'pos_err': round(err, 3)})
        return SkillResult(ok=ok, belief=True, pos_err=err)

    def _basket_est(self):
        """El cesto es un objeto ESTÁTICO del mapa semántico (conocido a priori, igual que los muebles): la planificación
        usa su posición en el mapa; la percepción solo lo verifica (identify)."""
        return self.basket_xy.copy()

    def _wrist_detect(self, not_before=None) -> dict:
        """Prendas vistas por la cámara de muñeca -> {nombre: xyz en el mapa (con la pose ESTIMADA de la base)}."""
        out = {}
        est = self.est
        if hasattr(self.backend, 'wrist_rgb'):                  # Gazebo: imagen real + rayo al plano del fondo
            w = self.backend.wrist_rgb()
            if w is None or (not_before is not None and w[3] < not_before):
                return out                                   # imagen anterior a que el brazo se detuviera
            rgb, k, T, _ = w
            c, s = math.cos(est[2]), math.sin(est[2])
            base = np.eye(4)
            base[:2, :2] = [[c, -s], [s, c]]
            base[:2, 3] = est[:2]
            box, _ = self._target_box('basket', pad=0.0)
            cam = base @ T
            for i, g in enumerate(self.garments):
                v, u = np.nonzero(hue_mask(rgb, self.colors[g]))
                if u.size < 15:
                    continue
                ray = cam[:3, :3] @ np.array([(u.mean() - k[0, 2]) / k[0, 0], (v.mean() - k[1, 2]) / k[1, 1], 1.0])
                if ray[2] > -1e-3:
                    continue
                plane_z = 0.035 + 0.021 * i                      # altura aproximada de la cara superior de la prenda
                p = cam[:3, 3] + ray * (plane_z - cam[2, 3]) / ray[2]
                if box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]:
                    out[g] = p
            return out
        # cinemático: cámara en arm_camera_link (FK de las articulaciones), cono de visión y oclusión por el borde
        js = {**dict(zip(ARM_JOINTS, self.arm_q)), 'torso_lift_joint': 0.0}
        T = self.chain.link_pose('arm_camera_link', js)
        tr = self.truth()
        R = self._rot(tr[2])
        cam = np.array([*(tr[:2] + R @ T[:2, 3]), T[2, 3]])
        axis = np.array([*(R @ T[:2, 2]), T[2, 2]])
        rng = self.detector.rng
        for g in self.garments:
            xyz = np.array(self.objects_truth[g])
            d = xyz - cam
            if math.acos(np.clip(axis @ d / np.linalg.norm(d), -1, 1)) > WRIST_HALF_FOV:
                continue
            if not self._visible_over_rim(xyz, cam):
                continue
            # medido en el frame de la base y llevado al mapa con la pose estimada
            local = R.T @ (xyz[:2] - tr[:2]) + rng.normal(0, 0.008, 2)
            out[g] = np.array([*(est[:2] + self._rot(est[2]) @ local), xyz[2]])
        return out

    def _look_into_basket(self, terms):
        if not self.mem['at_basket']:
            return SkillResult(ok=False, reason='no_en_pose')
        b = self._basket_est()
        r = self.skill_reach(np.array([b[0], b[1], LOOK_HEIGHT]), approach_axis=DOWN)
        if not r.ok:
            self.stow_arm()
            return SkillResult(ok=False, reason=r.get('reason') or 'brazo')
        dets = {}
        settled = getattr(self.backend, '_clock', 0.0)      # reloj del simulador (el de los sellos de las imágenes)
        for _ in range(10):
            self.sim_step(0, 0, self.backend.head_joints())
            for g, p in self._wrist_detect(not_before=settled + 0.05).items():
                dets.setdefault(g, []).append(p)
        self.mem['looked'] = True
        for g, ps in dets.items():
            p = np.mean(ps, axis=0)
            self.mem['garments'][g] = p
            err = float(np.linalg.norm(p[:2] - np.array(self.objects_truth[g][:2])))
            ok = err < LOCATE_M
            if ok and g not in self.done['locate']:
                self.done['locate'].add(g)
                terms['locate'] = terms.get('locate', 0.0) + REWARD['locate'] * (1.0 if len(self.done['locate']) == 1
                                                                                 else 0.25)
            self.events.append({'t': self.sim_time, 'ev': 'locate', 'garment': g, 'ok': ok, 'err': round(err, 3)})
        return SkillResult(ok=bool(dets), belief=bool(dets), reason=None if dets else 'sin_prendas_visibles')

    def _rot(self, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s], [s, c]])

    def _pregrasp_target(self):
        """Prenda localizada cuyo punto de pre-agarre (sobre el borde, pinza hacia abajo) es alcanzable por IK desde la
        pose ESTIMADA actual; si ninguna lo es, la más cercana (el IK lo reportará fuera de alcance)."""
        if not self.mem['garments']:
            return None, None
        est = self.est
        R = self._rot(est[2])
        cands = []
        for g, e in self.mem['garments'].items():
            p = np.array([e[0], e[1], self.basket_size[2] + PREGRASP_HEIGHT])
            local = np.array([*(R.T @ (p[:2] - est[:2])), p[2]])
            # alcanzable primero; entre esas, la de ARRIBA (la de abajo queda parcialmente tapada y su estimación se
            # sesga hacia la parte visible)
            cands.append((not self._ik_ok(local), -round(float(e[2]), 3), float(np.hypot(*local[:2])), g, p))
        cands.sort(key=lambda c: c[:3])
        return cands[0][3], cands[0][4]

    def _ik_ok(self, local_xyz, cache={}):
        key = tuple(np.round(local_xyz, 3))
        if key not in cache:
            from ..kinematics import solve_ik
            from .base_env import STOW
            _, err = solve_ik(self.chain, 'tcp_link', ARM_JOINTS, np.array(STOW), np.asarray(local_xyz), None,
                              {'torso_lift_joint': 0.0}, restarts=4, reg=0.002, approach_axis=np.array(DOWN))
            cache[key] = err < 0.03
        return cache[key]

    def _approach_basket(self, target_xy=None, z_check=LOOK_HEIGHT):
        """Base con el cesto a la IZQUIERDA (lado del brazo). Junto al cesto no se puede girar en el lugar (radio de giro
        0.50 m), así que se elige una pose final libre, una pose previa 0.6 m detrás donde SÍ se puede girar, y el
        último tramo se hace en línea recta. Todo con el mapa y la pose estimada."""
        p = self._basket_est() if target_xy is None else np.asarray(target_xy, float)
        here = foot_center(self.est)
        best = None
        for yaw in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            R = self._rot(yaw)
            fwd = R[:, 0]
            for rib in (REACH_CANDIDATES if target_xy is None else GARMENT_CANDIDATES):
                base = p - R @ rib
                # pose final y tramo recto con holgura EXACTA >= 5 cm (la grilla de 2 cm aceptaba tramos que rozaban
                # la pared del cesto: contacto detectado por la verificación exacta, laundry N3 semilla 1000)
                plan = self.final_approach_plan(base, yaw, margin=0.05)
                if plan is None:
                    continue
                ref, pre = plan
                if not self._ik_ok(np.array([*rib, z_check])):
                    continue
                if self.arena.plan(here, pre) is None:
                    continue
                cost = float(np.linalg.norm(pre - here)) + 0.3 * float(np.linalg.norm(rib - np.array([0.1, 0.5])))
                if best is None or cost < best[0]:
                    best = (cost, yaw, ref, pre)
                break
        if best is None:
            return SkillResult(ok=False, reason='sin_pose_de_manipulacion')
        _, yaw, ref, pre = best
        if self.mem['at_basket']:                      # junto al cesto no se puede girar: salir en recta primero
            self.stow_arm()
            back = self.skill_reverse(0.6)
            self.mem['at_basket'] = False
            if not back.ok:
                return back
        res = self.skill_navigate(pre, yaw)
        if not res.ok:
            return res
        res = self.skill_straight(ref, yaw)
        self.mem['at_basket'] = bool(res.ok)
        return res

    def _pregrasp(self, terms):
        g, p = self._pregrasp_target()
        if g is None or not self.mem['at_basket']:
            return SkillResult(ok=False, reason='no_en_pose')
        R = self._rot(self.est[2])
        if not self._ik_ok(np.array([*(R.T @ (p[:2] - self.est[:2])), p[2]])):
            # ninguna prenda alcanzable desde este lado: re-aproximarse por el lado más cercano a la prenda elegida
            res = self._approach_basket(target_xy=p[:2], z_check=p[2])
            if not res.ok:
                return SkillResult(ok=False, reason=res.get('reason') or 'sin_pose_de_pre_agarre')
        r = self.skill_reach(p, approach_axis=DOWN)
        tcp = self.tcp_truth()
        gt = np.array(self.objects_truth[g])
        dxy = float(np.linalg.norm(tcp[:2] - gt[:2]))
        above = tcp[2] - self.basket_size[2]
        ok = r.ok and dxy < PREGRASP_XY_M and PREGRASP_ABOVE_RIM[0] <= above <= PREGRASP_ABOVE_RIM[1]
        self.mem['pregrasp'] = bool(r.ok)
        if ok and not self.done['pregrasp']:
            self.done['pregrasp'] = True
            terms['pregrasp'] = REWARD['pregrasp']
        self.events.append({'t': self.sim_time, 'ev': 'pregrasp', 'garment': g, 'ok': ok, 'tcp_xy_err': round(dxy, 3),
                            'tcp_above_rim': round(float(above), 3), 'ik': r.get('reason')})
        self.skill_wait(0.5)
        self.stow_arm()
        return SkillResult(ok=ok, belief=bool(r.ok), reason=r.get('reason'))



class LaundryExpert:
    """Secuencia de referencia (experto GUIONADO) sobre la memoria del robot."""

    def __init__(self, env: LaundryEnv):
        self.env = env

    def act(self, obs=None):
        e = self.env
        S = SKILLS.index
        if not e.mem['door_open']:
            return np.array([S('wait_door'), 0])
        if e.arena.room_of(*foot_center(e.est)) != 'laundry':
            return np.array([S('go_laundry'), 0])
        need = e.L['need']
        if 'identify' in need:
            for k, t in enumerate(TARGETS):
                if t not in e.mem['identified']:
                    return np.array([S('identify'), k])
        if 'locate' in need:
            if not e.mem['at_basket']:
                return np.array([S('approach_basket'), 0])
            if not e.mem['garments']:
                return np.array([S('look_into_basket'), 0])
        if 'pregrasp' in need and not e.mem['pregrasp']:
            return np.array([S('pregrasp'), 0])
        return np.array([S('wait'), 0])
