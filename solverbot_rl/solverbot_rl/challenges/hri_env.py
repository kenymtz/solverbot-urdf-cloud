"""Desafío Human-Robot Interaction (imagen 04) como entorno Gymnasium jerárquico.

Subtareas implementadas (verificación física con la verdad del simulador):
  recepción     ir a la puerta, mirar al invitado (< 12° a su cabeza) y preguntar nombre y bebida; el invitado responde
                con datos ESTRUCTURADOS (no hay audio ni reconocimiento de voz: interfaz de prueba).
  guiado        el invitado sigue al robot hasta la sala; se mide la fracción del avance con la cabeza hacia adelante.
  asiento       mirar un asiento (< 15°) y ofrecerlo; el invitado se sienta SOLO si el asiento está libre (verdad).
  presentación  mirar a cada invitado (< 12°, posición real; los invitados pueden cambiar de asiento) diciendo su
                nombre y bebida (de la memoria del robot) al otro invitado presente en la sala.
  bolsa         manipulador a < 10 cm de la mano del 2.º invitado (FK de articulaciones medidas), informar al anfitrión
                y seguirlo hasta el destino. NO implementado: tomar la bolsa ni depositarla (sin modelo de bolsa/agarre).
No implementado (opcional en las reglas): detectar un timbre real (aquí es un evento estructurado), abrir la puerta,
describir visualmente al primer invitado.

Acción: MultiDiscrete([N_SKILLS, 4]) = (habilidad, argumento: persona 0-3 por color o asiento 0-2).
"""
from __future__ import annotations

import math

import numpy as np

from gymnasium import spaces

from .arena import FOOTPRINT
from .base_env import ChallengeEnv
from .nav import foot_center
from .sensing import head_ray_angles, look_at_angles

SKILLS = ['wait', 'go_reception', 'go_living', 'look_person', 'ask_name_drink', 'guide', 'offer_seat', 'introduce',
          'scan_room', 'reach_hand', 'tell_host', 'follow']
N_ARG = 4
RECEPTION = (1.30, 2.59, math.pi)
LIVING_VIEW = (1.80, 2.05, -math.pi / 2)
GUEST_WAIT = (0.45, 2.59)
REACH_IN_BASE = np.array([0.25, 0.45])     # mano de la persona en el frame base_footprint al recibir la bolsa

# Niveles del curriculum: qué subtareas exige el éxito del episodio
LEVELS = [
    {'name': 'recepcion_1', 'guests': 1, 'need': ['ask']},
    {'name': 'acomodar_1', 'guests': 1, 'need': ['ask', 'seat']},
    {'name': 'acomodar_2', 'guests': 2, 'need': ['ask', 'seat']},
    {'name': 'presentar', 'guests': 2, 'need': ['ask', 'seat', 'intro']},
    {'name': 'cambio_asiento', 'guests': 2, 'need': ['ask', 'seat', 'intro'], 'seat_changes': True},
    {'name': 'bolsa', 'guests': 2, 'need': ['ask', 'seat', 'intro', 'reach', 'follow'], 'seat_changes': True},
]
REWARD = {'ask': 1.0, 'guide': 0.5, 'seat': 2.0, 'intro': 1.0, 'reach': 1.5, 'tell': 0.5, 'follow': 2.0,
          'success': 5.0, 'skill_fail': -0.2, 'wrong_seat': -0.5, 'wrong_intro': -0.5, 'time_per_s': -0.005,
          'collision': -5.0}
PERSON_FEATS = 12


class HRIEnv(ChallengeEnv):
    def __init__(self, backend='kinematic', level=0, **kw):
        super().__init__(backend=backend, level=level, max_skills=kw.pop('max_skills', 70), **kw)
        self.cfg = self.ch['hri']
        self.n_obs = 12 + 4 * PERSON_FEATS + 3 + 8
        self.observation_space = spaces.Box(-10.0, 10.0, (self.n_obs,), np.float32)
        self.action_space = spaces.MultiDiscrete([len(SKILLS), N_ARG])

    # ================================================================================================ escenario
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'level' in options:
            self.level = int(options['level'])
        rng = self.np_random
        self.L = LEVELS[min(self.level, len(LEVELS) - 1)]
        names = self.world.person_names
        order = list(rng.permutation(len(names)))
        self.host = names[order[0]]
        self.guests = [names[i] for i in order[1:1 + self.L['guests']]]
        pick_n = rng.choice(len(self.ch['names']), len(self.guests), replace=False)
        pick_d = rng.choice(len(self.ch['drinks']), len(self.guests), replace=False)
        self.persona = {g: {'name': self.ch['names'][a], 'drink': self.ch['drinks'][b]}
                        for g, a, b in zip(self.guests, pick_n, pick_d)}
        start = self.cfg['start']
        self._start((*start['xy'], start['yaw']), door_open=True)
        host_seat = self.world.seats[int(rng.integers(len(self.world.seats)))]['name']
        self.humans.sit(self.host, host_seat)
        # memoria del robot (lo que puede saber): roles conocidos por la consigna, datos del diálogo, acciones hechas
        self.mem = {'name': {}, 'drink': {}, 'guided': set(), 'offered': {}, 'intro': set(), 'reached': False,
                    'told': False, 'destination': None, 'followed': False, 'last_ok': 1.0}
        self.done = {'ask': set(), 'guide': set(), 'seat': set(), 'intro': set(), 'reach': False, 'tell': False,
                     'follow': False}
        self.arrived = []           # invitados que ya llegaron (timbre)
        self.bag_offered = False
        self.dest = None
        self._arrive(self.guests[0])
        self.gaze_samples = []
        return self._obs(), {'level': self.level, 'level_name': self.L['name'], **self.common_info()}

    def _arrive(self, g):
        """Un invitado llega: aparece afuera y entra hasta detrás de la puerta (timbre = evento estructurado)."""
        self.humans.place(g, *self.cfg['guest_arrival_xy'], 0.0)
        self.humans.walk_to(g, GUEST_WAIT)
        self.arrived.append(g)

    def _scenario_tick(self):
        """Eventos del escenario entre habilidades: llegada del 2.º invitado, cambios de asiento, oferta de la bolsa."""
        if len(self.arrived) < len(self.guests) and self.guests[0] in self.done['seat']:
            self._arrive(self.guests[1])
        if self.L.get('seat_changes') and self.np_random.random() < 0.12:
            seated = [g for g in self.guests if self.humans.people[g].state == 'seated']
            free = self.humans.free_seats()
            if seated and free and not self.bag_offered:
                g = seated[int(self.np_random.integers(len(seated)))]
                self.humans.walk_and_sit(g, free[int(self.np_random.integers(len(free)))],
                                         blocked=self._robot_block())
        if ('reach' in self.L['need'] and not self.bag_offered and len(self.done['intro']) == len(self.guests)
                and len(self.guests) == 2):
            g2 = self.guests[1]
            self.humans.walk_to(g2, (1.80, 1.35), blocked=self._robot_block())
            self.bag_offered = True

    # ================================================================================================ observación
    def _obs(self):
        names = self.world.person_names
        feats = []
        for n in names:
            t = self.track(n)
            f = np.zeros(PERSON_FEATS)
            if t is not None:
                rel = self.rel(t)
                f[0:4] = [1.0, rel[0] / 5, rel[1] / 5, np.linalg.norm(rel) / 5]
                f[4] = float(t[2] < 0.95)                          # torso bajo: sentado (percibido)
            f[5] = float(n == self.host)                          # rol por consigna
            f[6] = float(n in self.arrived)                       # llegó (timbre)
            f[7] = float(n in self.mem['name'])
            f[8] = float(n in self.mem['guided'])
            f[9] = float(n in self.mem['offered'])
            f[10] = float(n in self.mem['intro'])
            f[11] = float(n == (self.guests[1] if len(self.guests) > 1 else None) and self.bag_offered)
            feats.append(f)
        seats = []
        for s in self.world.seats:
            occ = 0.0
            for n in names:
                t = self.track(n, 10.0)
                if t is not None and np.linalg.norm(t[:2] - np.array(s['xy'])) < 0.4 and t[2] < 0.95:
                    occ = 1.0
            seats.append(occ)
        waiting = any(g in self.arrived and g not in self.mem['name'] for g in self.guests)
        mem = [float(waiting), float(self.bag_offered), float(self.mem['reached']), float(self.mem['told']),
               float(self.mem['followed']), len(self.mem['name']) / 2, len(self.mem['intro']) / 2, self.mem['last_ok']]
        o = np.concatenate([self.robot_obs(), np.concatenate(feats), seats, mem]).astype(np.float32)
        return np.clip(np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)

    # ================================================================================================ paso
    def step(self, action):
        skill, arg = SKILLS[int(action[0])], int(action[1])
        t0 = self.sim_time
        terms = {}
        res = self._run(skill, arg, terms)
        # creencia del robot sobre su última habilidad (no la verificación con la verdad, que solo va a la recompensa)
        self.mem['last_ok'] = float(res.get('belief', res.ok))
        if not res.ok and not self.failed():
            terms['skill_fail'] = REWARD['skill_fail']
        terms['time'] = REWARD['time_per_s'] * (self.sim_time - t0)
        if self.failed():
            terms['collision'] = REWARD['collision']
        self.skills_done += 1
        self._scenario_tick()
        success = self._success()
        if success:
            terms['success'] = REWARD['success']
        terminated = success or self.failed()
        truncated = not terminated and (self.skills_done >= self.max_skills or self.sim_time >= self.max_sim_time)
        info = {'skill': skill, 'arg': arg, 'skill_ok': res.ok, 'skill_reason': res.get('reason'),
                'terms': terms, 'is_success': success, 'level': self.level, 'progress': self._progress(),
                **self.common_info()}
        if terminated or truncated:
            info['episode_summary'] = {'success': success, 'level': self.level, 'skills': self.skills_done,
                                       'progress': self._progress(), **self.common_info()}
        return self._obs(), float(sum(terms.values())), terminated, truncated, info

    def _progress(self):
        return {'ask': len(self.done['ask']), 'guide': len(self.done['guide']), 'seat': len(self.done['seat']),
                'intro': len(self.done['intro']), 'reach': self.done['reach'], 'tell': self.done['tell'],
                'follow': self.done['follow'], 'guests': len(self.guests)}

    def _success(self):
        n = len(self.guests)
        need = self.L['need']
        ok = True
        ok &= 'ask' not in need or len(self.done['ask']) == n
        ok &= 'seat' not in need or len(self.done['seat']) == n
        ok &= 'intro' not in need or len(self.done['intro']) == n
        ok &= 'reach' not in need or self.done['reach']
        ok &= 'follow' not in need or self.done['follow']
        return bool(ok) and not self.failed()

    # ================================================================================================ habilidades
    def _person_target(self, arg):
        names = self.world.person_names
        return names[arg] if arg < len(names) else None

    def _run(self, skill, arg, terms):
        from .base_env import SkillResult
        hp = self.humans.people
        if skill == 'wait':
            return self.skill_wait(1.0)
        if skill == 'go_reception':
            return self.skill_navigate(RECEPTION[:2], RECEPTION[2])
        if skill == 'go_living':
            return self.skill_navigate(LIVING_VIEW[:2], LIVING_VIEW[2])
        if skill == 'scan_room':
            for pan in np.linspace(-1.3, 1.3, 7):
                self.skill_wait(0.4, head=(pan, 0.25))
            return SkillResult(ok=not self.failed())
        p = self._person_target(arg)
        if skill == 'look_person':
            t = self.track(p)
            if t is None:
                return SkillResult(ok=False, reason='no_percibida')
            self.look_at(self._head_from_track(t))
            pan, tilt = self.backend.head_joints()
            est_err = math.degrees(head_ray_angles(self.est, pan, tilt, self._head_from_track(t))[0])
            return SkillResult(ok=self.gaze_error_truth(self.humans.head_xyz(p)) < self.cfg['gaze_max_angle_deg'],
                               belief=est_err < self.cfg['gaze_max_angle_deg'])
        if skill == 'ask_name_drink':
            return self._ask(p, terms)
        if skill == 'guide':
            return self._guide(p, terms)
        if skill == 'offer_seat':
            return self._offer_seat(arg, terms)
        if skill == 'introduce':
            return self._introduce(p, terms)
        if skill == 'reach_hand':
            return self._reach(p, terms)
        if skill == 'tell_host':
            return self._tell(p, terms)
        if skill == 'follow':
            return self._follow(p, terms)
        return SkillResult(ok=False, reason='desconocida')

    def _head_from_track(self, t):
        return self.head_from_track(t)

    def _talk(self, p, seconds=2.0):
        return self.talk(p, seconds, self.cfg['gaze_max_angle_deg'])

    def _ask(self, p, terms):
        from .base_env import SkillResult
        if p is None or p not in self.arrived or p == self.host:
            return SkillResult(ok=False, reason='no_es_invitado_presente')
        if self.track(p, 3.0) is None:
            return SkillResult(ok=False, reason='no_percibida')
        frac, err = self._talk(p)
        dist = float(np.linalg.norm(self.humans.people[p].xy - self.ref_truth()))
        ok = frac >= 0.6 and err < self.cfg['gaze_max_angle_deg'] and dist < 1.6
        if ok:
            # respuesta ESTRUCTURADA del invitado (no audio)
            self.mem['name'][p] = self.persona[p]['name']
            self.mem['drink'][p] = self.persona[p]['drink']
            if p not in self.done['ask']:
                self.done['ask'].add(p)
                terms['ask'] = REWARD['ask']
        self.events.append({'t': self.sim_time, 'ev': 'ask', 'person': p, 'ok': ok, 'gaze_frac': round(frac, 2),
                            'gaze_err_deg': round(err, 1), 'dist': round(dist, 2)})
        return SkillResult(ok=ok, reason=None if ok else 'sin_contacto_visual_o_lejos', gaze_frac=frac)

    def _guide(self, p, terms):
        from .base_env import SkillResult
        if p not in self.mem['name'] or self.humans.people[p].state == 'seated':
            return SkillResult(ok=False, reason='no_saludado_o_ya_sentado')
        if np.linalg.norm(self.humans.people[p].xy - self.ref_truth()) > 2.5:
            return SkillResult(ok=False, reason='invitado_lejos')
        self.humans.follow(p)
        res = self.skill_navigate(LIVING_VIEW[:2], LIVING_VIEW[2], follower=p)
        self.humans.people[p].state = 'standing'
        in_room = self.arena.room_of(*self.humans.people[p].xy) == 'living_room'
        near = np.linalg.norm(self.humans.people[p].xy - self.ref_truth()) < 1.8
        ok = res.ok and in_room and near
        # creencia: el invitado se percibe cerca al llegar (no la verdad); girar a buscarlo si quedó detrás
        t0 = self.sim_time
        if res.ok and self.track(p, 1.0) is None:
            self.skill_search(p, restore=True)
        t = self.track(p, self.sim_time - t0 + 1.0)
        belief = res.ok and t is not None and np.linalg.norm(t[:2] - foot_center(self.est)) < 1.8
        if belief:
            self.mem['guided'].add(p)
        if ok:
            if p not in self.done['guide']:
                self.done['guide'].add(p)
                terms['guide'] = REWARD['guide']
        self.events.append({'t': self.sim_time, 'ev': 'guide', 'person': p, 'ok': ok,
                            'forward_gaze': round(res.get('forward_gaze', 0.0), 2)})
        return SkillResult(ok=ok, belief=belief, reason=res.get('reason') or (None if ok else 'invitado_no_llego'))

    def _offer_seat(self, k, terms):
        from .base_env import SkillResult
        if k >= len(self.world.seats):
            return SkillResult(ok=False, reason='asiento_invalido')
        seat = self.world.seats[k]
        waiting = [g for g in self.mem['guided'] if self.humans.people[g].state != 'seated'
                   and g not in self.done['seat']]
        if not waiting:
            return SkillResult(ok=False, reason='nadie_esperando_asiento')
        g = waiting[0]
        point = np.array([seat['xy'][0], seat['xy'][1], 0.8])
        self.look_at(point)
        err = self.gaze_error_truth(point)
        self.skill_wait(1.0, head=self.backend.head_joints())          # "por favor, siéntese allí"
        free = self.humans.occupant(seat['name']) is None
        ok = err < self.cfg['seat_indicate_max_angle_deg'] and free
        if ok:
            self.humans.walk_and_sit(g, seat['name'], blocked=self._robot_block())
            for _ in range(150):
                self.sim_step(0, 0, self.backend.head_joints())
                if self.humans.people[g].state == 'seated' or self.failed():
                    break
            ok = self.humans.people[g].state == 'seated' and self.humans.people[g].seat == seat['name']
        elif not free:
            terms['wrong_seat'] = REWARD['wrong_seat']
        if ok:
            self.mem['offered'][g] = seat['name']
            self.done['seat'].add(g)
            terms['seat'] = REWARD['seat']
        self.events.append({'t': self.sim_time, 'ev': 'seat', 'person': g, 'seat': seat['name'], 'ok': ok,
                            'free_truth': free, 'gaze_err_deg': round(err, 1)})
        return SkillResult(ok=ok, reason=None if ok else ('asiento_ocupado' if not free else 'no_indicado'))

    def _introduce(self, p, terms):
        from .base_env import SkillResult
        others = [g for g in self.guests if g != p]
        if p not in self.guests or not others or p not in self.mem['name']:
            return SkillResult(ok=False, reason='datos_desconocidos')
        if any(self.arena.room_of(*self.humans.people[g].xy) != 'living_room' for g in self.guests):
            return SkillResult(ok=False, reason='invitados_fuera_de_la_sala')
        if self.track(p, 6.0) is None:
            return SkillResult(ok=False, reason='no_percibida')
        frac, err = self._talk(p, 2.0)
        content_ok = (self.mem['name'][p] == self.persona[p]['name'] and self.mem['drink'][p] == self.persona[p]['drink'])
        ok = frac >= 0.6 and content_ok
        self.mem['intro'].add(p)                 # creencia: la presentación se realizó (la calidad la mide la verdad)
        if ok:
            if p not in self.done['intro']:
                self.done['intro'].add(p)
                terms['intro'] = REWARD['intro']
        else:
            terms['wrong_intro'] = REWARD['wrong_intro']
        self.events.append({'t': self.sim_time, 'ev': 'intro', 'person': p, 'ok': ok, 'gaze_frac': round(frac, 2)})
        return SkillResult(ok=ok, belief=True, reason=None if ok else 'mirada_incorrecta')

    @staticmethod
    def _rot(yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([[c, -s], [s, c]])

    @staticmethod
    def _hand_point(person_xy, phi):
        """Mano derecha extendida hacia la dirección phi (convención del protocolo de entrega)."""
        c, s = math.cos(phi), math.sin(phi)
        return np.asarray(person_xy) + np.array([0.28 * c + 0.12 * s, 0.28 * s - 0.12 * c])

    def _nearest_body_point(self, xy):
        """Punto de la huella real del robot más cercano a xy (para que la persona simulada se gire hacia él)."""
        tr = self.truth()
        R = self._rot(tr[2])
        local = R.T @ (np.asarray(xy) - tr[:2])
        x0, x1, y0, y1 = FOOTPRINT
        q = np.array([np.clip(local[0], x0, x1), np.clip(local[1], y0, y1)])
        return tr[:2] + R @ q

    def _reach(self, p, terms):
        from .base_env import SkillResult
        if not self.bag_offered or len(self.guests) < 2 or p != self.guests[1]:
            return SkillResult(ok=False, reason='no_ofrece_bolsa')
        t = self.track(p, 6.0)
        if t is None:
            return SkillResult(ok=False, reason='no_percibida')
        # Pose para tener la mano a la IZQUIERDA del robot (el Piper va montado en el costado izquierdo). Convención
        # del protocolo (conocida por el robot): la persona extiende la mano derecha, 0.28 m al frente y 0.12 m a su
        # derecha, hacia el robot. Se buscan direcciones de aproximación alrededor de la persona ESTIMADA cuya pose de
        # base quede libre (mapa estático + otras personas percibidas) y se elige la más cercana al robot.
        # esperar (mirando) a que la persona percibida quede quieta: no se aproxima a alguien que camina
        prev = t[:2].copy()
        for _ in range(16):
            self.skill_wait(0.5, head=look_at_angles(self.est, (*prev, 1.0)))
            cur = self.track(p, 2.0)
            if cur is not None and np.linalg.norm(cur[:2] - prev) < 0.06:
                t = cur
                break
            prev = cur[:2].copy() if cur is not None else prev
        else:
            return SkillResult(ok=False, reason='persona_en_movimiento')
        here = foot_center(self.est)
        others = [(*self.track(n)[:2], self.person_radius) for n in self.humans.people
                  if n != p and self.track(n, 8.0) is not None]
        best = None
        for phi in np.linspace(-math.pi, math.pi, 24, endpoint=False):
            hand, yaw = self._hand_point(t[:2], phi), phi + math.pi / 2
            base = hand - self._rot(yaw) @ REACH_IN_BASE
            dyn = others + [(t[0], t[1], self.person_radius)]
            if self.arena.footprint_collides(base[0], base[1], yaw, margin=0.04, dynamic=dyn):
                continue
            ref = base + self._rot(yaw) @ np.array([0.084, 0.0025])
            cost = float(np.linalg.norm(ref - here))
            if best is None or cost < best[0]:
                best = (cost, phi, yaw, ref)
        if best is None:
            return SkillResult(ok=False, reason='sin_pose_de_entrega')
        _, phi, yaw, ref = best
        nav = self.skill_navigate(ref, yaw)
        if not nav.ok:
            return SkillResult(ok=False, reason=nav.get('reason'))
        # comportamiento scriptado de la persona: se gira hacia el punto más cercano del cuerpo del robot y ofrece la bolsa
        self.humans.face(p, self._nearest_body_point(self.humans.people[p].xy))
        # observar ~1.5 s y promediar las detecciones (reduce el ruido de la estimación de la persona)
        samples = []
        for _ in range(5):
            self.skill_wait(0.3, head=look_at_angles(self.est, (*t[:2], 1.0)))
            cur = self.track(p, 0.35)
            if cur is not None:
                samples.append(cur[:2])
        pxy = np.mean(samples, axis=0) if samples else t[:2]
        # la persona mira al punto más cercano del cuerpo del robot (calculado con la pose ESTIMADA)
        tr_est = self.est
        R = self._rot(tr_est[2])
        local = R.T @ (pxy - tr_est[:2])
        x0, x1, y0, y1 = FOOTPRINT
        near = tr_est[:2] + R @ np.array([np.clip(local[0], x0, x1), np.clip(local[1], y0, y1)])
        phi_seen = math.atan2(near[1] - pxy[1], near[0] - pxy[0])
        hand_est = np.array([*self._hand_point(pxy, phi_seen), 1.0])
        r = self.skill_reach(hand_est)
        dist = float(np.linalg.norm(self.tcp_truth() - self.humans.hand_xyz(p)))
        ok = r.ok and dist < self.cfg['hand_reach_m']
        if ok:
            self.mem['reached'] = True
            if not self.done['reach']:
                self.done['reach'] = True
                terms['reach'] = REWARD['reach']
        self.events.append({'t': self.sim_time, 'ev': 'reach', 'person': p, 'ok': ok, 'tcp_hand_dist': round(dist, 3)})
        self.skill_wait(0.5)
        self.stow_arm()
        return SkillResult(ok=ok, reason=None if ok else 'mano_lejos', tcp_hand_dist=dist)

    def _tell(self, p, terms):
        from .base_env import SkillResult
        if p != self.host or not self.mem['reached'] or self.track(p, 8.0) is None:
            return SkillResult(ok=False, reason='no_corresponde')
        frac, err = self._talk(p, 2.0)
        ok = frac >= 0.6
        if ok and not self.done['tell']:
            self.done['tell'] = True
            self.mem['told'] = True
            terms['tell'] = REWARD['tell']
            dest = self.cfg['bag_destinations'][int(self.np_random.integers(len(self.cfg['bag_destinations'])))]
            loc = self.arena.locations[dest]
            self.dest = dest
            self.mem['destination'] = dest           # respuesta estructurada del anfitrión
            self.humans.people[p].state = 'standing'
            self.humans.people[p].z = 0.0
            hp = self.humans.people[p]
            if hp.seat:          # se levanta y queda de pie frente a su asiento
                st = self.humans.seats[hp.seat]
                hp.pose[:2] = (st['xy'][0] + 0.65 * math.cos(st['yaw']), st['xy'][1] + 0.65 * math.sin(st['yaw']))
            self.humans.people[p].seat = None
            self.humans.lead_to(p, (loc.x, loc.y), blocked=self._robot_block())
        return SkillResult(ok=ok)

    def _follow(self, p, terms):
        from .base_env import SkillResult
        if p != self.host or self.dest is None:
            return SkillResult(ok=False, reason='nadie_a_quien_seguir')
        fr = self.skill_follow(p, done_room=self.dest)
        if fr.ok:
            self.mem['followed'] = True          # creencia por percepción (persona quieta en el destino)
        hp = self.humans.people[p]
        ok = (hp.state == 'standing' and self.arena.room_of(*hp.xy) == self.dest
              and np.linalg.norm(hp.xy - self.ref_truth()) < 1.8 and not self.failed())
        if ok and not self.done['follow']:
            self.done['follow'] = True
            terms['follow'] = REWARD['follow']
        self.events.append({'t': self.sim_time, 'ev': 'follow', 'ok': ok, 'dest': self.dest})
        return SkillResult(ok=ok, belief=fr.ok, reason=None if ok else 'no_llego_con_el_anfitrion')


class HRIExpert:
    """Controlador de referencia (máquina de estados sobre la MEMORIA y las PERCEPCIONES del robot, no la verdad)."""

    def __init__(self, env: HRIEnv):
        self.env = env

    def act(self, obs=None):
        e = self.env
        S = SKILLS.index
        names = e.world.person_names
        idx = names.index
        guests = [g for g in e.arrived]
        ref = foot_center(e.est)
        for g in guests:
            if g not in e.mem['name']:
                if np.linalg.norm(ref - np.array(RECEPTION[:2])) > 0.3:
                    return np.array([S('go_reception'), 0])
                if e.track(g, 3.0) is None:
                    return np.array([S('look_person'), idx(g)]) if e.track(g) is not None else np.array([S('wait'), 0])
                return np.array([S('ask_name_drink'), idx(g)])
            if g not in e.mem['guided']:
                if e.track(g, 3.0) is None:
                    return np.array([S('scan_room'), 0])
                return np.array([S('guide'), idx(g)])
            if g not in e.mem['offered']:
                free = self._free_seat()
                if free is None:
                    return np.array([S('scan_room'), 0])
                return np.array([S('offer_seat'), free])
        if len(e.arrived) < len(e.guests):
            return np.array([S('go_reception'), 0]) if np.linalg.norm(ref - np.array(RECEPTION[:2])) > 0.3 \
                else np.array([S('wait'), 0])
        for g in e.guests:
            if 'intro' in e.L['need'] and g not in e.mem['intro']:
                if e.track(g, 2.0) is None or self._stale_scan:
                    self._stale_scan = False
                    return np.array([S('scan_room'), 0])
                self._stale_scan = True
                return np.array([S('introduce'), idx(g)])
        if 'reach' in e.L['need']:
            g2 = e.guests[1]
            if not e.bag_offered:
                return np.array([S('wait'), 0])
            if not e.mem['reached']:
                if e.track(g2, 3.0) is None:
                    return np.array([S('scan_room'), 0])
                return np.array([S('reach_hand'), idx(g2)])
            if not e.mem['told']:
                if e.track(e.host, 3.0) is None:
                    return np.array([S('scan_room'), 0])
                return np.array([S('tell_host'), idx(e.host)])
            if not e.mem['followed']:
                return np.array([S('follow'), idx(e.host)])
        return np.array([S('wait'), 0])

    _stale_scan = True

    def _free_seat(self):
        e = self.env
        for k, s in enumerate(e.world.seats):
            occ = False
            for n in e.world.person_names:
                t = e.track(n, 12.0)
                if t is not None and np.linalg.norm(t[:2] - np.array(s['xy'])) < 0.4:
                    occ = True
            if not occ:
                return k
        return None
