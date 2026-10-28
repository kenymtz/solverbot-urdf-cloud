"""Desafío General Purpose Service Robot (imagen 02) como entorno Gymnasium jerárquico.

Separación pedida por la consigna:
  INTERPRETACIÓN  gpsr_grammar.CommandParser: texto (gramática cerrada) -> Command estructurado. El robot trabaja
                  SOLO con lo interpretado; el escenario conserva la verdad (Command generado) para evaluar.
  PLANIFICACIÓN   GPSRPlanner (experto de referencia): Command(s) -> secuencia de habilidades; en modo "los tres
                  juntos" reordena los comandos por cercanía (planificación combinada).
  EJECUCIÓN       habilidades físicas de ChallengeEnv (navegación con pose estimada, búsqueda, mirada, seguir/guiar).
  EVALUACIÓN      _complete_*: verificación con la VERDAD del simulador (posición real, mirada real, respuesta
                  correcta según el estado real), nunca con la memoria del robot.

Flujo (reglas): el robot empieza afuera; cuando se abre la puerta (evento del escenario, detectado con el LiDAR)
navega al Instruction Point; el operador da tres comandos; el robot decide si recibirlos de a uno (debe volver al
operador después de cada tarea) o juntos (puede combinarlos); al finalizar vuelve al Instruction Point.

NO implementado: comandos que requieren agarre (bring: se interpretan y se rechazan), abrir puertas o cajones,
características de personas más allá del color de camiseta. Sin audio: el texto del comando es la interfaz.

Acción: MultiDiscrete([N_SKILLS, 4]); el argumento es una habitación (search_room) o un índice de comando
(select_command); el resto de las habilidades actúa sobre el COMANDO ACTUAL interpretado.
"""
from __future__ import annotations

import math

import numpy as np

from gymnasium import spaces

from .base_env import PERSON_SOCIAL, ChallengeEnv, SkillResult
from .gpsr_grammar import (FURNITURE, NEEDS_GRASP, OBJECT_SURFACES, OBJECTS, ROOMS as GRAMMAR_ROOMS, VERBS,
                           CommandGenerator, CommandParser)
from .nav import foot_center
from .sensing import look_at_angles

SKILLS = ['wait', 'wait_door', 'go_instruction', 'receive_one', 'receive_all', 'select_command', 'go_target',
          'search_room', 'approach_person', 'greet', 'inspect', 'report', 'follow', 'guide']
ROOM_IDS = ['living_room', 'kitchen', 'laundry', 'bedroom']
N_ARG = 4
REPORT_VERBS = {'where_person', 'object_yesno', 'count_people'}
LEVELS = [
    {'name': 'ir_1', 'n': 1, 'verbs': ['go']},
    {'name': 'simple_1', 'n': 1, 'verbs': ['go', 'greet', 'object_yesno']},
    {'name': 'simple_3', 'n': 3, 'verbs': ['go', 'greet', 'object_yesno', 'count_people']},
    {'name': 'completo_3', 'n': 3, 'verbs': ['go', 'greet', 'object_yesno', 'count_people', 'where_person', 'follow',
                                             'guide']},
]
REWARD = {'door': 0.5, 'instruction': 0.5, 'command': 3.0, 'report_ok': 1.0, 'report_wrong': -1.0, 'success': 5.0,
          'skill_fail': -0.2, 'time_per_s': -0.005, 'collision': -5.0}
GREET_GAZE_DEG = 12.0


class GPSREnv(ChallengeEnv):
    def __init__(self, backend='kinematic', level=0, **kw):
        super().__init__(backend=backend, level=level, max_skills=kw.pop('max_skills', 80),
                         max_sim_time=kw.pop('max_sim_time', 1500.0), **kw)
        self.cfg = self.ch['gpsr']
        ip = self.cfg['instruction_point']
        self.instr = (ip['xy'][0], ip['xy'][1], ip['yaw'])
        self.operator_xy = np.array(self.cfg['operator_xy'], float)
        self.parser = CommandParser()
        self.locations = [k for k in self.arena.locations if k in GRAMMAR_ROOMS or k in FURNITURE]
        self.n_obs = 12 + 8 + 3 * 6 + len(VERBS) + 4 + 4 + 3 + 4 * 4
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
        self.operator = names[int(rng.integers(len(names)))]
        others = [n for n in names if n != self.operator]
        gen = CommandGenerator(self.L['verbs'], rng, self.locations)
        for attempt in range(30):
            # comandos consistentes: cada persona aparece en a lo sumo un comando
            self.commands, used = [], set()
            while len(self.commands) < self.L['n']:
                c = gen.sample()
                if c.person and (c.person in used or c.person == self.operator):
                    continue
                if c.verb == 'guide' and c.location in (c.room,) + tuple(
                        f for f, v in self.world.furniture.items() if v['room'] == c.room):
                    continue                              # guiar dentro de la misma habitación: trivial
                used.add(c.person)
                self.commands.append(c)
            objects = self._place_objects(rng)
            sp = self.ch['spawn']
            self._start((sp['xy'][0] + 0.084 * math.cos(sp['yaw']), sp['xy'][1] + 0.084 * math.sin(sp['yaw']),
                         sp['yaw']), door_open=False, objects=objects)
            try:
                self._place_people(rng, others)
                break
            except RuntimeError:
                continue                                  # escenario sin lugar para las personas: se vuelve a sortear
        else:
            raise RuntimeError('no se pudo generar un escenario GPSR válido')
        self.door_opens_at = float(rng.uniform(2.0, 6.0))
        self.humans.place(self.operator, *self.operator_xy, -math.pi / 2)
        # estado del robot (creencias) y del escenario (verdad)
        self.mode = None
        self.received: list = []                        # comandos interpretados (lo que sabe el robot)
        self.given = 0
        self.cur = None
        self.mem = {'door_open': False, 'answers': {}, 'reported': set(), 'believed_done': set(), 'counts': {},
                    'last_ok': 1.0}
        self.done = {'door': False, 'instruction': False, 'complete': set(), 'reports': {}}
        self.count_truth = {}
        return self._obs(), {'level': self.level, 'level_name': self.L['name'],
                             'commands_text': [c.text for c in self.commands], **self.common_info()}

    def _place_objects(self, rng) -> dict:
        """Objetos del vocabulario GPSR: a lo sumo uno por mueble. object_yesno: 50 % sobre el mueble preguntado."""
        surfaces = list(OBJECT_SURFACES)
        rng.shuffle(surfaces)
        placed, free_s = {}, [s for s in surfaces]
        for c in self.commands:
            if c.verb == 'object_yesno' and c.obj not in placed:
                if rng.random() < 0.5 and c.furniture in free_s:
                    s = c.furniture
                else:
                    s = next(x for x in free_s if x != c.furniture)
                free_s.remove(s)
                placed[c.obj] = s
        for o in OBJECTS:
            if o not in placed and rng.random() < 0.5 and free_s:
                placed[o] = free_s.pop()
        out = {}
        for o, s in placed.items():
            x, y, z = self.world.sample_on_surface(s, rng, margin=0.12)
            out[o] = (x, y, z + self.world.objects[o]['size'][2] / 2)
        self.object_on = placed
        return out

    def _free_spot(self, rng, room, avoid=()):
        """Lugar libre para una persona: lejos de muebles, de las puertas, de las poses de aproximación, del operador y
        de otras personas, sin cortar el acceso a ninguna habitación. Segunda pasada con márgenes menores."""
        a = self.arena
        x0, y0 = self.world.spec['rooms'][room]['min']
        x1, y1 = self.world.spec['rooms'][room]['max']
        for clear, door_d, loc_d in ((0.55, 1.3, 0.9), (0.30, 1.0, 0.6)):
            for _ in range(300):
                x, y = rng.uniform(x0 + 0.35, x1 - 0.35), rng.uniform(y0 + 0.35, y1 - 0.35)
                i, j = a.to_ij(x, y)
                if a.nav_dist[i, j] < clear or any(np.hypot(x - p[0], y - p[1]) < 1.0 for p in avoid):
                    continue
                if any(np.hypot(x - d.center[0], y - d.center[1]) < door_d for d in a.doors if d.usable):
                    continue                          # no bloquear los accesos a las puertas
                if any(np.hypot(x - l.x, y - l.y) < loc_d for k, l in a.locations.items() if k not in ROOM_IDS):
                    continue                          # ni las poses de aproximación a los muebles
                if np.hypot(x - self.instr[0], y - self.instr[1]) < 1.2 or \
                        np.hypot(x - self.operator_xy[0], y - self.operator_xy[1]) < 1.0:
                    continue
                if not self._keeps_reachable([*avoid, (x, y)]):
                    continue                          # ni cortar el paso a alguna habitación
                return x, y
        raise RuntimeError(f'sin lugar libre en {room}')

    def _keeps_reachable(self, people_xy) -> bool:
        dyn = [(x, y, self.person_radius + PERSON_SOCIAL) for x, y in people_xy]
        for r in ROOM_IDS:
            loc = self.arena.locations[r]
            if self.arena.plan(self.instr[:2], (loc.x, loc.y), dynamic=dyn) is None:
                return False
        return True

    def _place_people(self, rng, others):
        """Primero las personas que nombran los comandos (en su habitación); después distractores y personas a contar,
        que se omiten si la habitación ya no tiene lugar libre."""
        spots = []
        need = {c.person: c for c in self.commands if c.person}
        count_rooms = [c.room for c in self.commands if c.verb == 'count_people']
        for p in sorted(others, key=lambda n: n not in need):
            if p in need:
                c = need[p]
                room = c.room or ROOM_IDS[int(rng.integers(len(ROOM_IDS)))]
            elif count_rooms and rng.random() < 0.6:
                room = count_rooms[0]
            elif rng.random() < 0.4:
                room = ROOM_IDS[int(rng.integers(len(ROOM_IDS)))]
            else:
                continue
            try:
                x, y = self._free_spot(rng, room, spots)
            except RuntimeError:
                if p in need:
                    raise
                continue
            spots.append((x, y))
            self.humans.place(p, x, y, float(rng.uniform(-math.pi, math.pi)))

    def _scenario_tick(self):
        if not self.door_open and self.sim_time >= self.door_opens_at:
            self.open_entrance_door()                  # el árbitro abre la puerta

    def sim_step(self, v=0.0, w=0.0, head=None, perceive_every=3):
        super().sim_step(v, w, head, perceive_every)
        self._scenario_tick()

    def perceivable_objects_truth(self) -> dict:
        return {o: np.array(xyz) for o, xyz in self.objects_truth.items()}

    def detection_gates(self) -> dict:
        g = getattr(self, '_gates', None)
        return dict(g) if g else {}

    def detect_names(self):
        return [n for n in self.world.person_names if self.humans.people[n].state != 'parked'] + \
            (list(self._gates) if getattr(self, '_gates', None) else [])

    # ================================================================================================ observación
    def _cmd(self):
        return self.received[self.cur] if self.cur is not None and self.cur < len(self.received) else None

    def _obs(self):
        est = self.est
        c = self._cmd()
        at_instr = float(np.linalg.norm(foot_center(est) - np.array(self.instr[:2])) < 0.4)
        status = [float(self.mem['door_open']), float(self.door_scan_open()), at_instr,
                  float(self.mode == 'one'), float(self.mode == 'all'), len(self.received) / 3.0,
                  len(self.mem['believed_done']) / 3.0, float(self.cur is not None)]
        slots = []
        for k in range(3):
            if k < len(self.received):
                ck = self.received[k]
                slots += [1.0, float(k in self.mem['believed_done']), float(ck.verb in REPORT_VERBS),
                          float(k in self.mem['answers']), float(k in self.mem['reported']), float(ck is None)]
            else:
                slots += [0.0] * 6
        verb = np.zeros(len(VERBS))
        tgt = np.zeros(4)
        room = np.zeros(4)
        flags = np.zeros(3)
        if c is not None:
            verb[VERBS.index(c.verb)] = 1.0
            loc = self._target_xy(c)
            if loc is not None:
                r = self.rel(loc)
                tgt[:3] = [1.0, r[0] / 8, r[1] / 8]
            t = self.track(c.person, 5.0) if c.person else None
            if t is not None:
                r = self.rel(t)
                tgt[3] = np.linalg.norm(r) / 5
            if c.room:
                room[ROOM_IDS.index(c.room)] = 1.0
            flags = [float(t is not None), float(self.cur in self.mem['answers']),
                     float(c.verb in NEEDS_GRASP)]
        people = []
        for n in self.world.person_names:
            t = self.track(n)
            f = np.zeros(4)
            if t is not None:
                r = self.rel(t)
                f[:] = [1.0, r[0] / 5, r[1] / 5, float(n == self.operator)]
            people.append(f)
        o = np.concatenate([self.robot_obs(), status, slots, verb, tgt, room, flags, np.concatenate(people)])
        return np.clip(np.nan_to_num(o.astype(np.float32), nan=0.0, posinf=10.0, neginf=-10.0), -10, 10)

    def _target_xy(self, c):
        name = c.location if c.verb in ('go', 'guide') else c.furniture if c.verb == 'object_yesno' else None
        if name is None or name not in self.arena.locations:
            return None
        loc = self.arena.locations[name]
        return np.array([loc.x, loc.y])

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
                'is_success': success, 'level': self.level, 'progress': self._progress(), **self.common_info()}
        if terminated or truncated:
            info['episode_summary'] = {'success': success, 'level': self.level, 'skills': self.skills_done,
                                       'progress': self._progress(), 'commands': [c.text for c in self.commands],
                                       'verbs': [c.verb for c in self.commands], **self.common_info()}
        return self._obs(), float(sum(terms.values())), terminated, truncated, info

    def _progress(self):
        return {'door': self.done['door'], 'instruction': self.done['instruction'], 'received': len(self.received),
                'complete': sorted(self.done['complete']), 'n': len(self.commands), 'mode': self.mode,
                'reports': dict(self.done['reports'])}

    def _at_instruction_truth(self):
        return float(np.linalg.norm(self.ref_truth() - np.array(self.instr[:2]))) < 0.5

    def _success(self):
        return (len(self.done['complete']) == len(self.commands) and self.given == len(self.commands)
                and self._at_instruction_truth() and not self.failed())

    def _complete(self, k, terms):
        if k not in self.done['complete']:
            self.done['complete'].add(k)
            terms['command'] = terms.get('command', 0.0) + REWARD['command']

    # ================================================================================================ habilidades
    def _run(self, skill, arg, terms):
        if skill == 'wait':
            return self.skill_wait(1.0)
        if skill == 'wait_door':
            return self.skill_wait_door(terms, REWARD['door'])
        if skill == 'go_instruction':
            return self._go_instruction(terms)
        if skill == 'receive_one':
            return self._receive(False)
        if skill == 'receive_all':
            return self._receive(True)
        if skill == 'select_command':
            if arg >= len(self.received) or self.received[arg] is None:
                return SkillResult(ok=False, reason='comando_inexistente')
            self.cur = arg
            return SkillResult(ok=True)
        if skill == 'search_room':
            return self._search_room(ROOM_IDS[arg], terms)
        c = self._cmd()
        if c is None:
            return SkillResult(ok=False, reason='sin_comando')
        if c.requires_grasp:
            return SkillResult(ok=False, reason='requiere_agarre_no_implementado')
        if skill == 'go_target':
            return self._go_target(c, terms)
        if skill == 'approach_person':
            return self._approach(c)
        if skill == 'greet':
            return self._greet(c, terms)
        if skill == 'inspect':
            return self._inspect(c)
        if skill == 'report':
            return self._report(terms)
        if skill == 'follow':
            return self._follow(c, terms)
        if skill == 'guide':
            return self._guide(c, terms)
        return SkillResult(ok=False, reason='desconocida')

    # ---------------------------------------------------------------------------------------------- puerta / operador
    def _go_instruction(self, terms):
        res = self.skill_navigate(self.instr[:2], self.instr[2])
        if res.ok and not self.done['instruction'] and self._at_instruction_truth():
            self.done['instruction'] = True
            terms['instruction'] = REWARD['instruction']
        return res

    def _near_operator(self):
        """Robot cerca del Instruction Point (estimación) y operador percibido."""
        return (np.linalg.norm(foot_center(self.est) - np.array(self.instr[:2])) < 0.5
                and self.track(self.operator, 30.0) is not None)

    def _receive(self, all_three):
        if not self._near_operator():
            self.look_at((*self.operator_xy, 1.4))
            if not self._near_operator():
                return SkillResult(ok=False, reason='lejos_del_operador')
        if all_three and self.given > 0 or (not all_three and self.mode == 'all'):
            return SkillResult(ok=False, reason='modo_incompatible')
        if not all_three and self.cur is not None and self.cur not in self.mem['believed_done'] \
                and self.cur not in self.mem['reported']:
            return SkillResult(ok=False, reason='comando_actual_pendiente')
        n = len(self.commands) - self.given if all_three else 1
        if n <= 0:
            return SkillResult(ok=False, reason='sin_comandos')
        self.mode = 'all' if all_three else 'one'
        self.talk(self.operator, 1.5, 30.0)
        for _ in range(n):
            truth = self.commands[self.given]
            parsed = self.parser.parse(truth.text)          # interpretación: SOLO el texto llega al robot
            self.received.append(parsed)
            self.events.append({'t': self.sim_time, 'ev': 'command', 'k': self.given, 'text': truth.text,
                                'parsed_ok': parsed is not None and parsed.key() == truth.key()})
            self.given += 1
        self.cur = len(self.received) - n
        return SkillResult(ok=self.received[self.cur] is not None)

    # ---------------------------------------------------------------------------------------------- tareas
    def _go_target(self, c, terms):
        xy = self._target_xy(c) if c.verb != 'guide' else None
        if xy is None and c.room:
            loc = self.arena.locations[c.room]
            xy = np.array([loc.x, loc.y])
        if xy is None:
            return SkillResult(ok=False, reason='sin_destino')
        name = c.location if c.verb == 'go' else c.furniture if c.verb == 'object_yesno' else c.room
        res = self.skill_navigate(xy, self.arena.locations[name].yaw)
        if c.verb == 'go' and res.ok:
            self.mem['believed_done'].add(self.cur)
            k = self.cur
            if float(np.linalg.norm(self.ref_truth() - xy)) < 0.3 and self._truth_cmd(k).verb == 'go' \
                    and self._truth_cmd(k).location == c.location:
                self._complete(k, terms)
        return res

    def _truth_cmd(self, k):
        return self.commands[k]

    def _search_room(self, room, terms):
        loc = self.arena.locations[room]
        nav = self.skill_navigate((loc.x, loc.y), None)
        if not nav.ok and nav.get('reason') in ('colision', 'vuelco'):
            return nav
        t0 = self.sim_time
        seen_before = {n: self.tracks.get(n, (None, -1))[1] for n in self.world.person_names}
        self.skill_search(None, max_turn=2 * math.pi, w=0.5)
        seen = [n for n in self.world.person_names if n != self.operator
                and self.tracks.get(n, (None, -1))[1] > max(seen_before[n], t0)
                and self.arena.room_of(*self.tracks[n][0][:2]) == room]
        self.mem['counts'][room] = len(seen)
        # verdad para evaluar el conteo: personas (no estacionadas) en la habitación durante la búsqueda
        self.count_truth[room] = sum(1 for n, p in self.humans.people.items() if n != self.operator
                                     and p.state != 'parked' and self.arena.room_of(*p.xy) == room)
        c = self._cmd()
        if c is not None and c.verb == 'count_people' and c.room == room:
            self.mem['answers'][self.cur] = len(seen)
        if c is not None and c.verb == 'where_person' and c.person in seen:
            self.mem['answers'][self.cur] = room
        return SkillResult(ok=not self.failed(), belief=True, seen=len(seen))

    def _approach(self, c):
        """Ir a ~1 m de la persona percibida: punto libre de su misma habitación (anillo de 0.9-1.2 m) con ruta."""
        t = self.track(c.person, 20.0) if c.person else None
        if t is None:
            return SkillResult(ok=False, reason='persona_no_percibida')
        a = self.arena
        here = foot_center(self.est)
        room = a.room_of(*t[:2])
        others = [(*self.track(n, 20.0)[:2], self.person_radius + PERSON_SOCIAL) for n in self.humans.people
                  if n != c.person and self.track(n, 20.0) is not None]
        cands = []
        for rad in (1.0, 1.2, 0.9, 1.4, 1.6, 1.8):
            for ang in np.linspace(-math.pi, math.pi, 16, endpoint=False):
                q = t[:2] + rad * np.array([math.cos(ang), math.sin(ang)])
                i, j = a.to_ij(*q, a.plan_res)
                # punto donde se pueda girar en el lugar (la siguiente navegación empieza girando)
                if not (0 <= i < a.rot_free.shape[0] and 0 <= j < a.rot_free.shape[1]) or not a.rot_free[i, j]:
                    continue
                if a.room_of(*q) != room or any(np.hypot(q[0] - x, q[1] - y) < r + 0.4 for x, y, r in others):
                    continue
                cands.append((float(np.linalg.norm(q - here)), q))
            if cands:
                break
        for _, q in sorted(cands, key=lambda x: x[0])[:4]:
            if a.plan(here, q) is None:
                continue
            yaw = math.atan2(t[1] - q[1], t[0] - q[0])
            res = self.skill_navigate(q, None, max_time=90.0, ignore=c.person,
                                      head_fn=lambda pose: look_at_angles(pose, (t[0], t[1], 1.3)))
            if res.ok:
                self.skill_turn(yaw)
                self.look_at(self.head_from_track(t), turn=True)
            return res
        return SkillResult(ok=False, reason='sin_ruta')

    def _reacquire(self, person):
        """Track reciente de la persona: si es viejo, volver a mirar donde se la vio por última vez."""
        t = self.track(person, 20.0)
        if t is None:
            return None
        if self.track(person, 2.0) is None:
            self.look_at(self.head_from_track(t), settle=6, turn=True)
        return self.track(person, 1.0)

    def _greet(self, c, terms):
        if self._reacquire(c.person) is None:
            return SkillResult(ok=False, reason='persona_no_percibida')
        frac, err = self.talk(c.person, 2.0, GREET_GAZE_DEG)
        dist = float(np.linalg.norm(self.humans.people[c.person].xy - self.ref_truth()))
        k = self.cur
        tc = self._truth_cmd(k)
        ok = frac >= 0.6 and dist < 2.0 and tc.verb == 'greet' and tc.person == c.person
        if ok:
            self._complete(k, terms)
        belief = self.last_talk_belief >= 0.6
        if belief:
            self.mem['believed_done'].add(k)
        self.events.append({'t': self.sim_time, 'ev': 'greet', 'k': k, 'ok': ok, 'gaze_frac': round(frac, 2),
                            'gaze_frac_belief': round(self.last_talk_belief, 2)})
        return SkillResult(ok=ok, belief=belief)

    def _inspect(self, c):
        fu = self.world.furniture[c.furniture]
        sx, sy = fu['size'][0] / 2 + 0.1, fu['size'][1] / 2 + 0.1
        box = (fu['xy'][0] - sx, fu['xy'][0] + sx, fu['xy'][1] - sy, fu['xy'][1] + sy)
        h = self.world.surface_height(c.furniture)
        self._gates = {o: ((h - 0.05, h + 0.35), box) for o in OBJECTS}
        found = False
        try:
            for dx in (0.0, -0.15, 0.15):
                self.look_at((fu['xy'][0] + dx, fu['xy'][1], h), settle=6)
                for _ in range(4):
                    self.sim_step(0, 0, self.backend.head_joints(), perceive_every=1)
                t = self.track(c.obj, 1.0)
                if t is not None and box[0] <= t[0] <= box[1] and box[2] <= t[1] <= box[3]:
                    found = True
                    break
        finally:
            self._gates = None
        self.mem['answers'][self.cur] = found
        return SkillResult(ok=True, belief=True, found=found)

    def _report(self, terms):
        if not self._near_operator():
            return SkillResult(ok=False, reason='lejos_del_operador')
        pend = [k for k in self.mem['answers'] if k not in self.mem['reported']]
        if self.mode == 'one':
            pend = [k for k in pend if k == self.cur]
        if not pend:
            return SkillResult(ok=False, reason='nada_que_informar')
        self.talk(self.operator, 1.0 + 0.5 * len(pend), 30.0)
        ok_all = True
        for k in pend:
            tc = self._truth_cmd(k)
            ans = self.mem['answers'][k]
            if tc.verb == 'where_person':
                truth = self.arena.room_of(*self.humans.people[tc.person].xy)
            elif tc.verb == 'object_yesno':
                truth = self.object_on.get(tc.obj) == tc.furniture
            elif tc.verb == 'count_people':
                truth = self.count_truth.get(tc.room)
            else:
                truth = None
            ok = truth is not None and ans == truth
            ok_all &= ok
            self.done['reports'][k] = {'answer': ans, 'truth': truth, 'ok': ok}
            terms['report'] = terms.get('report', 0.0) + (REWARD['report_ok'] if ok else REWARD['report_wrong'])
            if ok:
                self._complete(k, terms)
            self.mem['reported'].add(k)
            self.mem['believed_done'].add(k)
        return SkillResult(ok=ok_all, belief=True)

    def _follow(self, c, terms):
        t = self._reacquire(c.person)
        if t is None:
            return SkillResult(ok=False, reason='persona_no_percibida')
        p = self.humans.people[c.person]
        k = self.cur
        tc = self._truth_cmd(k)
        if tc.verb == 'follow' and tc.person == c.person and p.state == 'standing' \
                and np.linalg.norm(p.xy - self.ref_truth()) < 2.5:
            # la persona camina a un lugar de otra habitación (escenario)
            dest_room = [r for r in ROOM_IDS if r != self.arena.room_of(*p.xy)]
            dr = dest_room[int(self.np_random.integers(len(dest_room)))]
            others = [tuple(q.xy) for n, q in self.humans.people.items() if n != c.person and q.state != 'parked']
            self.follow_dest = self._free_spot(self.np_random, dr, others)
            self.humans.lead_to(c.person, self.follow_dest, blocked=self._robot_block())
        res = self.skill_follow(c.person)
        ok = (getattr(self, 'follow_dest', None) is not None and p.state == 'standing'
              and np.linalg.norm(p.xy - np.array(self.follow_dest)) < 0.2
              and np.linalg.norm(p.xy - self.ref_truth()) < 1.8 and not self.failed())
        if ok:
            self._complete(k, terms)
        if res.ok:
            self.mem['believed_done'].add(k)
        self.events.append({'t': self.sim_time, 'ev': 'follow', 'k': k, 'ok': ok})
        return SkillResult(ok=ok, belief=res.ok, reason=res.get('reason'))

    def _guide(self, c, terms):
        t = self._reacquire(c.person)
        if t is None or np.linalg.norm(t[:2] - foot_center(self.est)) > 2.0:
            return SkillResult(ok=False, reason='persona_lejos_o_no_percibida')
        k = self.cur
        tc = self._truth_cmd(k)
        p = self.humans.people[c.person]
        if tc.verb == 'guide' and tc.person == c.person and np.linalg.norm(p.xy - self.ref_truth()) < 2.5:
            self.humans.follow(c.person)                 # la persona acepta seguir al robot
        loc = self.arena.locations[c.location]
        res = self.skill_navigate((loc.x, loc.y), loc.yaw, follower=c.person)
        self.skill_wait(2.0, head=(0.0, 0.2))
        if p.state == 'following':
            p.state = 'standing'
        dest = np.array([loc.x, loc.y])
        ok = (res.ok and np.linalg.norm(p.xy - dest) < 1.5 and self.arena.room_of(*p.xy) == loc.room
              and not self.failed())
        if ok:
            self._complete(k, terms)
        t = self.track(c.person, 3.0)
        belief = res.ok and t is not None and np.linalg.norm(t[:2] - foot_center(self.est)) < 1.8
        if res.ok:
            self.mem['believed_done'].add(k)
        self.events.append({'t': self.sim_time, 'ev': 'guide', 'k': k, 'ok': ok})
        return SkillResult(ok=ok, belief=belief, reason=res.get('reason'))


# ==================================================================================================== planificador
class GPSRPlanner:
    """Planificador de referencia (experto GUIONADO, no aprendido): decide recibir los comandos de a uno o juntos y
    los descompone en habilidades usando SOLO la interpretación del texto y las percepciones del robot."""

    def __init__(self, env: GPSREnv, batch: bool | None = None):
        self.env = env
        self.batch = batch
        self.searched: dict[int, list] = {}

    def act(self, obs=None):
        e = self.env
        S = SKILLS.index
        if not e.mem['door_open']:
            return np.array([S('wait_door'), 0])
        at_instr = np.linalg.norm(foot_center(e.est) - np.array(e.instr[:2])) < 0.4
        if e.given == 0:
            if not at_instr:
                return np.array([S('go_instruction'), 0])
            batch = self.batch if self.batch is not None else len(e.commands) > 1
            return np.array([S('receive_all' if batch else 'receive_one'), 0])
        # comando a ejecutar: el actual si no terminó; en modo "juntos", el pendiente más cercano
        pending = [k for k, c in enumerate(e.received) if c is not None and not c.requires_grasp
                   and k not in e.mem['believed_done'] and k not in e.mem['answers']]
        if e.cur is not None and e.cur in pending:
            k = e.cur
        elif pending:
            k = min(pending, key=lambda j: self._cost(j))
            return np.array([S('select_command'), k])
        else:
            k = None
        if k is None:
            # todo ejecutado (o con respuesta): volver, informar, recibir el siguiente, terminar en el IP
            unreported = [j for j in e.mem['answers'] if j not in e.mem['reported']]
            if not at_instr:
                return np.array([S('go_instruction'), 0])
            if unreported:
                if e.mode == 'one' and e.cur not in unreported:
                    return np.array([S('select_command'), unreported[0]])
                return np.array([S('report'), 0])
            if e.given < len(e.commands):
                return np.array([S('receive_one'), 0])
            return np.array([S('wait'), 0])
        c = e.received[k]
        if c.verb == 'go':
            return np.array([S('go_target'), 0])
        if c.verb == 'object_yesno':
            loc = e.arena.locations[c.furniture]
            if np.linalg.norm(foot_center(e.est) - np.array([loc.x, loc.y])) > 0.3:
                return np.array([S('go_target'), 0])
            return np.array([S('inspect'), 0])
        if c.verb == 'count_people':
            return np.array([S('search_room'), ROOM_IDS.index(c.room)])
        # verbos con persona
        t = e.track(c.person, 20.0)
        if t is None or (c.room and e.arena.room_of(*t[:2]) != c.room):
            rooms = [c.room] if c.room else self._room_order(k)
            tried = self.searched.setdefault(k, [])
            todo = [r for r in rooms if r not in tried] or rooms
            tried.append(todo[0])
            return np.array([S('search_room'), ROOM_IDS.index(todo[0])])
        if c.verb == 'where_person':
            # la respuesta la deja search_room al verla; si se la vio antes, buscar en esa habitación
            return np.array([S('search_room'), ROOM_IDS.index(e.arena.room_of(*t[:2]))])
        if np.linalg.norm(t[:2] - foot_center(e.est)) > 1.4:
            return np.array([S('approach_person'), 0])
        return np.array([S({'greet': 'greet', 'follow': 'follow', 'guide': 'guide'}[c.verb]), 0])

    def _cost(self, k):
        e = self.env
        c = e.received[k]
        xy = e._target_xy(c)
        if xy is None and c.room:
            loc = e.arena.locations[c.room]
            xy = np.array([loc.x, loc.y])
        return float(np.linalg.norm(foot_center(e.est) - xy)) if xy is not None else 50.0

    def _room_order(self, k):
        e = self.env
        here = foot_center(e.est)
        return sorted(ROOM_IDS, key=lambda r: np.hypot(e.arena.locations[r].x - here[0],
                                                       e.arena.locations[r].y - here[1]))

