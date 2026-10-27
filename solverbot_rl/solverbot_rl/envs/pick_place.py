"""Entorno Gymnasium de Pick and Place para SolverBot v3.

Observación (sin ground truth del simulador salvo object_source=privileged, que queda registrado en `info`):
    state: [q brazo normalizado (6), pinza (1), TCP por FK (3), eje z del TCP (3),
            objeto estimado - TCP (3), detección válida (1), zona objetivo - TCP (3), acción previa (7), meta one-hot (4)]
    wrist/head (opcional, solo Gazebo): imagen uint8 (3, H, W)
Acción (7, en [-1, 1]): incrementos articulares del brazo (x arm_delta_max rad) + pinza (-1 abrir ... +1 cerrar).
"""
from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium import spaces

from .. import task as T
from ..config import load_config, load_urdf
from ..kinematics import UrdfChain, solve_ik

ARM_POINT_LINKS = ('link4', 'link5', 'link6', 'gripper_base', 'link7', 'link8', 'tcp_link')
STATE_DIM = 6 + 1 + 3 + 3 + 3 + 1 + 3 + 7 + len(T.GOALS)


class SolverBotPickPlaceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, backend: str = 'kinematic', config_path: str | None = None, level: int = 0,
                 overrides: dict | None = None, world: str = 'pick_place_training'):
        super().__init__()
        self.cfg = load_config(config_path, overrides)
        c, r, o = self.cfg['control'], self.cfg['robot'], self.cfg['observation']
        self.levels = self.cfg['curriculum']['levels']
        self.level = int(level)
        self.arm_joints = r['arm_joints']
        self.dt, self.delta = c['dt'], c['arm_delta_max']
        self.max_steps = c['max_episode_steps']
        self.object_source = o['object_source']
        self.image_keys = tuple(o['images'])
        self.image_size = tuple(o['image_size'])
        self.backend_name = backend
        if backend == 'kinematic':
            from ..backends.kinematic import KinematicBackend
            if self.image_keys:
                raise ValueError('El backend cinemático no tiene cámaras: use observation.images: []')
            self.chain = UrdfChain(load_urdf())
            self.backend = KinematicBackend(self.chain, self.cfg)
        elif backend == 'gazebo':
            from ..backends.gazebo import GazeboBackend
            self.backend = GazeboBackend(self.cfg, world=world, images=self.image_keys,
                                         perception=self.object_source == 'perception')
            self.chain = self.backend.chain
        else:
            raise ValueError(backend)
        self.lo, self.hi = self.chain.limits(self.arm_joints)

        state = spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)
        if self.image_keys:
            h, w = self.image_size
            self.observation_space = spaces.Dict(
                {'state': state, **{k: spaces.Box(0, 255, (3, h, w), np.uint8) for k in self.image_keys}})
        else:
            self.observation_space = state
        self.action_space = spaces.Box(-1.0, 1.0, (7,), np.float32)

    # --------------------------------------------------------------------------------------- utilidades
    def set_level(self, level: int):
        self.level = int(np.clip(level, 0, len(self.levels) - 1))

    def _joints(self):
        return self.backend.joint_positions()

    def _task_state(self, js) -> T.TaskState:
        tcp = self.chain.link_pose('tcp_link', js)
        pts = np.array([self.chain.link_pose(lk, js)[:3, 3] for lk in ARM_POINT_LINKS])
        obj = self.backend.object_pose_truth()
        vel = float(np.linalg.norm(obj - self._prev_obj) / self.dt) if self._prev_obj is not None else 0.0
        self._prev_obj = obj
        return T.TaskState(tcp_pos=tcp[:3, 3], tcp_z=tcp[:3, 2], obj_pos=obj, gripper_q=float(js.get('joint7', 0.0)),
                           gripper_cmd_closed=self._grip_closed, arm_points=pts, joints_at_limit=self._at_limit,
                           obj_vel=vel, gripper_target=self._g_cmd, gripper_q8=js.get('joint8'))

    def _object_obs(self):
        if self.object_source == 'privileged':
            return self.backend.object_pose_truth(), True
        if self.object_source == 'perception':
            return self.backend.object_estimate()
        return np.zeros(3), False

    def _obs(self, s: T.TaskState):
        js = self._joints()
        q = np.array([js[j] for j in self.arm_joints])
        qn = 2 * (q - self.lo) / (self.hi - self.lo) - 1
        est, valid = self._object_obs()
        if valid:
            self._obj_est = est
        goal = np.zeros(len(T.GOALS))
        goal[T.GOALS.index(self.params.goal)] = 1.0
        target = np.array([*self.params.target_xy, self.params.table_height + self.params.obj_half_height])
        state = np.concatenate([qn, [s.gripper_q / self.cfg['robot']['gripper_open'] * 2 - 1], s.tcp_pos, s.tcp_z,
                                (self._obj_est - s.tcp_pos) if self._obj_est is not None else np.zeros(3), [float(valid)],
                                target - s.tcp_pos, self._prev_action, goal]).astype(np.float32)
        if not self.image_keys:
            return state
        out = {'state': state}
        imgs = self.backend.images()
        h, w = self.image_size
        for k in self.image_keys:
            im = imgs.get(k)
            if im is None:
                out[k] = np.zeros((3, h, w), np.uint8)
            else:   # submuestreo por vecino más cercano (sin dependencias extra)
                yy = (np.arange(h) * im.shape[0] / h).astype(int)
                xx = (np.arange(w) * im.shape[1] / w).astype(int)
                out[k] = np.ascontiguousarray(im[yy][:, xx].transpose(2, 0, 1))
        return out

    def _start_q(self, obj_xy):
        r = self.cfg['robot']
        home = np.array(r['home'], float)
        if self.levels[self.level]['start'] != 'pregrasp':
            return home
        # Arranque cerca del objeto (habilidades de agarre aisladas): pinza 4 cm sobre el objeto, IK guionada
        obj = np.array([*obj_xy, self.params.table_height + self.params.obj_half_height])
        ax = np.array([np.sin(np.radians(20)), 0.0, -np.cos(np.radians(20))])
        q, err = solve_ik(self.chain, 'tcp_link', self.arm_joints, home, obj - ax * 0.04, None,
                          {'torso_lift_joint': r['lift']}, approach_axis=ax, rot_weight=1.0, restarts=4)
        return q if err < 0.01 else home

    # --------------------------------------------------------------------------------------- API Gymnasium
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options and 'level' in options:
            self.set_level(options['level'])
        lv = self.levels[self.level]
        r, wd = self.cfg['robot'], self.cfg['world']
        self.params = T.TaskParams.from_config(self.cfg, lv['goal'])
        rng = self.np_random
        obj_xy = np.asarray(wd['object']['xy'], float) + rng.uniform(-1, 1, 2) * lv['object_xy_range']
        q0 = self._start_q(obj_xy)
        self.backend.reset(rng, obj_xy, float(rng.uniform(-0.3, 0.3)), q0, r['gripper_open'], r['lift'], r['head'])
        js = self._joints()
        self._q_cmd = np.array([js[j] for j in self.arm_joints])
        self._grip_closed = False
        self._g_cmd = float(r['gripper_open'])
        self._at_limit = 0
        self._prev_action = np.zeros(7, np.float32)
        self._prev_obj = None
        self._obj_est = None
        self.flags = T.EpisodeFlags()
        self.steps = 0
        self.ep_return = 0.0
        self.ep_terms: dict[str, float] = {}
        self.params.target_xy = self.backend.world_xy_to_base(wd['target_zone']['xy'])
        self._state = self._task_state(js)
        self.flags.best_phi = T.potential(self._state, self.params)
        return self._obs(self._state), {'level': self.level, 'goal': lv['goal'], 'object_source': self.object_source}

    def step(self, action):
        a = np.clip(np.asarray(action, np.float32), -1, 1)
        target = self._q_cmd + a[:6] * self.delta
        self._at_limit = int(np.sum((target <= self.lo + 1e-4) | (target >= self.hi - 1e-4)))
        self._q_cmd = np.clip(target, self.lo, self.hi)
        # Pinza: a[6] = -1 -> abierta (precierre gripper_open), +1 -> cerrada. El objetivo se mueve con rampa
        # limitada (gripper_rate) para un cierre gradual y simétrico; nunca se ordena el límite físico de 35 mm.
        r = self.cfg['robot']
        g_open, g_closed = r['gripper_open'], r['gripper_closed']
        g_req = g_closed + (g_open - g_closed) * (1.0 - float(a[6])) / 2.0
        rate = r.get('gripper_rate', g_open)
        self._g_cmd = float(np.clip(self._g_cmd + np.clip(g_req - self._g_cmd, -rate, rate), g_closed, g_open))
        self._grip_closed = bool(a[6] > 0)
        self.backend.command(self._q_cmd, self._g_cmd)
        self.backend.advance(self.dt)
        prev = self._state
        self._state = self._task_state(self._joints())
        reward, terms, terminated, stalled = T.step_reward(prev, self._state, a, self._prev_action, self.params,
                                                           self.flags)
        self._prev_action = a
        self.steps += 1
        truncated = (not terminated) and (stalled or self.steps >= self.max_steps)
        self.ep_return += reward
        for k, v in terms.items():
            self.ep_terms[k] = self.ep_terms.get(k, 0.0) + v
        info = {'is_success': bool(terms['success']), 'terms': terms, 'level': self.level,
                'goal': self.params.goal, 'object_source': self.object_source,
                'grasped': self.flags.grasped_once, 'lifted': self.flags.lifted_once,
                'gripper_target': self._g_cmd, 'mimic_error': T.mimic_error(self._state),
                'stale_images': getattr(self.backend, 'stale_images', 0),
                'stale_joint_states_total': getattr(self.backend, 'stale_joint_states', 0)}
        if terminated or truncated:
            info['episode_summary'] = {'return': self.ep_return, 'length': self.steps, 'success': info['is_success'],
                                       'level': self.level, 'goal': self.params.goal, 'stalled': stalled,
                                       'grasped': self.flags.grasped_once, 'lifted': self.flags.lifted_once,
                                       'terms': dict(self.ep_terms)}
        return self._obs(self._state), reward, terminated, truncated, info

    def close(self):
        self.backend.close()
