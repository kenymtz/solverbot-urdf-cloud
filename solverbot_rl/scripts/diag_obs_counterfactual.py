#!/usr/bin/env python3
"""Fase F: separa visión y control con contrafácticos de la observación del objeto (DIAGNÓSTICO, no despliegue).

La MISMA política congelada se ejecuta en Gazebo con las mismas semillas; solo cambia, del lado de la política, la parte
"objeto" de la observación de 31 valores (obj_rel = estimación − TCP y la bandera obj_valid):

  perception      sin cambios (referencia; la observación tal como la entrega el entorno)
  hold_valid      misma estimación que perception (el entorno ya conserva la última válida), bandera = 1 si alguna vez
                  hubo detección: ¿la caída de la bandera, por sí sola, desvía a la política?
  first_estimate  la primera detección válida del episodio (desde home el cubo está a la vista) congelada, bandera = 1:
                  estimación continua SIN verdad del simulador (el cubo no se mueve en reach)
  oracle          posición REAL del cubo (Gazebo), bandera = 1. PRIVILEGIADO: solo responde "¿falla la política aun con
                  una estimación continua y exacta?". Nunca es un modo de percepción.

El entorno, la recompensa y el criterio de éxito no cambian; los .npz guardan la observación ORIGINAL del entorno y
cams_<seed>.json la observación modificada que vio la política.

    python3 solverbot_rl/scripts/diag_obs_counterfactual.py --mode oracle \
        --policy solverbot_rl/docs/resultados/reach_bc_v1/modelos/bc_v2w.zip --level reach_var_m --seeds 902003 \
        --out runs/dual_camera/cf/bc_v2w_var_m_oracle
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import OBS_SLICES, run_episode  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_reach import aggregate  # noqa: E402

MODES = ('perception', 'hold_valid', 'first_estimate', 'oracle')


class ObjectObsOverride:
    """Reescribe obj_rel / obj_valid de la observación ANTES de pasarla a la política."""

    def __init__(self, mode, env):
        assert mode in MODES
        self.mode, self.env = mode, env
        self.reset()

    def reset(self):
        self.seen = False
        self.first = None
        self.log = []

    def __call__(self, obs):
        o = np.array(obs, np.float32, copy=True)
        tcp = o[OBS_SLICES['tcp']]
        valid = bool(o[16] > 0.5)
        est = o[OBS_SLICES['obj_rel']] + tcp
        if valid and self.first is None:
            self.first = est.copy()
        self.seen |= valid
        if self.mode == 'hold_valid' and self.seen:
            o[16] = 1.0
        elif self.mode == 'first_estimate' and self.first is not None:
            o[OBS_SLICES['obj_rel']] = self.first - tcp
            o[16] = 1.0
        elif self.mode == 'oracle':
            o[OBS_SLICES['obj_rel']] = self.env.unwrapped.backend.object_pose_truth() - tcp
            o[16] = 1.0
        be = self.env.unwrapped.backend
        truth = be.object_pose_truth()
        # antigüedad de la imagen de muñeca respecto del reloj (fase cámara/paso; constante dentro de un lanzamiento)
        img = getattr(be, '_img', {}).get('wrist')
        self.log.append({'env_valid': valid, 'policy_valid': bool(o[16] > 0.5),
                         'policy_est': (o[OBS_SLICES['obj_rel']] + tcp).tolist(),
                         'policy_est_err_mm': float(1000 * np.linalg.norm(o[OBS_SLICES['obj_rel']] + tcp - truth)),
                         'wrist_image_age_s': None if img is None else float(be._clock - img[0])})
        return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--policy', required=True)
    ap.add_argument('--experiment', default='reach_bc_v1')
    ap.add_argument('--level', default='reach_var_m')
    ap.add_argument('--seeds', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    from solverbot_rl.imitation.bc import load_policy

    exp = load_experiment(args.experiment)
    env = SolverBotPickPlaceEnv(backend='gazebo', **env_kwargs(exp))
    lvl = level_index(env, args.level)
    pol = load_policy(args.policy, deterministic=True)
    ov = ObjectObsOverride(args.mode, env)
    os.makedirs(args.out, exist_ok=True)
    done = {r['seed'] for r in D.read_index(args.out)}
    seeds = [s for s in parse_seeds(args.seeds) if s not in done]
    print(f'[cf] modo={args.mode} política={args.policy} nivel={args.level} {len(seeds)} episodios', flush=True)
    for seed in seeds:
        ov.reset()
        arrays, s = run_episode(env, seed, lvl, act=lambda o: pol(ov(o)), rng=np.random.default_rng(seed))
        pe = [r['policy_est_err_mm'] for r in ov.log]
        ages = [r['wrist_image_age_s'] for r in ov.log if r['wrist_image_age_s'] is not None]
        s.update({'policy': args.policy, 'expert_in_loop': False, 'obs_mode': args.mode,
                  'privileged_obs': args.mode == 'oracle',
                  'policy_valid_rate': float(np.mean([r['policy_valid'] for r in ov.log])),
                  'policy_est_err_mean_mm': float(np.mean(pe)), 'policy_est_err_max_mm': float(np.max(pe)),
                  'wrist_image_age_median_s': float(np.median(ages)) if ages else None})
        D.save_episode(args.out, arrays, s)
        with open(os.path.join(args.out, f'cf_{seed}.json'), 'w') as fh:
            json.dump(ov.log, fh)
        print(f'[cf] seed={seed} éxito={s["success"]} pasos={s["length"]} d0={s["d0_mm"]:.1f} dmin={s["dmin_mm"]:.1f} '
              f'dfin={s["dfinal_mm"]:.1f} zf={s["tcp_z2_final"]:.3f} det_muñeca={s["valid_rate"]:.2f} '
              f'err_obs_política={s["policy_est_err_mean_mm"]:.1f} mm', flush=True)
    rows = sorted(D.read_index(args.out), key=lambda r: r['seed'])
    summ = {'policy': args.policy, 'obs_mode': args.mode, 'privileged_obs': args.mode == 'oracle',
            'level': args.level, 'seeds': args.seeds, **aggregate(rows),
            'policy_est_err_mean_mm': float(np.mean([r['policy_est_err_mean_mm'] for r in rows]))}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1)
    print(json.dumps({k: summ[k] for k in ('episodes', 'successes', 'success_wilson95', 'failed_seeds')}), flush=True)
    env.close()


if __name__ == '__main__':
    main()
