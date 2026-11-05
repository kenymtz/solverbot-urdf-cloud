"""Pruebas del pipeline de imitación (backend cinemático: SIN física; validan estructura y software, no el robot)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from solverbot_rl.config import load_config  # noqa: E402
from solverbot_rl.envs.pick_place import SolverBotPickPlaceEnv  # noqa: E402
from solverbot_rl.expert import ScriptedExpert  # noqa: E402
from solverbot_rl.imitation import dataset as D  # noqa: E402
from solverbot_rl.imitation.experiment import env_kwargs, level_index, load_experiment, parse_seeds  # noqa: E402
from solverbot_rl.imitation.rollout import OBS_DIM, OBS_SLICES, run_episode  # noqa: E402

torch = pytest.importorskip('torch')
from solverbot_rl.imitation import bc  # noqa: E402
from solverbot_rl.imitation import finetune as FT  # noqa: E402


@pytest.fixture(scope='module')
def exp_env():
    env = SolverBotPickPlaceEnv(backend='kinematic', **env_kwargs(load_experiment('reach_bc_v1')))
    yield env
    env.close()


@pytest.fixture(scope='module')
def demo_dir(tmp_path_factory, exp_env):
    root = str(tmp_path_factory.mktemp('demos'))
    ex = ScriptedExpert(exp_env)
    lvl = level_index(exp_env, 'reach_var_m')
    for s in range(100, 112):
        arrays, summ = run_episode(exp_env, s, lvl, expert=ex)
        D.save_episode(root, arrays, summ)
    return root


# ------------------------------------------------------------------------------------------- experimento / entorno
def test_experiment_keeps_original_levels(exp_env):
    base = load_config()['curriculum']['levels']
    assert exp_env.levels[:len(base)] == base
    assert level_index(exp_env, 'reach') == 0
    assert exp_env.levels[level_index(exp_env, 'reach_var_m')]['start_q_noise'] > 0
    assert parse_seeds('5-7,9') == [5, 6, 7, 9]


def test_original_level_reset_unchanged_by_experiment(exp_env):
    """El nivel 0 con la configuración experimental produce exactamente el mismo reset que la configuración original
    (start_q_noise ausente no consume la RNG)."""
    ref = SolverBotPickPlaceEnv(backend='kinematic')
    for seed in (0, 7, 123):
        o1, _ = ref.reset(seed=seed, options={'level': 0})
        o2, _ = exp_env.reset(seed=seed, options={'level': 0})
        np.testing.assert_array_equal(o1, o2)
        np.testing.assert_array_equal(ref.start_q, np.array(ref.cfg['robot']['home']))


def test_start_noise_within_bounds(exp_env):
    lvl = level_index(exp_env, 'reach_var_m')
    home = np.array(exp_env.cfg['robot']['home'])
    starts = []
    for seed in range(20):
        exp_env.reset(seed=seed, options={'level': lvl})
        starts.append(exp_env.start_q)
        assert np.all(np.abs(exp_env.start_q - home) <= 0.10 + 1e-9)
    assert np.std(np.array(starts), axis=0).min() > 0.01      # varía de verdad


# ------------------------------------------------------------------------------------------- dataset
def test_demo_structure(demo_dir):
    rows = D.read_index(demo_dir)
    assert len(rows) == 12 and all(r['success'] for r in rows)
    for r in rows:
        with np.load(os.path.join(demo_dir, r['file'])) as z:
            arrays = {k: z[k] for k in z.files}
        assert D.validate_episode(arrays) == []
        assert arrays['obs'].shape[1] == OBS_DIM and arrays['obs'].dtype == np.float32
        np.testing.assert_array_equal(arrays['actions'], arrays['expert_actions'])
        assert r['dims_ok'] and not r['has_nan'] and r['object_source'] == 'perception'


def test_validate_detects_corruption(demo_dir):
    r = D.read_index(demo_dir)[0]
    with np.load(os.path.join(demo_dir, r['file'])) as z:
        good = {k: z[k] for k in z.files}
    bad = dict(good, obs=good['obs'].copy())
    bad['obs'][3, 5] = np.nan
    assert any('NaN' in e for e in D.validate_episode(bad))
    bad = dict(good, actions=good['actions'][:, :6])
    assert D.validate_episode(bad)
    bad = dict(good, next_obs=good['next_obs'] + 1.0)
    assert any('next_obs' in e for e in D.validate_episode(bad))


def test_no_privileged_information_in_policy_inputs(demo_dir, tmp_path):
    """La entrada de la política es solo `obs`: alterar la verdad del simulador guardada (priv_*) no cambia nada, y la
    estimación del objeto en obs NO es la pose real (proviene del detector, con ruido)."""
    ds = D.load_datasets([demo_dir])
    root2 = str(tmp_path / 'tampered')
    os.makedirs(os.path.join(root2, 'episodes'))
    for r in D.read_index(demo_dir):
        with np.load(os.path.join(demo_dir, r['file'])) as z:
            arrays = {k: z[k] for k in z.files}
        arrays['priv_obj'] = arrays['priv_obj'] + 5.0
        arrays['priv_tcp'] = arrays['priv_tcp'] * 0
        D.save_episode(root2, arrays, r)
    ds2 = D.load_datasets([root2])
    np.testing.assert_array_equal(ds.obs, ds2.obs)
    np.testing.assert_array_equal(ds.actions, ds2.actions)
    with np.load(os.path.join(demo_dir, D.read_index(demo_dir)[0]['file'])) as z:
        est = z['obs'][1:, OBS_SLICES['obj_rel']] + z['obs'][1:, OBS_SLICES['tcp']]
        truth = z['priv_obj'][:-1]
    assert np.abs(est - truth).max() > 1e-3


def test_split_by_episode_is_disjoint(demo_dir):
    ds = D.load_datasets([demo_dir])
    tr, va = D.split_by_episode(ds, 0.25, seed=0)
    assert set(ds.episode[tr]).isdisjoint(set(ds.episode[va]))
    assert len(tr) + len(va) == len(ds)
    tr2, va2 = D.split_by_episode(ds, 0.25, seed=0)
    np.testing.assert_array_equal(va, va2)


# ------------------------------------------------------------------------------------------- BC
def test_normalization_and_mask():
    rng = np.random.default_rng(0)
    obs = rng.normal(3.0, 0.2, (500, OBS_DIM)).astype(np.float32)
    obs[:, 27:31] = 0.0
    mean, std = bc.compute_normalization(obs)
    for name in bc.UNSCALED:
        s = OBS_SLICES[name]
        assert np.all(mean[s] == 0) and np.all(std[s] == 1)
    assert np.allclose(mean[:6], 3.0, atol=0.05) and np.all(std >= 0.01)
    mask = bc.default_actor_mask()
    model = bc.make_sac(None, mean, std, actor_mask=mask)
    fe_actor = model.actor.features_extractor
    fe_critic = model.critic.features_extractor
    x = torch.as_tensor(obs[:4])
    expected = np.clip((obs[:4] - mean) / std, -10, 10)
    np.testing.assert_allclose(fe_critic(x).numpy(), expected, rtol=1e-5, atol=1e-5)
    out_a = fe_actor(x).numpy()
    assert np.all(out_a[:, OBS_SLICES['prev_action']] == 0)
    np.testing.assert_allclose(out_a[:, :20], expected[:, :20], rtol=1e-5, atol=1e-5)


def test_bc_trains_saves_and_loads(demo_dir, tmp_path):
    ds = D.load_datasets([demo_dir])
    tr, va = D.split_by_episode(ds, 0.25, seed=0)
    mean, std = bc.compute_normalization(ds.obs[tr])
    model = bc.make_sac(None, mean, std, actor_mask=bc.default_actor_mask())
    before = bc.evaluate_actor(model.actor, ds.obs[va], ds.actions[va])['mse']
    res = bc.train_bc(model, ds.obs, ds.actions, tr, va, epochs=60, patience=60)
    assert res['best_val_mse'] < 0.5 * before
    path = str(tmp_path / 'm.zip')
    model.save(path)
    act = bc.load_policy(path)
    a = np.array([act(o) for o in ds.obs[va][:50]])
    assert a.shape == (50, 7) and np.all(np.abs(a) <= 1.0)
    np.testing.assert_allclose(a, model.predict(ds.obs[va][:50], deterministic=True)[0], atol=1e-6)
    loaded = bc.load_model(path)
    assert loaded.observation_space.shape == (OBS_DIM,)
    np.testing.assert_array_equal(loaded.actor.features_extractor.mean.numpy(), np.float32(mean))


# ------------------------------------------------------------------------------------------- fine-tuning
def test_finetune_freeze_update_and_resume(demo_dir, tmp_path, exp_env):
    from stable_baselines3.common.logger import configure

    ds = D.load_datasets([demo_dir])
    mean, std = bc.compute_normalization(ds.obs)
    m0 = bc.make_sac(None, mean, std, actor_mask=bc.default_actor_mask())
    bc.train_bc(m0, ds.obs, ds.actions, np.arange(len(ds)), np.arange(len(ds))[:50], epochs=5, patience=5)
    p_bc = str(tmp_path / 'bc.zip')
    m0.save(p_bc)
    model = FT.SACBC.load(p_bc, env=exp_env, device='cpu', learning_starts=0)
    assert np.array_equal(FT.actor_param_vector(model), FT.actor_param_vector(m0))
    model.set_logger(configure(None, ['csv']))
    model.set_entropy(0.01, learn=False)
    tr = {k: [] for k in ('obs', 'next_obs', 'actions', 'rewards', 'terminated', 'truncated')}
    for r in D.read_index(demo_dir):
        with np.load(os.path.join(demo_dir, r['file'])) as z:
            for k in tr:
                tr[k].append(z[k])
    tr = {k: np.concatenate(v) for k, v in tr.items()}
    n = FT.add_demos_to_buffer(model, tr['obs'], tr['next_obs'], tr['actions'], tr['rewards'], tr['terminated'],
                               tr['truncated'])
    assert model.replay_buffer.size() == n
    model.set_bc_data(ds.obs, ds.actions)
    model.actor_freeze_updates, model.bc_coef, model.q_alpha = 20, 1.0, 2.5
    p0 = FT.actor_param_vector(model)
    c0 = np.concatenate([p.detach().numpy().ravel() for p in model.critic.parameters()])
    model.train(gradient_steps=20, batch_size=64)
    assert np.array_equal(p0, FT.actor_param_vector(model))                 # actor congelado
    c1 = np.concatenate([p.detach().numpy().ravel() for p in model.critic.parameters()])
    assert not np.array_equal(c0, c1)                                         # el crítico sí aprende
    model.train(gradient_steps=5, batch_size=64)
    assert not np.array_equal(p0, FT.actor_param_vector(model))             # el actor recibe gradiente
    # checkpoint + reanudación (modelo, replay buffer, alpha fijo)
    path, rb = str(tmp_path / 'ft.zip'), str(tmp_path / 'ft_rb')
    model.save(path)
    model.save_replay_buffer(rb)
    m2 = FT.SACBC.load(path, env=exp_env, device='cpu')
    m2.load_replay_buffer(rb + '.pkl')
    assert m2.replay_buffer.size() == model.replay_buffer.size()
    assert np.array_equal(FT.actor_param_vector(m2), FT.actor_param_vector(model))
    assert m2.log_ent_coef is None and abs(float(m2.ent_coef_tensor) - 0.01) < 1e-9
    assert m2._n_updates == model._n_updates
