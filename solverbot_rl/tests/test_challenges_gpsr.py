"""GPSR: gramática cerrada (generador <-> intérprete) y entorno Gymnasium (backend cinemático)."""
import numpy as np
import pytest

from solverbot_rl.challenges.gpsr_env import LEVELS, SKILLS, GPSREnv, GPSRPlanner
from solverbot_rl.challenges.gpsr_grammar import (FURNITURE, NEEDS_GRASP, ROOMS, VERBS, CommandGenerator,
                                                  CommandParser)


def test_parser_roundtrip_closed_grammar():
    gen = CommandGenerator(VERBS, np.random.default_rng(7), list(ROOMS) + list(FURNITURE))
    parser = CommandParser()
    texts = set()
    for _ in range(3000):
        c = gen.sample()
        texts.add(c.text)
        r = parser.parse(c.text)
        assert r is not None and r.key() == c.key(), c.text
    assert len(texts) > 1000                     # variedad de formulaciones sin cambiar código


@pytest.mark.parametrize('text', ['hola robot', 'prepará un café', 'andá', 'buscá a la persona de camiseta roja'])
def test_parser_rejects_out_of_grammar(text):
    assert CommandParser().parse(text) is None


def test_paraphrases_map_to_same_command():
    p = CommandParser()
    a = p.parse('andá a la cocina y contá cuántas personas hay')
    b = p.parse('decime cuántas personas hay en la cocina')
    assert a.key() == b.key() == ('count_people', None, 'kitchen', None, None, None)
    # "la cocina eléctrica" es un mueble, no la habitación; "la mesa del lavadero" tampoco es la habitación
    assert p.parse('ve a la cocina eléctrica').location == 'cooking_table'
    assert p.parse('andá a la mesa del lavadero').location == 'laundry_table'


def test_manipulation_commands_flagged():
    c = CommandParser().parse('traeme la esponja de la mesada')
    assert c.verb == 'bring' and c.requires_grasp and 'bring' in NEEDS_GRASP


@pytest.fixture(scope='module')
def env():
    e = GPSREnv(level=2)
    yield e
    e.close()


def test_spaces_and_reset(env):
    obs, info = env.reset(seed=3)
    assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
    assert len(info['commands_text']) == LEVELS[2]['n']
    assert tuple(env.action_space.nvec) == (len(SKILLS), 4)
    assert not env.door_open                    # el robot empieza afuera con la puerta cerrada


def test_reset_reproducible(env):
    o1, i1 = env.reset(seed=21)
    o2, i2 = env.reset(seed=21)
    np.testing.assert_array_equal(o1, o2)
    assert i1['commands_text'] == i2['commands_text']


def test_commands_hidden_until_received(env):
    env.reset(seed=4)
    assert env.received == [] and env._cmd() is None
    _, r, _, _, info = env.step(np.array([SKILLS.index('receive_all'), 0]))
    assert not info['skill_ok']                 # lejos del operador: no recibe nada
    assert env.received == []


def test_moving_before_door_opens_is_safe(env):
    env.reset(seed=5)
    _, _, te, _, info = env.step(np.array([SKILLS.index('wait_door'), 0]))
    assert info['skill_ok'] and env.door_open and env.mem['door_open'] and not info['contact']


@pytest.mark.parametrize('level', [0, 1])
def test_planner_solves(level):
    e = GPSREnv(level=level)
    obs, _ = e.reset(seed=1000)
    ex = GPSRPlanner(e)
    done = False
    while not done:
        obs, r, te, tr, info = e.step(ex.act(obs))
        done = te or tr
    assert info['episode_summary']['success'], info['episode_summary']
    assert all(ev['parsed_ok'] for ev in e.events if ev['ev'] == 'command')
    e.close()
