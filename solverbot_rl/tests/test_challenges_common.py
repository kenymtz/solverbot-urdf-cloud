"""Infraestructura común de los desafíos (mapa, navegación, localización, personas simuladas). Backend cinemático."""
import math

import numpy as np
import pytest

from solverbot_rl.challenges.arena import CIRCUMRADIUS, ArenaMap
from solverbot_rl.challenges.backend import KinematicMobileBackend
from solverbot_rl.challenges.humans import ROBOT_KEEP, HumanSim
from solverbot_rl.challenges.nav import NavController, ScanLocalizer, foot_center
from solverbot_rl.challenges.world import ChallengeWorld, challenge_arena


@pytest.fixture(scope='module')
def world():
    return ChallengeWorld()


def _drive(arena, start_ref, goal, final_yaw=None, use_loc=True, extra=None, steps=2500):
    be = KinematicMobileBackend(arena)
    bx, by = ArenaMap.base_from_ref(start_ref[:2], start_ref[2])
    base = np.array([bx, by, start_ref[2]])
    be.entity_radius = {k: 0.22 for k in (extra or {})}
    be.reset(np.random.default_rng(1), base, extra or {})
    loc = ScanLocalizer(arena, be.lidar_angles)
    loc.reset(base, be.odom())
    wps = arena.plan(foot_center(base), goal)
    assert wps is not None
    nav = NavController(wps, final_yaw=final_yaw)
    errs = []
    for _ in range(steps):
        est = loc.update(be.odom(), be.scan()) if use_loc else be.pose_truth()
        errs.append(float(np.linalg.norm(est[:2] - be.pose_truth()[:2])))
        v, w = nav.act(est)
        be.cmd_vel(v, w)
        be.advance(0.1)
        assert not be.collided_truth()
        if nav.done:
            break
    return be, nav, errs


def test_doors_usability(world):
    doors = {d.name: d for d in world.arena.doors}
    # el lavarropas queda a 5 cm del eje de la puerta lavadero-dormitorio (plano 10): no es transitable con la huella
    assert not doors['laundry_bedroom'].usable
    for n in ('entrance_living', 'laundry_living', 'living_kitchen', 'bedroom_kitchen'):
        assert doors[n].usable, n


def test_plan_has_explicit_door_maneuvers(world):
    wps = world.arena.plan((1.8, 2.0), (2.0, 5.5))       # sala -> lavadero
    kinds = [w.kind for w in wps]
    assert 'door_pre' in kinds and 'door_post' in kinds
    assert kinds.index('door_pre') + 1 == kinds.index('door_post')


def test_route_through_door_with_localization(world):
    be, nav, errs = _drive(world.arena, (1.8, 2.05, -math.pi / 2), (2.0, 5.6), final_yaw=0.0)
    assert nav.done
    assert np.linalg.norm(foot_center(be.pose_truth()) - np.array([2.0, 5.6])) < 0.15
    assert max(errs) < 0.08


def test_localizer_robust_to_people_near_robot(world):
    # personas pegadas al robot no deben arrastrar la estimación (rechazo de puntos ajenos al mapa + prior de odometría)
    people = {'p1': (1.8, 1.35, 0.0), 'p2': (2.4, 1.5, 0.0), 'p3': (1.3, 1.6, 0.0)}
    _, nav, errs = _drive(world.arena, (1.8, 2.3, -math.pi / 2), (2.9, 3.4), extra=people)
    assert nav.done
    assert max(errs) < 0.08


def test_footprint_collision_detects_wall(world):
    a = world.arena
    assert a.footprint_collides(0.10, 1.0, 0.0)            # sobre la pared oeste
    assert not a.footprint_collides(*ArenaMap.base_from_ref((1.8, 2.05), 0.0), 0.0)


def test_people_yield_to_robot(world):
    hs = HumanSim(challenge_arena(), world.person_names, world.parking, world.seats, -0.42)
    n = world.person_names[0]
    robot = np.array([1.8, 2.3])
    hs.place(n, 1.8, 1.0)
    hs.walk_to(n, (1.8, 3.4))                              # ruta que pasa por el robot
    dmin = 1e9
    for _ in range(600):
        hs.update(0.1, robot)
        dmin = min(dmin, float(np.linalg.norm(hs.people[n].xy - robot)))
    assert dmin >= ROBOT_KEEP - 0.05
    assert CIRCUMRADIUS + 0.22 <= ROBOT_KEEP


def test_world_config_consistency(world):
    assert len(world.person_names) == 4 and len(set(map(tuple, world.person_rgb.values()))) == 4
    for s in world.seats:
        assert world.arena.room_of(*s['xy']) == 'living_room'
    for g in world.ch['laundry']['garments']:
        assert g['model'] == 'rigid_proxy'          # no se presenta un proxy rígido como tela
