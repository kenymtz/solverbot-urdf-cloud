"""Compuertas de percepción de personas (hallazgo GPSR: camiseta naranja confundida con la madera de los muebles)."""
import numpy as np

from solverbot_rl.challenges.sensing import ColorHeadDetector, locate_gated, saturation
from solverbot_rl.challenges.world import challenge_arena
from solverbot_rl.perception import color_mask

ORANGE = (0.95, 0.50, 0.05)
WOOD = (0.55, 0.36, 0.20)


def _img(rgb, shade=1.0, n=40):
    return np.tile((np.asarray(rgb) * shade * 255).astype(np.uint8), (n, n, 1))


def test_chromaticity_alone_confuses_wood_with_orange():
    assert color_mask(_img(WOOD), ORANGE).all()           # el problema original


def test_saturation_gate_separates_orange_from_wood_under_shading():
    det = ColorHeadDetector({'person_c': ORANGE}, persons=['person_c'])
    t = np.asarray(ORANGE)
    thr = det.PERSON_SAT_FRAC * (t.max() - t.min()) / t.max()
    for shade in (0.4, 0.7, 1.0):                         # la iluminación difusa escala el color: no cambia la saturación
        assert (saturation(_img(ORANGE, shade)) >= thr).all()
        assert not (saturation(_img(WOOD, shade)) >= thr).any()


def test_static_surface_gate():
    a = challenge_arena()
    p = next(p for p in a.prims if p.shape == 'box' and p.z1 > 0.7 and '/' in p.name and 'wall' not in p.name)
    on = np.array([[p.cx, p.cy, (p.z0 + p.z1) / 2]])
    assert a.on_static_surface(on).all()
    # punto en espacio libre (lejos de toda primitiva de navegación) a altura de torso
    i, j = np.unravel_index(np.argmax(a.nav_dist), a.nav_dist.shape)
    free = np.array([[a.x0 + (j + 0.5) * a.res, a.y0 + (i + 0.5) * a.res, 1.0]])
    assert not a.on_static_surface(free).any()


def test_locate_gated_rejects_points_on_furniture():
    rgb = _img(ORANGE)
    depth = np.full(rgb.shape[:2], 2.0, np.float32)
    k = np.array([[100.0, 0, 20], [0, 100.0, 20], [0, 0, 1]])
    cam = np.eye(4)
    assert locate_gated(rgb, depth, k, cam, ORANGE) is not None
    assert locate_gated(rgb, depth, k, cam, ORANGE, reject=lambda P: np.ones(len(P), bool)) is None


def test_size_gate_rejects_small_object_of_shirt_color():
    """La fruta naranja (objeto de GPSR) sobre la mesada: mismo color que la camiseta, pero ~8 cm de alto."""
    rgb = _img(ORANGE)
    k = np.array([[100.0, 0, 20], [0, 100.0, 20], [0, 0, 1]])
    cam = np.eye(4)                                        # óptico -> mapa: x = z_opt, y = -x_opt, z = 1.0 - y_opt
    cam[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    cam[2, 3] = 1.0
    depth = np.full(rgb.shape[:2], 2.0, np.float32)        # 40 px a 2 m con f = 100 -> 0.8 m de alto
    assert locate_gated(rgb, depth, k, cam, ORANGE, min_height=0.2) is not None
    depth_far = np.full(rgb.shape[:2], 0.3, np.float32)    # 40 px a 0.3 m -> 0.12 m de alto
    assert locate_gated(rgb, depth_far, k, cam, ORANGE, min_height=0.2) is None
