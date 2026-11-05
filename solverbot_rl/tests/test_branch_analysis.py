"""Lógica de clasificación del diagnóstico de ramas de muñeca (imitation/branch_analysis.py), con datos sintéticos.

Fija las distinciones exigidas: un cambio de signo de la acción de joint4 NO es un cambio de rama; un cambio real de
rama exige que la configuración medida pase de estar comprometida con una familia a estarlo con la otra.
"""
import numpy as np

from solverbot_rl.imitation import branch_analysis as B

REP_A = np.array([0.2, 1.0, -0.8, 1.6, 0.9, 0.5])
REP_B = np.array([0.2, 1.0, -0.8, -1.5, -0.9, 0.5])


def _q(j4, j5):
    return np.array([0.2, 1.0, -0.8, j4, j5, 0.5])


def test_family_label_is_sign_of_joint4():
    assert B.family_of(REP_A) == 'A' and B.family_of(REP_B) == 'B'


def test_affinity_and_executed_branch_hysteresis():
    traj = [_q(-0.6, 0.9)] + [_q(-0.6 - 0.1 * k, 0.9 - 0.18 * k) for k in range(1, 10)]
    aff = np.array([B.affinity(q, REP_A, REP_B) for q in traj])
    lab, changes = B.executed_branch(aff)
    assert lab[-1] == 'B' and changes == []           # se compromete con B una vez y no cambia


def test_sign_change_without_real_branch_change():
    """La orden de joint4 cambia de signo dos veces pero la muñeca nunca deja de estar comprometida con B."""
    a4 = np.array([-1, -1, -0.9, 0.8, 0.9, -0.7, -1.0])
    assert B.sign_changes(a4) == [3, 5]
    traj = [_q(-1.2, -0.6), _q(-1.25, -0.65), _q(-1.3, -0.7), _q(-1.25, -0.72), _q(-1.2, -0.75), _q(-1.25, -0.8),
            _q(-1.3, -0.85)]
    aff = np.array([B.affinity(q, REP_A, REP_B) for q in traj])
    _, changes = B.executed_branch(aff)
    assert changes == []


def test_real_branch_change_detected():
    """La muñeca pasa de estar sobre B a estar sobre A: cambio real (uno solo, pese a la banda intermedia)."""
    js = np.linspace(-1.4, 1.5, 15)
    traj = [_q(j, 0.62 * j) for j in js]
    aff = np.array([B.affinity(q, REP_A, REP_B) for q in traj])
    lab, changes = B.executed_branch(aff)
    assert lab[0] == 'B' and lab[-1] == 'A' and len(changes) == 1


def test_intention_debounce_and_active_mask():
    intent = ['B', 'B', 'B', 'A', 'B', 'B', None, 'A', 'A', 'A']
    assert B.intention_switches(intent) == [7]        # el 'A' aislado del paso 3 no cuenta
    active = np.array([True] * 6 + [False] * 4)
    assert B.intention_switches(intent, active=active) == []


def test_intended_branch_from_wrist_command():
    q = _q(-0.6, 0.9)
    assert B.intended_branch(q, np.array([0, 0, 0, -1.0, -1.0, 0, 0]), REP_A, REP_B) == 'B'
    assert B.intended_branch(q, np.array([0, 0, 0, 1.0, 0.0, 0, 0]), REP_A, REP_B) == 'A'
    assert B.intended_branch(q, np.zeros(7), REP_A, REP_B) is None


def test_detour_and_wrist_limit():
    assert abs(B.detour_mm([0.2, 0.15, 0.1, 0.14, 0.05]) - 40.0) < 1e-9
    lo, hi = np.full(6, -1.22), np.full(6, 1.22)
    q = np.array([_q(-1.0, -1.21), _q(-1.0, -0.5)])
    assert B.wrist_limit_frac(q, lo, hi) == 0.5


def test_no_false_commitment_at_start():
    """Arranque cerca de la solución B (sin haber elegido nada) y movimiento monótono hacia A: un solo
    compromiso (A), ningún cambio de rama."""
    rep_b = np.array([0.2, 1.0, -0.8, -1.16, 0.57, 0.5])
    traj = np.array([_q(j, 0.84) for j in np.linspace(-0.5, 1.5, 15)])
    aff = np.array([B.affinity(q, REP_A, rep_b) for q in traj])
    lab_naive, ch_naive = B.executed_branch(aff)
    assert lab_naive[0] == 'B' and len(ch_naive) == 1            # la versión sin acercamiento lo contaba
    lab, changes = B.executed_branch(aff, dists=B.wrist_distances(traj, REP_A, rep_b))
    assert lab[-1] == 'A' and changes == []


def test_label_conflicts_counts_only_cross_episode_opposite_pairs():
    X = np.array([[0.0], [0.01], [0.02], [5.0]])
    fam = np.array(['A', 'B', 'A', 'B'])
    ep = np.array(['e1', 'e2', 'e1', 'e3'])
    a4 = np.array([1.0, -1.0, 1.0, -1.0])
    r = B.label_conflicts(X, fam, ep, a4, radius=0.05)
    assert r['pairs_conflict'] == 2 and r['pairs_same'] == 0 and r['da4_conflict_mean'] == 2.0
