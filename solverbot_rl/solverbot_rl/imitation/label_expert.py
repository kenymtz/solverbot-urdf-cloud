"""Experto de ETIQUETADO para `reach` (solo imitación; no reemplaza a solverbot_rl.expert.ScriptedExpert).

Problema medido (Gazebo, 100 demostraciones de ScriptedExpert en reach_var_m): la IK global con eje de aproximación
(yaw libre) y reinicios aleatorios elige ramas de muñeca equivalentes de forma arbitraria: joint4 arranca positivo en
74 episodios y negativo en 26, y la configuración final de joint4/joint5 abarca casi todo su rango. Para un estado
dado hay dos "respuestas correctas" opuestas; la clonación con MSE las promedia y la política termina contra el
límite de joint5.

Este experto conserva todo el comportamiento de ScriptedExpert salvo la elección del objetivo articular lejano en
`reach`: en cada paso usa la solución de IK LOCAL (sin reinicios aleatorios, partiendo de la orden articular actual)
-> la más cercana en espacio articular. La etiqueta es así una función continua y unimodal del estado, también en
estados que el experto no habría visitado (requisito de DAgger). Si la IK local no converge se usa la solución
global original.
"""
from __future__ import annotations

import numpy as np

from ..expert import ScriptedExpert
from ..kinematics import solve_ik

TILTS = (20, 30, 40)


class ReachLabelExpert(ScriptedExpert):
    def _ik(self, p):
        e = self.env
        if e.params.goal != 'reach' or self.R is not None:
            return super()._ik(p)
        for tilt in TILTS:
            ax = np.array([np.sin(np.radians(tilt)), 0.0, -np.cos(np.radians(tilt))])
            q, err = solve_ik(e.chain, 'tcp_link', e.arm_joints, e._q_cmd, p, None, self._fixed(), approach_axis=ax,
                              rot_weight=1.0, restarts=0, reg=0.002)
            if err < 0.003:
                self.p_goal, self.q_goal = p.copy(), q
                return q
        self.p_goal = None                      # sin caché: la global se recalcula desde el estado actual
        return super()._ik(p)
