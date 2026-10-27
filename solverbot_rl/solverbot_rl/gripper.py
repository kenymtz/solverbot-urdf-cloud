"""Emulación del acoplamiento rígido de los dedos del Piper en simulación (sin ROS, testeable).

En el Piper real ambos dedos los mueve un solo motor con acoplamiento mecánico: no pueden desplazarse "en bloque".
En Gazebo Harmonic + DART no hay restricción mimic física y, con un servo de posición por dedo, al apretar un objeto
ambos servos quedan saturados a su esfuerzo máximo: el equilibrio lateral es neutro y el par de dedos deriva en bloque
hasta un tope (medido el 2026-10-27: j7 20 -> 5 mm, j8 -20 -> -35 mm con la apertura constante en 40 mm).

Ley de centrado (aplicada a 50 Hz por el backend Gazebo mientras se aprieta):
    w = (q7 - q8) / 2   semiapertura medida        m = (q7 + q8) / 2   descentrado (0 = simétrico)
    objetivo7 = (w - squeeze) - m        objetivo8 = -(w - squeeze) - m
Si m = 0 ambos dedos aprietan `squeeze` hacia adentro; si el par se descentra, el dedo adelantado cede y el otro empuja
hasta recentrar (zona muerta |m| < squeeze). La pinza sigue teniendo UNA orden lógica (g, objetivo de joint7).
"""
from __future__ import annotations

import numpy as np

Q7_RANGE = (0.0, 0.035)
Q8_RANGE = (-0.035, 0.0)


def is_squeezing(g_cmd: float, q7: float, q8: float, squeeze: float) -> bool:
    """La orden pide cerrar más allá de la apertura actual: los dedos están (o estarán) apretando algo."""
    return g_cmd < (q7 - q8) / 2.0 - squeeze


def coupled_finger_targets(g_cmd: float, q7: float, q8: float, squeeze: float = 0.002) -> tuple[float, float]:
    """Objetivos de posición (joint7, joint8) para una orden lógica `g_cmd` de joint7."""
    if not is_squeezing(g_cmd, q7, q8, squeeze):
        return float(np.clip(g_cmd, *Q7_RANGE)), float(np.clip(-g_cmd, *Q8_RANGE))
    w = (q7 - q8) / 2.0
    m = (q7 + q8) / 2.0
    hold = max(w - squeeze, g_cmd)
    return float(np.clip(hold - m, *Q7_RANGE)), float(np.clip(-hold - m, *Q8_RANGE))
