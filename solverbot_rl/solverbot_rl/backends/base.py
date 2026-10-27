"""Interfaz común de los backends de simulación del entorno de Pick and Place."""
from __future__ import annotations

import abc

import numpy as np


class Backend(abc.ABC):
    """Un backend avanza el mundo y expone mediciones. Separa explícitamente:
      * mediciones disponibles en el robot real (articulaciones, imágenes, estimación por percepción);
      * ground truth del simulador (`object_pose_truth`), SOLO para recompensas y métricas.
    """

    physical = False          # True solo si los contactos/agarres los resuelve un motor de física

    @abc.abstractmethod
    def reset(self, rng: np.random.Generator, obj_xy: np.ndarray, obj_yaw: float, arm_q: np.ndarray,
              gripper: float, lift: float, head: dict[str, float]) -> None: ...

    @abc.abstractmethod
    def command(self, arm_q: np.ndarray, gripper: float) -> None:
        """Objetivos de posición del brazo (6) y de la pinza (joint7)."""

    @abc.abstractmethod
    def advance(self, dt: float) -> None:
        """Avanza `dt` segundos de tiempo SIMULADO y vuelve cuando las mediciones corresponden a ese instante."""

    @abc.abstractmethod
    def joint_positions(self) -> dict[str, float]: ...

    @abc.abstractmethod
    def object_pose_truth(self) -> np.ndarray:
        """Posición real del objeto (solo recompensa/métricas)."""

    @abc.abstractmethod
    def object_estimate(self) -> tuple[np.ndarray, bool]:
        """Posición estimada por percepción y si la detección es válida."""

    def object_yaw_truth(self) -> float:
        """Yaw real del objeto en el frame de la base (solo experto guionado / métricas)."""
        return 0.0

    def world_xy_to_base(self, xy) -> np.ndarray:
        """Punto fijo del mundo (p. ej. la zona objetivo) en el frame de la base. Identidad si la base no se mueve."""
        return np.asarray(xy, float)

    def images(self) -> dict[str, np.ndarray]:
        return {}

    def close(self) -> None:
        pass
