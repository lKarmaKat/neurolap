"""Pilote réflexe de référence : n'utilise que l'observation (comme le futur réseau de neurones)."""

from __future__ import annotations

import numpy as np


class RayPilot:
    """Observation (9 valeurs) -> action [steering, throttle].

    Braque vers le côté le plus dégagé et règle la vitesse cible sur la distance devant.
    """

    def __init__(self, gain: float = 3.0, base: float = 0.2, slope: float = 0.8, kv: float = 4.0):
        self.gain, self.base, self.slope, self.kv = gain, base, slope, kv

    def __call__(self, obs: np.ndarray) -> list[float]:
        rays, speed = obs[:7], obs[7]
        steer = np.clip(self.gain * (rays[4:7].sum() - rays[0:3].sum()), -1.0, 1.0)
        target_speed = self.base + self.slope * rays[3]  # fraction de la vitesse max
        throttle = np.clip(self.kv * (target_speed - speed), -1.0, 1.0)
        return [float(steer), float(throttle)]
