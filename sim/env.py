"""Environnement de simulation, API de type Gymnasium (sans dépendance)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.car import CarParams, CarState, step_car


class NeurolapEnv:
    def __init__(
        self, params: CarParams | None = None, dt: float = 1 / 30, max_steps: int = 1000
    ) -> None:
        self.params = params or CarParams()
        self.dt = dt
        self.max_steps = max_steps
        self.car = CarState()
        self.steps = 0
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self.car = CarState()
        self.steps = 0
        return self._observe(), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (2,):
            raise ValueError(f"action attendue de forme (2,), reçu {a.shape}")
        steering, throttle = np.clip(a, -1.0, 1.0)
        self.car = step_car(self.car, float(steering), float(throttle), self.params, self.dt)
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._observe(), 0.0, False, truncated, {}

    def _observe(self) -> np.ndarray:
        p = self.params
        speed = self.car.speed / p.max_speed
        yaw = np.clip(self.car.yaw_rate / p.max_yaw_rate, -1.0, 1.0)
        return np.array([speed, yaw], dtype=np.float32)
