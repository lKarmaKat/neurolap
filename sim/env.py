"""Environnement de simulation, API de type Gymnasium (sans dépendance)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.car import CarParams, CarState, step_car
from sim.sensors import MAX_RANGE, cast_rays, wall_segments
from sim.track import generate_track


class NeurolapEnv:
    def __init__(
        self, params: CarParams | None = None, dt: float = 1 / 30, max_steps: int = 1000
    ) -> None:
        self.params = params or CarParams()
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self._new_episode()

    def _new_episode(self) -> None:
        """Génère un circuit depuis le rng et place la voiture sur la ligne de départ."""
        self.track = generate_track(self.rng)
        x, y = self.track.start_pos
        self.car = CarState(x=x, y=y, heading=self.track.start_heading)
        self._segments = wall_segments(self.track)
        self.steps = 0

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self._new_episode()
        return self._observe(), self._info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (2,):
            raise ValueError(f"action attendue de forme (2,), reçu {a.shape}")
        steering, throttle = np.clip(a, -1.0, 1.0)
        self.car = step_car(self.car, float(steering), float(throttle), self.params, self.dt)
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return self._observe(), 0.0, False, truncated, self._info()

    def _rays(self) -> np.ndarray:
        """Distances (m) des rayons aux murs."""
        c = self.car
        return cast_rays(c.x, c.y, c.heading, *self._segments)

    def _info(self) -> dict[str, Any]:
        return {"ray_distances": self._rays()}

    def _observe(self) -> np.ndarray:
        """9 valeurs : 7 rayons dans [0, 1], puis vitesse et vitesse angulaire dans [-1, 1]."""
        p = self.params
        speed = self.car.speed / p.max_speed
        yaw = np.clip(self.car.yaw_rate / p.max_yaw_rate, -1.0, 1.0)
        rays = self._rays() / MAX_RANGE
        return np.concatenate([rays, [speed, yaw]]).astype(np.float32)
