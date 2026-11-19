"""Environnement de simulation, API de type Gymnasium (sans dépendance)."""

from __future__ import annotations

from typing import Any

import numpy as np

from sim.car import CarParams, CarState, car_corners, step_car
from sim.progress import ProgressTracker, project
from sim.sensors import MAX_RANGE, cast_rays, wall_segments, window_segments
from sim.track import Track, generate_track


class NeurolapEnv:
    def __init__(
        self,
        params: CarParams | None = None,
        dt: float = 1 / 30,
        max_steps: int = 3000,
        track: Track | None = None,
    ) -> None:
        self.params = params or CarParams()
        self.dt = dt
        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self._new_episode(track)

    def _new_episode(self, track: Track | None = None) -> None:
        """Circuit donné (partageable, immuable) ou généré ; voiture sur la ligne de départ."""
        self.track = track if track is not None else generate_track(self.rng)
        x, y = self.track.start_pos
        self.car = CarState(x=x, y=y, heading=self.track.start_heading)
        self._segments = wall_segments(self.track)
        s0, _, i0 = project(self.track, np.array([[x, y]]))  # recherche complète au départ
        self._k = int(i0[0])  # segment de ligne centrale le plus proche (fenêtres de recherche)
        self.progress = ProgressTracker(self.track.length, float(s0[0]))
        self.collided = False
        self.done = False
        self.steps = 0

    def reset(
        self, seed: int | None = None, track: Track | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.rng = np.random.default_rng(seed)
        self._new_episode(track)
        rays = self._rays()
        return self._observe(rays), self._info(rays)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("épisode terminé : appeler reset() avant step()")
        a = np.asarray(action, dtype=np.float64)
        if a.shape != (2,):
            raise ValueError(f"action attendue de forme (2,), reçu {a.shape}")
        steering, throttle = np.clip(a, -1.0, 1.0)
        self.car = step_car(self.car, float(steering), float(throttle), self.params, self.dt)
        self.steps += 1
        pts = np.array(car_corners(self.car, self.params) + [(self.car.x, self.car.y)])
        s, lateral, idx = project(self.track, pts, hint=self._k)
        self._k = int(idx[4])
        self.collided = bool(np.any(lateral[:4] > self.track.width / 2))
        gain = self.progress.update(float(s[4]), count=not self.collided)
        terminated = self.collided or self.progress.lap_done
        truncated = self.steps >= self.max_steps and not terminated
        self.done = terminated or truncated
        reward = gain / self.track.length
        rays = self._rays()  # calculés une seule fois par pas
        return self._observe(rays), reward, terminated, truncated, self._info(rays)

    def _rays(self) -> np.ndarray:
        """Distances (m) des rayons aux murs, en ne testant que les segments proches."""
        c = self.car
        return cast_rays(c.x, c.y, c.heading, *window_segments(*self._segments, self._k))

    def _info(self, rays: np.ndarray) -> dict[str, Any]:
        return {
            "ray_distances": rays,
            "completion": self.progress.completion,
            "collision": self.collided,
        }

    def _observe(self, rays: np.ndarray) -> np.ndarray:
        """9 valeurs : 7 rayons dans [0, 1], puis vitesse et vitesse angulaire dans [-1, 1]."""
        p = self.params
        speed = self.car.speed / p.max_speed
        yaw = np.clip(self.car.yaw_rate / p.max_yaw_rate, -1.0, 1.0)
        return np.concatenate([rays / MAX_RANGE, [speed, yaw]]).astype(np.float32)
