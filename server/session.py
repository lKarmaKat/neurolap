"""Session de jeu : une simulation, l'entrée courante, fin d'épisode (sans réseau)."""

from __future__ import annotations

from typing import Any

import numpy as np

from server.protocol import encode_state, encode_track
from sim.env import NeurolapEnv

# Rampe de braquage (unités de steering par seconde) : entrée clavier uniquement.
STEER_RISE_RATE = 4.0  # 0 -> 1 en 0,25 s
STEER_RETURN_RATE = 8.0  # recentrage plus rapide


class Session:
    def __init__(
        self, env: NeurolapEnv | None = None, end_pause_ticks: int = 30, seed: int | None = None
    ) -> None:
        self.env = env or NeurolapEnv()
        self.end_pause_ticks = end_pause_ticks
        self._seed_rng = np.random.default_rng(seed)
        self.steering = 0.0  # braquage réellement appliqué (rampe vers la cible)
        self.target_steering = 0.0  # braquage demandé par le clavier
        self.throttle = 0.0
        self._pause = 0
        self._track_msg: dict[str, Any] = {}
        self._state_msg: dict[str, Any] = {}
        self.reset()

    def set_input(self, steering: float, throttle: float) -> None:
        self.target_steering, self.throttle = steering, throttle

    def _ramp_steering(self) -> None:
        """Simule un volant analogique : le braquage rejoint la cible avec une vitesse limitée."""
        cur, tgt = self.steering, self.target_steering
        rate = STEER_RETURN_RATE if (abs(tgt) < abs(cur) or tgt * cur < 0) else STEER_RISE_RATE
        step = rate * self.env.dt
        self.steering = cur + max(-step, min(step, tgt - cur))

    def reset(self, seed: int | None = None) -> list[dict[str, Any]]:
        """Nouvel épisode (seed tirée au hasard si absente). Retourne [track, state]."""
        if seed is None:
            seed = int(self._seed_rng.integers(2**31))
        _, info = self.env.reset(seed=seed)
        self._pause = 0
        self.steering = 0.0
        self._track_msg = encode_track(self.env, seed)
        self._state_msg = encode_state(self.env, 0.0, False, False, info)
        return self.snapshot()

    def snapshot(self) -> list[dict[str, Any]]:
        """Messages à envoyer à un client qui vient de se connecter."""
        return [self._track_msg, self._state_msg]

    def tick(self) -> list[dict[str, Any]]:
        """Avance d'un pas de simulation. Retourne les messages à diffuser."""
        if self.env.done:
            self._pause += 1
            if self._pause >= self.end_pause_ticks:
                return self.reset()
            return [self._state_msg]
        self._ramp_steering()
        _, reward, terminated, truncated, info = self.env.step([self.steering, self.throttle])
        self._state_msg = encode_state(self.env, reward, terminated, truncated, info)
        return [self._state_msg]
