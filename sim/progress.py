"""Progression le long de la ligne centrale (projection, record, complétion)."""

from __future__ import annotations

import numpy as np

from sim.track import Track


def project(track: Track, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Projette K points (K, 2) sur la ligne centrale.

    Retourne (s, lateral) : abscisse curviligne et distance à la ligne centrale (m), formes (K,).
    """
    a = track.center
    ab = np.roll(a, -1, axis=0) - a  # (N, 2)
    seg_len2 = (ab**2).sum(axis=1)  # (N,)
    ap = points[:, None, :] - a[None, :, :]  # (K, N, 2)
    t = np.clip((ap * ab[None]).sum(axis=2) / seg_len2, 0.0, 1.0)  # (K, N)
    diff = ap - t[:, :, None] * ab[None]
    dist = np.linalg.norm(diff, axis=2)  # (K, N)
    i = dist.argmin(axis=1)
    k = np.arange(len(points))
    s = track.arclength[i] + t[k, i] * np.sqrt(seg_len2[i])
    return s, dist[k, i]


def wrap_delta(delta: float, length: float) -> float:
    """Ramène une différence d'abscisse dans [-L/2, L/2) (passage de la ligne de départ)."""
    return (delta + length / 2) % length - length / 2


class ProgressTracker:
    """Suit le progrès « déroulé » et son record : seul un nouveau record rapporte."""

    def __init__(self, length: float, s0: float) -> None:
        self.length = length
        self._s_prev = s0
        self._cum = 0.0
        self.best = 0.0

    def update(self, s: float, count: bool = True) -> float:
        """Intègre la nouvelle abscisse. Retourne le gain de record (m) si `count`."""
        self._cum += wrap_delta(s - self._s_prev, self.length)
        self._s_prev = s
        target = min(self._cum, self.length)  # le record est plafonné à un tour
        if not count or target <= self.best:
            return 0.0
        gain = target - self.best
        self.best = target
        return gain

    @property
    def completion(self) -> float:
        return min(max(self.best / self.length, 0.0), 1.0)

    @property
    def lap_done(self) -> bool:
        return self.best >= self.length
