"""Progression le long de la ligne centrale (projection, record, complétion)."""

from __future__ import annotations

import numpy as np

from sim.track import Track

PROJECT_WINDOW = 30  # demi-fenêtre de recherche (segments) autour de l'indice précédent


def project(
    track: Track, points: np.ndarray, hint: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projette K points (K, 2) sur la ligne centrale.

    Sans `hint` : recherche sur tous les segments. Avec `hint` (indice de segment connu, par
    exemple celui du pas précédent) : recherche limitée à une fenêtre autour, bien plus rapide.
    Retourne (s, lateral, index) : abscisse curviligne (m), distance à la ligne centrale (m)
    et indice du segment le plus proche, formes (K,).
    """
    n = len(track.center)
    idx = (
        np.arange(n)
        if hint is None
        else (hint + np.arange(-PROJECT_WINDOW, PROJECT_WINDOW + 1)) % n
    )
    a = track.center[idx]  # (W, 2)
    ab = track.center[(idx + 1) % n] - a
    seg_len2 = (ab**2).sum(axis=1)  # (W,)
    ap = points[:, None, :] - a[None, :, :]  # (K, W, 2)
    t = np.clip((ap * ab[None]).sum(axis=2) / seg_len2, 0.0, 1.0)  # (K, W)
    dist = np.linalg.norm(ap - t[:, :, None] * ab[None], axis=2)  # (K, W)
    j = dist.argmin(axis=1)
    k = np.arange(len(points))
    s = track.arclength[idx[j]] + t[k, j] * np.sqrt(seg_len2[j])
    return s, dist[k, j], idx[j]


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
