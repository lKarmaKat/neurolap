"""Capteurs : raycasts vers les murs (intersection rayon-segment vectorisée)."""

from __future__ import annotations

import numpy as np

from sim.track import Track

# 7 rayons de -90° à +90° par rapport au cap (négatif = gauche à l'écran, y vers le bas)
RAY_ANGLES = np.radians(np.linspace(-90.0, 90.0, 7))
MAX_RANGE = 50.0  # portée des rayons (m)


def wall_segments(track: Track) -> tuple[np.ndarray, np.ndarray]:
    """Segments (début, fin) des deux murs fermés, concaténés : deux tableaux (S, 2)."""
    a = np.concatenate([track.left, track.right])
    b = np.concatenate([np.roll(track.left, -1, axis=0), np.roll(track.right, -1, axis=0)])
    return a, b


RAY_WINDOW = 70  # demi-fenêtre (en segments, ~1 m chacun) autour de la voiture, > MAX_RANGE


def window_segments(
    seg_a: np.ndarray, seg_b: np.ndarray, k: int, half: int = RAY_WINDOW
) -> tuple[np.ndarray, np.ndarray]:
    """Segments des deux murs proches de l'indice de ligne centrale k (indices circulaires).

    Le segment i du mur gauche est à l'indice i, celui du mur droit à i + N (voir wall_segments).
    """
    n = len(seg_a) // 2
    idx = (k + np.arange(-half, half + 1)) % n
    both = np.concatenate([idx, idx + n])
    return seg_a[both], seg_b[both]


def cast_rays(
    x: float,
    y: float,
    heading: float,
    seg_a: np.ndarray,
    seg_b: np.ndarray,
    angles: np.ndarray = RAY_ANGLES,
    max_range: float = MAX_RANGE,
) -> np.ndarray:
    """Distance (m) au premier mur pour chaque rayon, plafonnée à `max_range`. Forme (R,)."""
    theta = heading + angles
    d = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (R, 2) directions
    e = seg_b - seg_a  # (S, 2)
    ap = seg_a - np.array([x, y])  # (S, 2)
    denom = d[:, 0:1] * e[None, :, 1] - d[:, 1:2] * e[None, :, 0]  # (R, S)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ap[:, 0] * e[:, 1] - ap[:, 1] * e[:, 0])[None, :] / denom
        u = (ap[:, 0][None, :] * d[:, 1:2] - ap[:, 1][None, :] * d[:, 0:1]) / denom
    hit = (np.abs(denom) > 1e-12) & (t >= 0.0) & (u >= 0.0) & (u <= 1.0)
    return np.minimum(np.where(hit, t, np.inf).min(axis=1), max_range)
