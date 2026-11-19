"""Circuit : générateur procédural (cercle bruité -> spline fermée -> murs).

Repère écran (y vers le bas), comme le Canvas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Track:
    center: np.ndarray  # (N, 2) ligne centrale, circuit fermé (pas de doublon du 1er point)
    left: np.ndarray  # (N, 2) mur gauche
    right: np.ndarray  # (N, 2) mur droit
    width: float  # largeur de la piste (m)
    arclength: np.ndarray  # (N,) abscisse curviligne de chaque point de la ligne centrale
    length: float  # longueur totale du circuit (m)
    start_pos: tuple[float, float]
    start_heading: float


def _catmull_rom_closed(pts: np.ndarray, per_segment: int = 40) -> np.ndarray:
    """Spline Catmull-Rom fermée passant par tous les points. Retourne un polygone dense."""
    p0, p1, p2, p3 = (np.roll(pts, k, axis=0) for k in (1, 0, -1, -2))
    t = np.linspace(0.0, 1.0, per_segment, endpoint=False)[None, :, None]
    a, b, c, d = (
        v[:, None, :]
        for v in (2 * p1, p2 - p0, 2 * p0 - 5 * p1 + 4 * p2 - p3, 3 * p1 - p0 - 3 * p2 + p3)
    )
    return (0.5 * (a + b * t + c * t**2 + d * t**3)).reshape(-1, 2)


def _resample(poly: np.ndarray, spacing: float) -> np.ndarray:
    """Redécoupe un polygone fermé en points équidistants (abscisse curviligne)."""
    closed = np.vstack([poly, poly[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(round(s[-1] / spacing)), 8)
    targets = np.linspace(0.0, s[-1], n, endpoint=False)
    return np.stack([np.interp(targets, s, closed[:, 0]), np.interp(targets, s, closed[:, 1])], 1)


def _is_valid(center: np.ndarray, width: float, spacing: float) -> bool:
    n = len(center)
    t = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
    ang = np.arctan2(t[:, 1], t[:, 0])
    dang = np.abs((np.diff(np.append(ang, ang[0])) + math.pi) % (2 * math.pi) - math.pi)
    if dang.max() / spacing > 1.0 / (0.6 * width):  # rayon de virage min = 0,6 x largeur
        return False
    # deux portions éloignées le long du tracé ne doivent pas se frôler
    idx = np.arange(n)
    gap = np.abs(idx[:, None] - idx[None, :])
    gap = np.minimum(gap, n - gap) * spacing
    dist = np.linalg.norm(center[:, None, :] - center[None, :, :], axis=2)
    return not np.any((gap > 2 * width) & (dist < width + 4.0))


def generate_track(
    rng: np.random.Generator,
    n_control: int = 12,
    r_min: float = 55.0,
    r_max: float = 105.0,
    width: float = 12.0,
    spacing: float = 1.0,
    max_attempts: int = 50,
) -> Track:
    """Génère un circuit valide. Déterministe pour un état de `rng` donné."""
    step = 2 * math.pi / n_control
    for _ in range(max_attempts):
        angles = np.arange(n_control) * step + rng.uniform(-0.4, 0.4, n_control) * step
        radii = rng.uniform(r_min, r_max, n_control)
        ctrl = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
        center = _resample(_catmull_rom_closed(ctrl), spacing)
        seg = np.linalg.norm(np.roll(center, -1, axis=0) - center, axis=1)  # inclut le bouclage
        if not _is_valid(center, width, float(seg.mean())):
            continue
        t = np.roll(center, -1, axis=0) - np.roll(center, 1, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        right_n = np.stack([-t[:, 1], t[:, 0]], axis=1)  # y vers le bas : droite = (-ty, tx)
        return Track(
            center=center,
            left=center - right_n * width / 2,
            right=center + right_n * width / 2,
            width=width,
            arclength=np.concatenate([[0.0], np.cumsum(seg)[:-1]]),
            length=float(seg.sum()),
            start_pos=(float(center[0, 0]), float(center[0, 1])),
            start_heading=float(math.atan2(t[0, 1], t[0, 0])),
        )
    raise RuntimeError(f"aucun circuit valide après {max_attempts} tentatives")
