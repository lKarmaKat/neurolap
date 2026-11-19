import numpy as np

from sim.env import NeurolapEnv
from sim.sensors import MAX_RANGE, RAY_ANGLES, cast_rays


def corridor() -> tuple[np.ndarray, np.ndarray]:
    """Couloir rectangulaire : murs y=±6 sur x in [0, 40], fond x=40, arrière x=0."""
    pts = np.array([[0, -6], [40, -6], [40, 6], [0, 6]], dtype=float)
    return pts, np.roll(pts, -1, axis=0)


def test_known_geometry():
    a, b = corridor()
    d = cast_rays(10.0, 0.0, 0.0, a, b)
    assert np.isclose(d[3], 30.0)  # 0° : fond du couloir
    assert np.isclose(d[0], 6.0) and np.isclose(d[6], 6.0)  # ±90° : murs latéraux


def test_diagonal_ray():
    a, b = corridor()
    d = cast_rays(10.0, 0.0, 0.0, a, b)
    assert np.isclose(d[5], 6.0 / np.sin(RAY_ANGLES[5]))  # +60° : touche le mur y=6


def test_max_range_when_nothing_hit():
    a, b = corridor()
    d = cast_rays(10.0, 0.0, 0.0, a * 1000, b * 1000)
    assert np.all(d == MAX_RANGE)


def test_obs_shape_and_bounds_on_generated_track():
    env = NeurolapEnv()
    obs, info = env.reset(seed=4)
    assert obs.shape == (9,) and info["ray_distances"].shape == (7,)
    assert np.all(obs[:7] >= 0) and np.all(obs[:7] <= 1)
    assert abs(obs[2] - obs[4]) < 0.2  # voiture centrée au départ : ±60° quasi symétriques
    assert abs(obs[0] - obs[6]) < 0.2  # ±90° aussi
