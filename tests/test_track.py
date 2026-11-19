import numpy as np

from sim.track import generate_track


def make(seed: int):
    return generate_track(np.random.default_rng(seed))


def _cross(o, a, b):
    return (a[..., 0] - o[..., 0]) * (b[..., 1] - o[..., 1]) - (a[..., 1] - o[..., 1]) * (
        b[..., 0] - o[..., 0]
    )


def _crossings(a0, a1, b0, b1):
    """Matrice booléenne : le segment i de A croise-t-il le segment j de B ?"""
    a0, a1 = a0[:, None, :], a1[:, None, :]
    b0, b1 = b0[None, :, :], b1[None, :, :]
    d1, d2 = _cross(a0, a1, b0), _cross(a0, a1, b1)
    d3, d4 = _cross(b0, b1, a0), _cross(b0, b1, a1)
    return (d1 * d2 < 0) & (d3 * d4 < 0)


def test_same_seed_same_track():
    a, b = make(3), make(3)
    assert np.array_equal(a.center, b.center) and np.array_equal(a.left, b.left)


def test_different_seed_different_track():
    a, b = make(1), make(2)
    assert a.center.shape != b.center.shape or not np.allclose(a.center, b.center)


def test_closed_and_uniform_spacing():
    t = make(5)
    seg = np.linalg.norm(np.roll(t.center, -1, axis=0) - t.center, axis=1)  # inclut le bouclage
    assert seg.max() / seg.min() < 1.1
    assert abs(t.length - seg.sum()) < 1e-6


def test_width_between_walls():
    t = make(7)
    assert np.allclose(np.linalg.norm(t.left - t.right, axis=1), t.width)


def test_walls_do_not_intersect():
    for seed in range(10):
        t = make(seed)
        n = len(t.center)
        for wall in (t.left, t.right):
            c = _crossings(wall, np.roll(wall, -1, axis=0), wall, np.roll(wall, -1, axis=0))
            idx = np.arange(n)
            gap = np.abs(idx[:, None] - idx[None, :])
            c &= np.minimum(gap, n - gap) > 1  # on ignore les segments voisins
            assert not c.any()
        lr = _crossings(t.left, np.roll(t.left, -1, axis=0), t.right, np.roll(t.right, -1, axis=0))
        assert not lr.any()


def test_100_seeds_all_valid():
    for seed in range(100):
        t = make(seed)
        assert t.length > 0
