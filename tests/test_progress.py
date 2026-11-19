import math

import numpy as np

from sim.car import CarState
from sim.env import NeurolapEnv
from sim.progress import ProgressTracker, project, wrap_delta


def test_wrap_delta():
    assert math.isclose(wrap_delta(-98.0, 100.0), 2.0)  # passage de la ligne de départ
    assert math.isclose(wrap_delta(98.0, 100.0), -2.0)
    assert math.isclose(wrap_delta(3.0, 100.0), 3.0)


def test_project_on_center_and_offset():
    env = NeurolapEnv()
    env.reset(seed=1)
    t = env.track
    i = 100
    s, lat, _ = project(t, np.array([t.center[i], t.right[i]]))
    assert np.isclose(s[0], t.arclength[i], atol=0.6) and lat[0] < 1e-6
    assert np.isclose(lat[1], t.width / 2, atol=0.1)


def test_backward_earns_nothing_and_no_farming():
    p = ProgressTracker(length=100.0, s0=0.0)
    total = p.update(10.0) + p.update(5.0) + p.update(10.0)  # avance, recule, re-avance
    assert total == 10.0 and p.best == 10.0
    total += p.update(5.0) + p.update(10.0)  # oscillation
    assert total == 10.0


def test_wrap_over_start_line_counts_forward():
    p = ProgressTracker(length=100.0, s0=95.0)
    assert math.isclose(p.update(3.0), 8.0)  # 95 -> 3 en passant la ligne
    assert math.isclose(p.completion, 0.08)


def test_collision_terminates():
    env = NeurolapEnv()
    env.reset(seed=2)
    t = env.track
    x, y = t.center[100] + 2 * (t.right[100] - t.center[100])
    env.car = CarState(x=float(x), y=float(y), heading=t.start_heading)
    _, reward, terminated, truncated, info = env.step([0.0, 0.0])
    assert terminated and not truncated and info["collision"] and reward == 0.0


def pilot(env: NeurolapEnv) -> list[float]:
    """Pilote scripté (pursuit sur la ligne centrale), seulement pour tester l'env."""
    c = env.track.center
    n = len(c)
    d = np.linalg.norm(c - [env.car.x, env.car.y], axis=1)
    target = c[(int(d.argmin()) + 8) % n]
    err = math.atan2(target[1] - env.car.y, target[0] - env.car.x) - env.car.heading
    err = (err + math.pi) % (2 * math.pi) - math.pi
    return [float(np.clip(2.0 * err, -1, 1)), 1.0 if env.car.speed < 10.0 else 0.0]


def test_full_lap_completes_and_rewards_sum_to_one():
    env = NeurolapEnv(max_steps=5000)
    env.reset(seed=3)
    total, info, terminated = 0.0, {}, False
    while not env.done:
        _, r, terminated, _, info = env.step(pilot(env))
        total += r
        assert 0.0 <= info["completion"] <= 1.0
    assert terminated and not info["collision"]
    assert info["completion"] == 1.0 and math.isclose(total, 1.0, abs_tol=1e-9)


def test_step_after_done_raises():
    import pytest

    env = NeurolapEnv(max_steps=1)
    env.reset()
    env.step([0.0, 0.0])
    with pytest.raises(RuntimeError):
        env.step([0.0, 0.0])


def _check_window_matches_full(env: NeurolapEnv) -> None:
    """Rayons et projection fenêtrés == recherche sur tous les segments."""
    from sim.sensors import cast_rays, wall_segments

    c = env.car
    full_rays = cast_rays(c.x, c.y, c.heading, *wall_segments(env.track))
    assert np.allclose(env._rays(), full_rays)
    pts = np.array([[c.x, c.y]])
    s_full, lat_full, _ = project(env.track, pts)
    s_win, lat_win, _ = project(env.track, pts, hint=env._k)
    # s est défini modulo la longueur du circuit (0 et L = même point : ligne de départ)
    assert abs(wrap_delta(float(s_full[0] - s_win[0]), env.track.length)) < 1e-6
    assert np.allclose(lat_full, lat_win)


def test_windowed_search_matches_full_on_laps():
    for seed in (0, 3, 8):
        env = NeurolapEnv(max_steps=5000)
        env.reset(seed=seed)
        while not env.done:
            env.step(pilot(env))
            _check_window_matches_full(env)


def test_windowed_search_matches_full_with_random_actions():
    rng = np.random.default_rng(0)
    for seed in range(20):
        env = NeurolapEnv()
        env.reset(seed=seed)
        _check_window_matches_full(env)
        while not env.done:
            env.step(rng.uniform(-1, 1, size=2))
            _check_window_matches_full(env)
