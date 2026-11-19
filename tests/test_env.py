import numpy as np
import pytest

from sim.env import NeurolapEnv


def rollout(seed: int, actions: list[list[float]]) -> list[np.ndarray]:
    env = NeurolapEnv()
    obs, _ = env.reset(seed=seed)
    out = [obs]
    for a in actions:
        obs, _, terminated, truncated, _ = env.step(a)
        out.append(obs)
        if terminated or truncated:
            break
    return out


def test_determinism_same_seed():
    actions = [[0.3, 1.0]] * 20 + [[-0.7, 0.5]] * 20
    for x, y in zip(rollout(42, actions), rollout(42, actions), strict=True):
        assert np.array_equal(x, y)


def test_action_is_clipped():
    a = rollout(0, [[5.0, -9.0]] * 10)
    b = rollout(0, [[1.0, -1.0]] * 10)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_bad_action_shape_raises():
    env = NeurolapEnv()
    env.reset()
    with pytest.raises(ValueError):
        env.step([0.0, 0.0, 0.0])


def test_observation_in_range():
    obs = rollout(0, [[1.0, 1.0]] * 200)
    assert all(np.all(np.abs(o) <= 1.0) for o in obs)


def test_truncated_exactly_at_max_steps():
    env = NeurolapEnv(max_steps=5)
    env.reset()
    flags = [env.step([0.0, 0.0])[3] for _ in range(5)]
    assert flags == [False] * 4 + [True]


def test_reset_places_car_on_start_and_same_seed_same_track():
    e1, e2 = NeurolapEnv(), NeurolapEnv()
    e1.reset(seed=9)
    e2.reset(seed=9)
    assert np.array_equal(e1.track.center, e2.track.center)
    assert (e1.car.x, e1.car.y) == e1.track.start_pos
    assert e1.car.heading == e1.track.start_heading
