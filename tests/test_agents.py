from agents.ray_pilot import RayPilot
from sim.env import NeurolapEnv


def test_ray_pilot_completes_laps():
    pilot = RayPilot()
    for seed in range(8):
        env = NeurolapEnv()
        obs, _ = env.reset(seed=seed)
        while not env.done:
            obs, _, _, _, info = env.step(pilot(obs))
        assert info["completion"] == 1.0 and not info["collision"], f"seed {seed}"


def test_ray_pilot_action_in_bounds_and_deterministic():
    env = NeurolapEnv()
    obs, _ = env.reset(seed=1)
    a, b = RayPilot()(obs), RayPilot()(obs)
    assert a == b and all(-1.0 <= v <= 1.0 for v in a)
