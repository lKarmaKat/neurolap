import numpy as np
import pytest

from agents.evolution import evaluate_genome, evolve
from agents.mlp import SIZES, MLPPilot, n_params


def test_n_params():
    assert n_params((9, 8, 2)) == 9 * 8 + 8 + 8 * 2 + 2


def test_pilot_output_in_bounds_and_deterministic():
    g = np.random.default_rng(0).normal(0, 1, n_params())
    obs = np.linspace(-1, 1, 9)
    a, b = MLPPilot(g)(obs), MLPPilot(g)(obs)
    assert a == b and len(a) == 2 and all(-1.0 <= v <= 1.0 for v in a)


def test_wrong_genome_size_raises():
    with pytest.raises(ValueError):
        MLPPilot(np.zeros(5))


def test_save_load_roundtrip(tmp_path):
    g = np.random.default_rng(1).normal(0, 1, n_params())
    MLPPilot(g).save(tmp_path / "m.npz")
    loaded = MLPPilot.load(tmp_path / "m.npz")
    assert np.array_equal(loaded.genome, g) and loaded.sizes == SIZES


def test_evaluation_is_deterministic():
    g = np.random.default_rng(2).normal(0, 0.5, n_params())
    assert evaluate_genome(g, [1, 2]) == evaluate_genome(g, [1, 2])


def test_evolution_with_elitism_never_regresses_on_fixed_circuits():
    best_per_gen: list[float] = []
    evolve(
        generations=4,
        pop_size=12,
        n_elite=3,
        seed=0,
        workers=0,
        fixed_seeds=[1, 2],
        on_generation=lambda g: best_per_gen.append(g.best_fitness),
    )
    assert all(b >= a for a, b in zip(best_per_gen, best_per_gen[1:], strict=False))


def test_evolution_is_reproducible():
    def run() -> np.ndarray:
        return evolve(generations=2, pop_size=8, n_elite=2, seed=5, workers=0, fixed_seeds=[3])

    assert np.array_equal(run(), run())
