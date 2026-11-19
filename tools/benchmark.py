"""Benchmark de la simulation : pas/s, répartition du temps, coût d'un reset.

Lancement : .venv\\Scripts\\python tools\\benchmark.py
"""

from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sim.env import NeurolapEnv  # noqa: E402


def run_steps(n_steps: int, seed: int = 0) -> int:
    """Joue n_steps pas avec des actions aléatoires seedées (reset à chaque fin d'épisode)."""
    env = NeurolapEnv()
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)
    episodes = 0
    for _ in range(n_steps):
        env.step(rng.uniform(-1, 1, size=2))
        if env.done:
            episodes += 1
            env.reset(seed=seed + episodes)
    return episodes


def main() -> None:
    run_steps(500)  # échauffement

    n = 20_000
    t0 = time.perf_counter()
    episodes = run_steps(n)
    dt = time.perf_counter() - t0
    print(f"step (reset inclus) : {n / dt:,.0f} pas/s ({dt / n * 1e6:.0f} µs/pas, {episodes} ép.)")

    # Coût d'un reset isolé (génération du circuit)
    env = NeurolapEnv()
    k = 30
    t0 = time.perf_counter()
    for s in range(k):
        env.reset(seed=1000 + s)
    reset_ms = (time.perf_counter() - t0) / k * 1e3
    print(f"reset : {reset_ms:.1f} ms")

    # Pas seuls, sur un circuit fixe, sans fin d'épisode (max_steps élevé, actions douces)
    env = NeurolapEnv(max_steps=10**9)
    env.reset(seed=1)
    m = 10_000
    t0 = time.perf_counter()
    for _ in range(m):
        env.step([0.0, 0.0])
    dt = time.perf_counter() - t0
    print(f"step seul (voiture à l'arrêt) : {m / dt:,.0f} pas/s  ({dt / m * 1e6:.0f} µs/pas)")

    # Répartition du temps (cProfile)
    prof = cProfile.Profile()
    prof.enable()
    run_steps(5_000)
    prof.disable()
    print("\nRépartition (tri par temps cumulé, fonctions du projet) :")
    stats = pstats.Stats(prof).sort_stats("cumulative")
    stats.print_stats("sim[\\\\/]", 12)


if __name__ == "__main__":
    main()
