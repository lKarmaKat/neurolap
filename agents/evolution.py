"""Neuroévolution : évaluation (fitness) et boucle d'évolution d'une population de génomes."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from agents.mlp import SIZES, MLPPilot, n_params
from sim.env import NeurolapEnv
from sim.track import Track, generate_track

MAX_STEPS = 1500  # 50 s : assez pour un tour, et borne le coût d'une voiture qui traîne
STALL_STEPS = 150  # arrêt si aucun nouveau record de progression pendant 5 s


@dataclass(frozen=True)
class EpisodeResult:
    completion: float
    steps: int
    finished: bool  # tour complet sans collision

    @property
    def fitness(self) -> float:
        """Complétion (0..1), plus un bonus de vitesse (0..1) si le tour est fini."""
        bonus = (MAX_STEPS - self.steps) / MAX_STEPS if self.finished else 0.0
        return self.completion + bonus


class Episode:
    """Un épisode pas à pas (même logique à l'entraînement et dans la rediffusion)."""

    def __init__(
        self,
        pilot: Callable[[np.ndarray], Sequence[float]],
        seed: int,
        track: Track | None = None,
    ) -> None:
        # Le circuit ne dépend que de la seed ; on peut donc le partager entre épisodes.
        track = track or generate_track(np.random.default_rng(seed))
        self.pilot = pilot
        self.env = NeurolapEnv(max_steps=MAX_STEPS, track=track)
        self.obs, _ = self.env.reset(seed=seed, track=track)
        self.best, self.last_gain = 0.0, 0
        self.stalled = False
        self.info: dict = {"completion": 0.0, "collision": False}

    @property
    def done(self) -> bool:
        return self.env.done or self.stalled

    def step(self) -> None:
        self.obs, _, _, _, self.info = self.env.step(self.pilot(self.obs))
        if self.info["completion"] > self.best + 1e-9:
            self.best, self.last_gain = self.info["completion"], self.env.steps
        elif self.env.steps - self.last_gain >= STALL_STEPS:
            self.stalled = True  # la voiture est bloquée ou tourne en rond

    @property
    def finished(self) -> bool:
        return self.info["completion"] >= 1.0 and not self.info["collision"]

    @property
    def status(self) -> str:
        if not self.done:
            return "running"
        if self.finished:
            return "finished"
        if self.info["collision"]:
            return "crashed"
        return "stalled" if self.stalled else "timeout"

    def result(self) -> EpisodeResult:
        return EpisodeResult(self.info["completion"], self.env.steps, self.finished)


def run_episode(pilot: Callable[[np.ndarray], Sequence[float]], seed: int) -> EpisodeResult:
    ep = Episode(pilot, seed)
    while not ep.done:
        ep.step()
    return ep.result()


def evaluate_pilot(pilot: Callable[[np.ndarray], Sequence[float]], seeds: Iterable[int]) -> float:
    """Fitness moyenne sur plusieurs circuits."""
    results = [run_episode(pilot, s) for s in seeds]
    return float(np.mean([r.fitness for r in results]))


def evaluate_genome(
    genome: np.ndarray, seeds: Sequence[int], sizes: Sequence[int] = SIZES
) -> float:
    return evaluate_pilot(MLPPilot(genome, sizes), seeds)


def _task(args: tuple[np.ndarray, Sequence[int], Sequence[int]]) -> float:
    return evaluate_genome(*args)  # fonction de module : picklable pour les processus


@dataclass
class Generation:
    index: int
    best_fitness: float
    mean_fitness: float
    best_genome: np.ndarray
    seeds: Sequence[int]
    population: np.ndarray  # tous les génomes, triés par fitness décroissante
    fitnesses: np.ndarray  # fitness correspondantes (même ordre)


def evolve(
    generations: int = 50,
    pop_size: int = 64,
    n_elite: int = 8,
    sigma: float = 0.1,
    seeds_per_gen: int = 3,
    seed: int = 0,
    workers: int | None = 0,
    sizes: Sequence[int] = SIZES,
    fixed_seeds: Sequence[int] | None = None,
    on_generation: Callable[[Generation], None] | None = None,
) -> np.ndarray:
    """Fait évoluer une population et retourne le meilleur génome de la dernière génération.

    workers : nombre de processus (0 = en série, dans ce processus ; None = tous les cœurs).
    fixed_seeds : circuits fixes (sinon de nouveaux circuits à chaque génération).
    """
    rng = np.random.default_rng(seed)
    pop = rng.normal(0.0, 0.5, size=(pop_size, n_params(sizes)))
    executor = ProcessPoolExecutor(max_workers=workers) if workers != 0 else None
    best = pop[0]
    try:
        for gen in range(generations):
            seeds = (
                list(fixed_seeds)
                if fixed_seeds is not None
                else [int(s) for s in rng.integers(0, 2**31, size=seeds_per_gen)]
            )
            tasks = [(g, seeds, sizes) for g in pop]
            fit = np.array(
                list(executor.map(_task, tasks)) if executor else [_task(t) for t in tasks]
            )
            order = np.argsort(-fit, kind="stable")
            elites = pop[order[:n_elite]]
            best = elites[0].copy()
            if on_generation:
                on_generation(
                    Generation(
                        gen,
                        float(fit[order[0]]),
                        float(fit.mean()),
                        best,
                        seeds,
                        pop[order].copy(),
                        fit[order].copy(),
                    )
                )
            # Élitisme : les meilleurs passent tels quels, les autres sont des élites mutées.
            parents = elites[rng.integers(0, n_elite, size=pop_size - n_elite)]
            children = parents + rng.normal(0.0, sigma, size=parents.shape)
            pop = np.vstack([elites, children])
    finally:
        if executor:
            executor.shutdown()
    return best
