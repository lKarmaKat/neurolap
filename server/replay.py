"""Rediffusion d'une génération : toute la population roule en même temps sur un même circuit."""

from __future__ import annotations

from typing import Any

import numpy as np

from agents.evolution import Episode
from agents.mlp import MLPPilot
from agents.runs import RunStore
from server.protocol import encode_track
from sim.env import NeurolapEnv
from sim.track import Track, generate_track

SPEEDS = (1, 2, 4, 8)  # pas de simulation par tick d'affichage (x1 = temps réel)
RESTART_PAUSE_TICKS = 60  # pause (2 s) quand tout le monde a fini, puis la rediffusion reboucle
REFRESH_TICKS = 60  # on relit le disque toutes les 2 s (entraînement en cours)


class ReplaySession:
    def __init__(self, store: RunStore) -> None:
        self.store = store
        self.run: str | None = None
        self.generation = 0
        self.n_generations = 0
        self.circuit = 0
        self.n_circuits = 0
        self.speed = 1
        self._episodes: list[Episode] = []
        self._track: Track | None = None
        self._seed = 0
        self._fitness: np.ndarray = np.zeros(0)
        self._pause = 0
        self._ticks = 0

    # --- messages ---------------------------------------------------------------------------
    def _runs_msg(self) -> dict[str, Any]:
        return {"type": "runs", "runs": self.store.runs(), "current": self.run}

    def _history_msg(self) -> dict[str, Any]:
        rows = self.store.history(self.run) if self.run else []
        return {"type": "history", "run": self.run, "rows": rows}

    def _state_msg(self) -> dict[str, Any]:
        cars = [
            {
                "x": round(ep.env.car.x, 2),
                "y": round(ep.env.car.y, 2),
                "heading": round(ep.env.car.heading, 3),
                "status": ep.status,
                "completion": round(ep.info["completion"], 3),
            }
            for ep in self._episodes
        ]
        return {
            "type": "replay_state",
            "run": self.run,
            "generation": self.generation,
            "n_generations": self.n_generations,
            "circuit": self.circuit,
            "n_circuits": self.n_circuits,
            "speed": self.speed,
            "step": self._episodes[0].env.steps if self._episodes else 0,
            "running": sum(ep.status == "running" for ep in self._episodes),
            "fitness": [round(float(f), 3) for f in self._fitness],
            "cars": cars,
        }

    def snapshot(self) -> list[dict[str, Any]]:
        msgs = [self._runs_msg(), self._history_msg()]
        if self._episodes and self._track is not None:
            env = NeurolapEnv(max_steps=1, track=self._track)
            msgs += [encode_track(env, self._seed), self._state_msg()]
        return msgs

    # --- sélection ---------------------------------------------------------------------------
    def select(
        self,
        run: str | None = None,
        generation: int | None = None,
        circuit: int | None = None,
        speed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Choisit entraînement, génération, circuit et vitesse (absents : valeurs gardées)."""
        if speed is not None and speed in SPEEDS:
            self.speed = speed
        runs = self.store.runs()
        if not runs:
            self.run, self._episodes, self._track = None, [], None
            return self.snapshot()
        names = [r["name"] for r in runs]
        new_run = run if run in names else (self.run if self.run in names else names[0])
        run_changed = new_run != self.run
        self.run = new_run
        self.n_generations = self.store.n_generations(self.run)
        if generation is None:
            generation = (
                self.n_generations - 1 if run_changed or self._track is None else self.generation
            )
        self.generation = max(0, min(generation, self.n_generations - 1))
        pop, fit, seeds = self.store.load_generation(self.run, self.generation)
        self.n_circuits = len(seeds)
        self.circuit = max(
            0, min(self.circuit if circuit is None else circuit, self.n_circuits - 1)
        )
        self._seed = seeds[self.circuit]
        self._track = generate_track(np.random.default_rng(self._seed))
        self._fitness = fit
        self._pilots = [MLPPilot(g) for g in pop]
        self._start()
        return self.snapshot()

    def _start(self) -> None:
        assert self._track is not None
        self._episodes = [Episode(p, self._seed, self._track) for p in self._pilots]
        self._pause = 0

    # --- boucle ------------------------------------------------------------------------------
    def tick(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        self._ticks += 1
        if self.run and self._ticks % REFRESH_TICKS == 0:
            n = self.store.n_generations(self.run)
            if n != self.n_generations:  # l'entraînement a avancé
                self.n_generations = n
                out += [self._runs_msg(), self._history_msg()]
        if not self._episodes:
            return out
        if all(ep.done for ep in self._episodes):
            self._pause += 1
            if self._pause >= RESTART_PAUSE_TICKS:
                self._start()
        else:
            for _ in range(self.speed):
                for ep in self._episodes:
                    if not ep.done:
                        ep.step()
        out.append(self._state_msg())
        return out
