"""Réseau de neurones (perceptron multicouche) utilisé comme pilote. NumPy pur, sans framework."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

SIZES = (9, 8, 2)  # 9 observations -> 8 neurones cachés -> 2 actions


def n_params(sizes: Sequence[int] = SIZES) -> int:
    """Nombre de poids + biais : c'est la taille du génome."""
    return sum(i * o + o for i, o in zip(sizes[:-1], sizes[1:], strict=True))


class MLPPilot:
    """Observation (9,) -> action [steering, throttle]. Les poids viennent d'un génome plat."""

    def __init__(self, genome: np.ndarray, sizes: Sequence[int] = SIZES) -> None:
        self.sizes = tuple(sizes)
        self.genome = np.asarray(genome, dtype=np.float64)
        if self.genome.shape != (n_params(self.sizes),):
            raise ValueError(f"génome de taille {self.genome.shape}, attendu ({n_params(sizes)},)")
        self._layers: list[tuple[np.ndarray, np.ndarray]] = []
        offset = 0
        for i, o in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            w = self.genome[offset : offset + i * o].reshape(i, o)
            offset += i * o
            b = self.genome[offset : offset + o]
            offset += o
            self._layers.append((w, b))

    def __call__(self, obs: np.ndarray) -> list[float]:
        x = np.asarray(obs, dtype=np.float64)
        for w, b in self._layers:
            x = np.tanh(x @ w + b)  # tanh : sortie dans [-1, 1], comme l'action attendue
        return [float(v) for v in x]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, genome=self.genome, sizes=np.array(self.sizes))

    @classmethod
    def load(cls, path: str | Path) -> MLPPilot:
        data = np.load(path)
        return cls(data["genome"], tuple(int(s) for s in data["sizes"]))
