"""Enregistrement des entraînements (écriture) et accès en lecture pour la rediffusion."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from agents.evolution import Generation

DEFAULT_ROOT = Path("models/runs")
NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
HISTORY_COLUMNS = ["generation", "best", "mean", "val_fitness", "val_finished", "val_lap_s"]


def _gen_path(run_dir: Path, index: int) -> Path:
    return run_dir / f"gen_{index:04d}.npz"


class RunRecorder:
    """Écrit un fichier par génération (population triée, fitness, seeds) et history.csv."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._history = self.run_dir / "history.csv"
        if not self._history.exists():
            with self._history.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(HISTORY_COLUMNS)

    def record(self, g: Generation, val: dict[str, float] | None = None) -> None:
        np.savez(
            _gen_path(self.run_dir, g.index),
            population=g.population,
            fitness=g.fitnesses,
            seeds=np.asarray(g.seeds, dtype=np.int64),
            generation=g.index,
        )
        v = val or {}
        row = [
            g.index,
            round(g.best_fitness, 4),
            round(g.mean_fitness, 4),
            *(round(v[k], 4) if k in v else "" for k in ("fitness", "finished", "lap_s")),
        ]
        with self._history.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(row)


class RunStore:
    """Lecture des entraînements enregistrés. Les noms sont validés (pas de chemin arbitraire)."""

    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def _dir(self, run: str) -> Path:
        if not NAME_RE.match(run):
            raise ValueError(f"nom d'entraînement invalide : {run!r}")
        return self.root / run

    def n_generations(self, run: str) -> int:
        d = self._dir(run)
        return len(list(d.glob("gen_*.npz"))) if d.is_dir() else 0

    def runs(self) -> list[dict[str, Any]]:
        """Entraînements disponibles, le plus récent d'abord."""
        if not self.root.is_dir():
            return []
        dirs = [d for d in self.root.iterdir() if d.is_dir() and NAME_RE.match(d.name)]
        dirs = [d for d in dirs if any(d.glob("gen_*.npz"))]
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return [{"name": d.name, "generations": self.n_generations(d.name)} for d in dirs]

    def load_generation(self, run: str, index: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
        with np.load(_gen_path(self._dir(run), index)) as data:
            return data["population"], data["fitness"], [int(s) for s in data["seeds"]]

    def history(self, run: str) -> list[dict[str, float | None]]:
        path = self._dir(run) / "history.csv"
        if not path.exists():
            return []
        with path.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return [
            {k: (float(v) if v not in ("", None) else None) for k, v in r.items()} for r in rows
        ]
