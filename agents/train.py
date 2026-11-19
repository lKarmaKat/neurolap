"""Entraînement par neuroévolution.

Lancement : .venv\\Scripts\\python -m agents.train --gens 60
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from agents.evolution import Generation, evolve, run_episode
from agents.mlp import MLPPilot
from agents.ray_pilot import RayPilot

VAL_SEEDS = list(range(10_000, 10_010))  # circuits jamais vus à l'entraînement


def report(name: str, pilot, seeds=VAL_SEEDS) -> float:
    """Affiche le taux de tours finis et le temps de tour moyen ; retourne la fitness moyenne."""
    res = [run_episode(pilot, s) for s in seeds]
    done = [r for r in res if r.finished]
    lap = f"{np.mean([r.steps for r in done]) / 30:.1f} s" if done else "-"
    fit = float(np.mean([r.fitness for r in res]))
    print(
        f"  [{name}] tours finis {len(done)}/{len(res)}, temps de tour moyen {lap}, "
        f"complétion moyenne {np.mean([r.completion for r in res]):.2f}, fitness {fit:.3f}"
    )
    return fit


def main() -> None:
    ap = argparse.ArgumentParser(description="Neuroévolution du pilote")
    ap.add_argument("--gens", type=int, default=60)
    ap.add_argument("--pop", type=int, default=64)
    ap.add_argument("--elite", type=int, default=8)
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--seeds-per-gen", type=int, default=3)
    ap.add_argument("--workers", type=int, default=None, help="processus (défaut : tous les cœurs)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-every", type=int, default=5)
    ap.add_argument("--out", default="models/best.npz")
    args = ap.parse_args()

    print("Référence (pilote réflexe) sur les circuits de validation :")
    baseline = report("RayPilot", RayPilot())

    state = {"best_val": -1.0, "t0": time.time()}

    def on_generation(g: Generation) -> None:
        print(
            f"gen {g.index:3d}  meilleur {g.best_fitness:.3f}  moyenne {g.mean_fitness:.3f}  "
            f"({time.time() - state['t0']:.0f} s)"
        )
        if (g.index + 1) % args.val_every == 0:
            pilot = MLPPilot(g.best_genome)
            val = report("validation", pilot)
            if val > state["best_val"]:
                state["best_val"] = val
                pilot.save(args.out)
                print(f"  -> sauvegardé dans {args.out}")

    evolve(
        generations=args.gens,
        pop_size=args.pop,
        n_elite=args.elite,
        sigma=args.sigma,
        seeds_per_gen=args.seeds_per_gen,
        seed=args.seed,
        workers=args.workers,
        on_generation=on_generation,
    )
    print(f"\nMeilleure validation : {state['best_val']:.3f} (référence RayPilot : {baseline:.3f})")


if __name__ == "__main__":
    main()
