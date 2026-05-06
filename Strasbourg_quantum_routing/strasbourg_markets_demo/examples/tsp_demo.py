# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""Strasbourg-markets TSP demo.

Builds the five-Christmas-market TSP instance (great-circle distances in
metres) and solves it three ways: brute force, Held–Karp dynamic
programming, and simulated annealing. Then runs a random 15-city instance
to show how brute force times out on the larger size while DP and SA both
return promptly.

Run with:
    cd strasbourg_markets_demo
    uv run python examples/tsp_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

# Make the in-repo package importable when this script is run directly with
# `uv run python examples/tsp_demo.py` and no package is installed in site-packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from strasbourg_markets_demo.tsp import (
    TSPInstance,
    brute_force,
    held_karp,
    simulated_annealing,
    two_opt,
)


def _format_tour(instance: TSPInstance, tour: np.ndarray) -> str:
    if instance.names is not None:
        names = [instance.names[i] for i in tour]
    else:
        names = [str(i) for i in tour]
    return " -> ".join(names) + f" -> {names[0]}"


def _run(
    name: str,
    instance: TSPInstance,
    solver: Callable[[TSPInstance], tuple[np.ndarray, float]],
    *,
    quiet_tour: bool = False,
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    tour, length = solver(instance)
    elapsed = time.perf_counter() - t0
    valid = instance.is_valid_tour(tour)
    print(f"--- {name} ---")
    if not quiet_tour:
        print(f"  tour          : {_format_tour(instance, tour)}")
    else:
        print(f"  tour (head)   : {tour[:6].tolist()} ... ({len(tour)} cities)")
    print(f"  length        : {length:.4f}")
    print(f"  valid         : {valid}")
    print(f"  wall-clock    : {elapsed * 1000:.3f} ms")
    print()
    return tour, length, elapsed


def strasbourg_demo() -> None:
    print("=" * 72)
    print("Strasbourg Christmas markets TSP   (5 cities, great-circle, metres)")
    print("=" * 72)
    instance = TSPInstance.strasbourg_markets()
    print(f"  cities  : {instance.n}")
    print(f"  names   : {instance.names}")
    print(f"  max d   : {instance.distances.max():.1f} m")
    print(f"  mean d  : {instance.distances[np.triu_indices(instance.n, 1)].mean():.1f} m")
    print()

    bf_tour, bf_len, _ = _run("brute_force", instance, brute_force)
    hk_tour, hk_len, _ = _run("held_karp", instance, held_karp)
    sa_tour, sa_len, _ = _run("simulated_annealing", instance,
                              lambda I: simulated_annealing(I, alpha=0.95))

    print(f"brute force vs held-karp agree: {np.allclose(bf_len, hk_len)}")
    print(f"SA gap to optimum             : {(sa_len - bf_len) / bf_len * 100:.3f} %")
    print()


def random_demo() -> None:
    print("=" * 72)
    print("Random Euclidean TSP   (n=15 cities, unit square)")
    print("=" * 72)
    instance = TSPInstance.random(n=15, seed=2026, coord_range=(0.0, 100.0))
    print(f"  cities  : {instance.n}")
    print()
    print("brute_force on n=15 enumerates 14! ≈ 8.7e10 tours -> skipped on purpose")
    print()

    hk_tour, hk_len, _ = _run("held_karp", instance, held_karp, quiet_tour=True)
    two_tour, two_len, _ = _run("two_opt", instance, two_opt, quiet_tour=True)
    sa_tour, sa_len, _ = _run("simulated_annealing", instance,
                              lambda I: simulated_annealing(I, alpha=0.95),
                              quiet_tour=True)

    print(f"2-opt gap to optimum  : {(two_len - hk_len) / hk_len * 100:.3f} %")
    print(f"SA gap to optimum     : {(sa_len - hk_len) / hk_len * 100:.3f} %")
    print()


def main() -> None:
    strasbourg_demo()
    random_demo()


if __name__ == "__main__":
    main()
