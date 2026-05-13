# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""Strasbourg bike-couriers m-VRP demo.

Builds the five-market multi-vehicle TSP (m-VRP) centred on the Gare
Centrale depot — three bike couriers cover all six Christmas-market stops
(five markets + Petite France). Solves it four ways: brute force (the
proven optimum), nearest-neighbour, Clarke-Wright savings, and Google
OR-tools. Also demonstrates the same solvers on a random 20-customer /
4-vehicle instance (without brute force, which is intractable there).

Run with:
    cd strasbourg_markets_demo
    uv run python examples/vrp_demo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

# Make the in-repo package importable when this script is run directly with
# `uv run python examples/vrp_demo.py` and no package is installed in site-packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from strasbourg_markets_demo.vrp import (
    VRPInstance,
    clarke_wright_savings,
    m_vrp_brute_force,
    nearest_neighbour,
    or_tools_solve,
)


def _format_route(instance: VRPInstance, route: list[int]) -> str:
    if not route:
        return "(empty)"
    if instance.names is not None:
        names = [instance.names[c] for c in route]
    else:
        names = [str(c) for c in route]
    load = sum(instance.demands[c - 1] for c in route)
    return f"depot -> {' -> '.join(names)} -> depot   [load={load:.1f}]"


def _run(
    name: str,
    instance: VRPInstance,
    solver: Callable[[VRPInstance], list[list[int]]],
) -> None:
    t0 = time.perf_counter()
    routes = solver(instance)
    elapsed = time.perf_counter() - t0
    total = instance.total_distance(routes)
    ok, reason = instance.is_feasible(routes)
    print(f"--- {name} ---")
    for k, r in enumerate(routes):
        print(f"  vehicle {k}: {_format_route(instance, r)}")
    print(f"  total distance : {total:.4f}")
    print(f"  feasible       : {ok}  ({reason})")
    print(f"  wall-clock     : {elapsed * 1000:.2f} ms")
    print()


def _run_or_tools(name: str, instance: VRPInstance, time_limit_s: int = 5) -> None:
    t0 = time.perf_counter()
    routes, total = or_tools_solve(instance, time_limit_s=time_limit_s)
    elapsed = time.perf_counter() - t0
    ok, reason = instance.is_feasible(routes)
    print(f"--- {name} ---")
    for k, r in enumerate(routes):
        print(f"  vehicle {k}: {_format_route(instance, r)}")
    print(f"  total distance : {total:.4f}")
    print(f"  feasible       : {ok}  ({reason})")
    print(f"  wall-clock     : {elapsed * 1000:.2f} ms")
    print()


def _run_brute(name: str, instance: VRPInstance) -> None:
    t0 = time.perf_counter()
    routes, total = m_vrp_brute_force(instance)
    elapsed = time.perf_counter() - t0
    ok, reason = instance.is_feasible(routes)
    print(f"--- {name} ---")
    for k, r in enumerate(routes):
        print(f"  courier {k}: {_format_route(instance, r)}")
    print(f"  total distance : {total:.4f}")
    print(f"  feasible       : {ok}  ({reason})")
    print(f"  wall-clock     : {elapsed * 1000:.2f} ms")
    print()


def strasbourg_demo() -> None:
    print("=" * 72)
    print("Strasbourg bike-couriers m-VRP   (Gare Centrale -> 5 markets, K=3)")
    print("=" * 72)
    print("Three bike couriers leaving the Gare de Strasbourg deliver mulled-wine")
    print("glasses to every Christmas market plus Petite France. Each courier")
    print("starts and ends at the station; minimise total cycling distance.")
    print()
    instance = VRPInstance.strasbourg_markets(n_vehicles=3)
    print(f"  customers : {instance.n_customers}")
    print(f"  couriers  : {instance.n_vehicles}")
    print(f"  capacity  : {instance.capacities[0]} stops/courier  (auto = ceil(n/K))")
    print(f"  metric    : {instance.metric} (km)")
    print()
    _run_brute("brute force (proven optimum)", instance)
    _run("nearest_neighbour", instance, nearest_neighbour)
    _run("clarke_wright_savings", instance, clarke_wright_savings)
    _run_or_tools("or_tools_solve", instance, time_limit_s=2)


def random_demo() -> None:
    print("=" * 72)
    print("Random Euclidean m-VRP   (n=20 customers, K=4 vehicles)")
    print("=" * 72)
    instance = VRPInstance.random(n_customers=20, n_vehicles=4, seed=42)
    print(f"  capacity (auto): {instance.capacities[0]} stops/vehicle")
    print()
    _run("nearest_neighbour", instance, nearest_neighbour)
    _run("clarke_wright_savings", instance, clarke_wright_savings)
    _run_or_tools("or_tools_solve", instance, time_limit_s=3)


def main() -> None:
    strasbourg_demo()
    random_demo()


if __name__ == "__main__":
    main()
