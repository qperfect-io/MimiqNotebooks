# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""Travelling Salesman Problem primitives for the Strasbourg-markets demo.

Provides a `TSPInstance` dataclass (distance matrix + optional 2D coords +
optional names) plus four classical solvers, all returning the canonical
`(tour, length)` pair:

- `brute_force` — exhaustive enumeration of `(n-1)!/2` distinct tours.
- `held_karp` — Bellman/Held–Karp dynamic programming, `O(n^2 2^n)`.
- `two_opt` — local search with the 2-opt neighbourhood.
- `simulated_annealing` — SA over 2-opt / Or-opt perturbations.

The exact and SA solvers delegate to `python-tsp`; we wrap them so the API
matches and so all returned tours are canonicalised (start at city 0, close
the cycle implicitly: a tour `[0, 2, 1, 3]` denotes the cycle
0 → 2 → 1 → 3 → 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Optional

import numpy as np


# Re-exported for callers that imported `STRASBOURG_MARKETS` directly;
# canonical data lives in `strasbourg.py`.
from . import strasbourg as _s

STRASBOURG_MARKETS: list[tuple[str, float, float]] = [
    (p.name, p.lat, p.lon) for p in _s.tsp_five()
]


# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------
from geopy.distance import geodesic as _geodesic


def _pairwise_geodesic_m(coords: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance matrix in metres for `(N, 2)` lat/lon."""
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _geodesic(tuple(coords[i]), tuple(coords[j])).meters
    return D


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def _canonicalise(tour: np.ndarray) -> np.ndarray:
    """Rotate so city 0 is first, then orient deterministically (forward vs
    reversed) — the two cycle representations have equal cost in symmetric TSP."""
    tour = np.asarray(tour, dtype=int).ravel()
    if 0 not in tour:
        return tour
    k = int(np.argmin(tour == 0).flat[0]) if tour[0] != 0 else 0
    k0 = int(np.where(tour == 0)[0][0])
    rolled = np.roll(tour, -k0)
    if len(rolled) > 2 and rolled[1] > rolled[-1]:
        rolled = np.concatenate([[rolled[0]], rolled[:0:-1]])
    return rolled


# -----------------------------------------------------------------------------
# TSP instance
# -----------------------------------------------------------------------------
@dataclass
class TSPInstance:
    """A symmetric TSP instance on `n` nodes.

    `distances[i, j]` is the cost from city `i` to city `j`. `coords` and
    `names` are optional — they only matter for plotting and pretty-printing.
    Tours are 1-D integer arrays of length `n` listing each city exactly once;
    the closing edge to the first city is implicit.
    """

    n: int
    distances: np.ndarray
    coords: Optional[np.ndarray] = None
    names: Optional[list[str]] = None

    def __post_init__(self) -> None:
        if self.distances.shape != (self.n, self.n):
            raise ValueError(
                f"distances must be ({self.n},{self.n}); got {self.distances.shape}"
            )
        if self.coords is not None and self.coords.shape != (self.n, 2):
            raise ValueError(
                f"coords must be ({self.n}, 2); got {self.coords.shape}"
            )
        if self.names is not None and len(self.names) != self.n:
            raise ValueError(
                f"names must have length {self.n}; got {len(self.names)}"
            )

    # --- factories -------------------------------------------------------
    @classmethod
    def from_coords(
        cls,
        coords: np.ndarray | list[tuple[float, float]],
        names: Optional[list[str]] = None,
        metric: str = "euclidean",
    ) -> "TSPInstance":
        """Build an instance from 2D coordinates, with chosen distance metric.

        `metric` is one of `"euclidean"` (default) or `"haversine"`. For the
        haversine metric, `coords` must be `(lat, lon)` pairs in degrees and
        the resulting distance matrix is in metres.
        """
        coords = np.asarray(coords, dtype=float)
        if metric == "euclidean":
            D = _pairwise_euclidean(coords)
        elif metric == "haversine":
            D = _pairwise_geodesic_m(coords)
        else:
            raise ValueError(f"unknown metric {metric!r}")
        return cls(n=coords.shape[0], distances=D, coords=coords, names=names)

    @classmethod
    def from_networkx(cls, G, weight: str = "weight") -> "TSPInstance":
        """Build from a complete `networkx` graph; missing edges become +inf.

        Node identifiers are mapped to dense 0..n-1 indices in the order that
        `G.nodes()` yields them. The names list captures the original IDs.
        """
        import networkx as nx

        nodes = list(G.nodes())
        n = len(nodes)
        idx = {nd: i for i, nd in enumerate(nodes)}
        D = np.full((n, n), np.inf)
        np.fill_diagonal(D, 0.0)
        for u, v, data in G.edges(data=True):
            w = float(data.get(weight, 1.0))
            i, j = idx[u], idx[v]
            D[i, j] = D[j, i] = w
        coords = None
        if all("pos" in G.nodes[n] for n in nodes):
            coords = np.array([G.nodes[n]["pos"] for n in nodes], dtype=float)
        return cls(n=n, distances=D, coords=coords, names=[str(n) for n in nodes])

    @classmethod
    def random(
        cls,
        n: int,
        seed: Optional[int] = None,
        metric: str = "euclidean",
        coord_range: tuple[float, float] = (0.0, 1.0),
    ) -> "TSPInstance":
        """Uniform random Euclidean instance on `[coord_range]^2`."""
        rng = np.random.default_rng(seed)
        coords = rng.uniform(coord_range[0], coord_range[1], size=(n, 2))
        return cls.from_coords(coords, metric=metric)

    @classmethod
    def strasbourg_markets(cls) -> "TSPInstance":
        """The five Strasbourg Christmas-market sites with great-circle distances.

        Distance unit is metres. Index 0 is Place Kléber. Coordinates come
        from `strasbourg.tsp_five()`.
        """
        places = _s.tsp_five()
        names = [p.name for p in places]
        coords = _s.coords_array(places)
        return cls.from_coords(coords, names=names, metric="haversine")

    # --- evaluation ------------------------------------------------------
    def tour_length(self, tour: np.ndarray | list[int]) -> float:
        """Sum of the n edges of a closed tour."""
        tour = np.asarray(tour, dtype=int).ravel()
        if tour.shape != (self.n,):
            raise ValueError(f"tour must have length {self.n}; got {tour.shape}")
        return float(self.distances[tour, np.roll(tour, -1)].sum())

    def is_valid_tour(self, tour: np.ndarray | list[int]) -> bool:
        """True iff the tour visits each city in 0..n-1 exactly once."""
        tour = np.asarray(tour, dtype=int).ravel()
        if tour.shape != (self.n,):
            return False
        return sorted(tour.tolist()) == list(range(self.n))

    # --- visualisation ---------------------------------------------------
    def plot(self, tour: Optional[np.ndarray | list[int]] = None, ax=None):
        """Scatter the cities and optionally draw a tour. Returns the Axes."""
        import matplotlib.pyplot as plt

        if self.coords is None:
            raise ValueError("plot requires coords to be set")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        # Convention: coords are (lat, lon); plot lon on x.
        x = self.coords[:, 1]
        y = self.coords[:, 0]
        ax.scatter(x, y, c="tab:blue", s=80, zorder=3)
        if self.names is not None:
            for i, nm in enumerate(self.names):
                ax.annotate(nm, (x[i], y[i]), xytext=(5, 5),
                            textcoords="offset points", fontsize=9)
        if tour is not None:
            tour = np.asarray(tour, dtype=int).ravel()
            cyc = np.append(tour, tour[0])
            ax.plot(x[cyc], y[cyc], "-", color="tab:red", linewidth=1.6, zorder=2)
        return ax


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------
def brute_force(instance: TSPInstance) -> tuple[np.ndarray, float]:
    """Exhaustive `(n-1)!` enumeration. Use only for `n <= 10` or so."""
    if instance.n <= 1:
        return np.arange(instance.n), 0.0
    D = instance.distances
    best_tour = None
    best_len = np.inf
    base = list(range(1, instance.n))
    for perm in permutations(base):
        tour = (0, *perm)
        length = sum(D[tour[i], tour[(i + 1) % instance.n]] for i in range(instance.n))
        if length < best_len:
            best_len = length
            best_tour = tour
    arr = _canonicalise(np.array(best_tour, dtype=int))
    return arr, float(best_len)


def held_karp(instance: TSPInstance) -> tuple[np.ndarray, float]:
    """Bellman / Held–Karp dynamic programming. `O(n^2 2^n)` time, `O(n 2^n)` memory.

    Practical up to `n ≈ 20–22` on a laptop.
    """
    n = instance.n
    D = instance.distances
    INF = float("inf")
    # dp[mask][i] = min cost path from 0 visiting cities in `mask` ending at i.
    dp: dict[int, dict[int, float]] = {1: {0: 0.0}}
    parent: dict[tuple[int, int], int] = {}
    for size in range(2, n + 1):
        for mask in _bitmasks_of_size(n, size, must_include_zero=True):
            dp[mask] = {}
            for last in range(n):
                if last == 0 or not (mask & (1 << last)):
                    continue
                prev_mask = mask ^ (1 << last)
                best = INF
                best_prev = -1
                for prev in dp.get(prev_mask, {}):
                    cost = dp[prev_mask][prev] + float(D[prev, last])
                    if cost < best:
                        best, best_prev = cost, prev
                if best_prev >= 0:
                    dp[mask][last] = best
                    parent[(mask, last)] = best_prev

    full = (1 << n) - 1
    end_costs = {i: dp[full][i] + float(D[i, 0]) for i in range(1, n) if i in dp[full]}
    best_end = min(end_costs, key=end_costs.get)
    best_cost = end_costs[best_end]

    tour: list[int] = []
    mask, cur = full, best_end
    while cur != 0:
        tour.append(cur)
        prev = parent[(mask, cur)]
        mask ^= 1 << cur
        cur = prev
    tour.append(0)
    tour.reverse()
    return _canonicalise(np.array(tour, dtype=int)), best_cost


def _bitmasks_of_size(n: int, size: int, must_include_zero: bool):
    """Yield bitmasks over `n` items with exactly `size` bits set; optionally
    require bit 0 to be set."""
    from itertools import combinations
    other = range(1, n) if must_include_zero else range(n)
    base = 1 if must_include_zero else 0
    pick = size - 1 if must_include_zero else size
    for combo in combinations(other, pick):
        mask = base
        for c in combo:
            mask |= 1 << c
        yield mask


def two_opt(
    instance: TSPInstance,
    initial_tour: Optional[np.ndarray | list[int]] = None,
) -> tuple[np.ndarray, float]:
    """2-opt local search: keep swapping non-adjacent edges while it improves."""
    n = instance.n
    D = instance.distances
    if initial_tour is None:
        tour = list(range(n))
    else:
        tour = list(np.asarray(initial_tour, dtype=int).ravel())

    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j - i == 1:
                    continue
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = (float(D[a, c]) + float(D[b, d])) - (float(D[a, b]) + float(D[c, d]))
                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1]
                    improved = True
    arr = _canonicalise(np.array(tour, dtype=int))
    return arr, instance.tour_length(arr)


def simulated_annealing(
    instance: TSPInstance,
    initial_tour: Optional[np.ndarray | list[int]] = None,
    *,
    n_iter: int = 20_000,
    alpha: float = 0.9995,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """Simulated annealing over the 2-opt neighbourhood.

    `alpha` is the geometric cooling rate (closer to 1 = slower cool, better
    quality). The temperature is initialised from a sample of move energy
    differences.
    """
    import random

    rng = random.Random(seed)
    n = instance.n
    D = instance.distances
    if initial_tour is None:
        tour = list(range(n))
    else:
        tour = list(np.asarray(initial_tour, dtype=int).ravel())

    def total(t: list[int]) -> float:
        return float(sum(D[t[i], t[(i + 1) % n]] for i in range(n)))

    cur_len = total(tour)
    best_tour = tour[:]
    best_len = cur_len
    T = max(1e-6, 0.5 * float(np.mean(D)))
    for _ in range(n_iter):
        i = rng.randint(1, n - 2)
        j = rng.randint(i + 1, n - 1)
        a, b = tour[i - 1], tour[i]
        c, d = tour[j], tour[(j + 1) % n]
        delta = (float(D[a, c]) + float(D[b, d])) - (float(D[a, b]) + float(D[c, d]))
        if delta < 0 or rng.random() < float(np.exp(-delta / max(T, 1e-12))):
            tour[i:j + 1] = tour[i:j + 1][::-1]
            cur_len += delta
            if cur_len < best_len:
                best_len = cur_len
                best_tour = tour[:]
        T *= alpha
    arr = _canonicalise(np.array(best_tour, dtype=int))
    return arr, instance.tour_length(arr)


# -----------------------------------------------------------------------------
# QAOA / Ising encoding — Lucas-2014 one-hot
# -----------------------------------------------------------------------------
def to_qubo(
    instance: TSPInstance,
    *,
    penalty: Optional[float] = None,
) -> "QUBO":
    """Encode `instance` as a QUBO using Lucas-2014 one-hot.

    Variables: `x_{i,t}` for city `i ∈ 0..n-1` and tour position `t ∈ 0..n-1`.
    Linear index: `q = i * n + t`. Total `n²` qubits.

    Cost: `Σ_t Σ_{i≠j} d(i, j) · x_{i,t} · x_{j, (t+1) mod n}` (closed tour).

    Constraints (added as squared penalties):
        each city visited exactly once: `Σ_t x_{i,t} = 1`
        each position used exactly once: `Σ_i x_{i,t} = 1`

    `penalty` defaults to `4 · max(d)` per Lucas-2014 §7.1; pass a custom
    value to tune feasibility-vs-cost balance for QAOA.

    Returns a `QUBO` whose `labels[q]` is `(city_name, position)` so a
    measurement bitstring can be decoded back via `tour_from_bitstring`.
    """
    from .qubo import QUBO

    n = instance.n
    if n == 0:
        raise ValueError("instance has no cities")
    D = instance.distances
    if penalty is None:
        penalty = 4.0 * float(D.max())
    n_qubits = n * n
    Q = np.zeros((n_qubits, n_qubits), dtype=float)
    offset = 0.0

    def q_idx(i: int, t: int) -> int:
        return i * n + t

    # Cost: closed-tour edges (cyclic position)
    for t in range(n):
        t_next = (t + 1) % n
        for i in range(n):
            for j in range(n):
                if i == j or D[i, j] == 0:
                    continue
                a, b = q_idx(i, t), q_idx(j, t_next)
                half = 0.5 * float(D[i, j])
                Q[a, b] += half
                Q[b, a] += half

    # Penalty: each city visited exactly once  (rows of the n×n grid)
    for i in range(n):
        offset += penalty
        for t in range(n):
            Q[q_idx(i, t), q_idx(i, t)] -= penalty
            for s in range(t + 1, n):
                Q[q_idx(i, t), q_idx(i, s)] += penalty
                Q[q_idx(i, s), q_idx(i, t)] += penalty

    # Penalty: each position used exactly once  (columns)
    for t in range(n):
        offset += penalty
        for i in range(n):
            Q[q_idx(i, t), q_idx(i, t)] -= penalty
            for j in range(i + 1, n):
                Q[q_idx(i, t), q_idx(j, t)] += penalty
                Q[q_idx(j, t), q_idx(i, t)] += penalty

    names = instance.names if instance.names is not None else [str(i) for i in range(n)]
    labels = tuple((names[i], t) for i in range(n) for t in range(n))
    return QUBO(Q=Q, offset=offset, labels=labels)


def tour_from_bitstring(
    qubo: "QUBO",
    bitstring,
    n_cities: int,
) -> Optional[np.ndarray]:
    """Decode a Lucas-2014 one-hot bitstring to a tour, or `None` if
    infeasible (some position has zero or more than one city assigned)."""
    from .qubo import bitstring_to_array

    n = n_cities
    x = bitstring_to_array(bitstring, n * n).reshape(n, n)
    tour = np.full(n, -1, dtype=int)
    for t in range(n):
        col = np.where(x[:, t] == 1)[0]
        if len(col) != 1:
            return None
        tour[t] = int(col[0])
    if len(set(tour.tolist())) != n:
        return None
    return _canonicalise(tour)


def tour_from_histogram(
    qubo: "QUBO",
    histogram: dict,
    n_cities: int,
) -> Optional[tuple[np.ndarray, int, float]]:
    """Walk `histogram` by descending count and return the first feasible
    `(tour, count, energy)` triple, or `None` if no bitstring decodes.
    """
    sorted_out = sorted(histogram.items(), key=lambda kv: -kv[1])
    for bs, count in sorted_out:
        decoded = tour_from_bitstring(qubo, bs, n_cities)
        if decoded is not None:
            return decoded, int(count), float(qubo.evaluate(bs))
    return None
