# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""Vehicle Routing Problem (VRP) primitives for the Strasbourg-markets demo.

`VRPInstance` defaults to **uncapacitated multi-vehicle TSP** (m-VRP) when
constructed with only `n_customers` and `n_vehicles` — every customer has
demand 1 and the per-vehicle capacity is set to `ceil(n_customers / K)` so
all `K` vehicles end up serving roughly equal shares. To get a CVRP, pass
`demands=` and `capacity=` explicitly.

Solvers:
- `m_vrp_brute_force` — exact, surjection-based; tractable up to ~10 customers.
- `nearest_neighbour` — round-robin greedy.
- `clarke_wright_savings` — Clarke & Wright 1964 savings heuristic.
- `or_tools_solve` — Google OR-tools `RoutingModel` with capacity dimension.

Index 0 of every distance / coordinate / demand array is the depot. Customers
are indexed 1..n_customers. Routes are returned as a list of per-vehicle
sequences of *customer* indices (no depot prefix/suffix); the depot is
implicit at both ends.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Tuples derived from `strasbourg.py`, kept for backwards-compatible imports.
from . import strasbourg as _s

_VRP_DEPOT, _VRP_CUSTOMERS = _s.vrp_default()
STRASBOURG_DEPOT: tuple[float, float] = (_VRP_DEPOT.lat, _VRP_DEPOT.lon)
STRASBOURG_MARKETS: list[tuple[str, float, float]] = [
    (p.name, p.lat, p.lon) for p in _VRP_CUSTOMERS
]
# Synthetic peak-hour visitor estimates (thousands) per customer. With the
# default capacity of 7 they force a multi-vehicle partition.
STRASBOURG_DEMANDS: list[float] = [3.0, 4.0, 2.0, 1.0, 2.0]


# -----------------------------------------------------------------------------
# Distance helpers
# -----------------------------------------------------------------------------
from geopy.distance import geodesic as _geodesic


def _pairwise_geodesic_km(coords: np.ndarray) -> np.ndarray:
    """Pairwise geodesic distance matrix in km for an (N, 2) lat/lon array."""
    n = coords.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = _geodesic(tuple(coords[i]), tuple(coords[j])).kilometers
    return D


def _pairwise_euclidean(coords: np.ndarray) -> np.ndarray:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# -----------------------------------------------------------------------------
# VRP instance
# -----------------------------------------------------------------------------
@dataclass(kw_only=True)
class VRPInstance:
    """A vehicle-routing instance — m-VRP by default, CVRP when demands+capacity given.

    Index 0 is always the depot. Customer i has demand `demands[i-1]` and
    coordinates `coords[i]` if available. Distances are symmetric; non-Euclidean
    matrices are accepted but feasibility checks assume the triangle inequality
    is at most mildly violated.

    Leaving `demands` and `capacity` as `None` produces an **m-VRP** instance:
    `demands = ones(n)` and `capacity = ceil(n / n_vehicles)`. With those
    values, capacity-aware solvers (Clarke-Wright, OR-tools) are forced to
    distribute customers across all `K` vehicles, recovering m-VRP semantics.
    """

    n_customers: int
    n_vehicles: int
    distances: np.ndarray
    capacity: Optional[float | list[float]] = None
    demands: Optional[np.ndarray] = None
    coords: Optional[np.ndarray] = None
    names: Optional[list[str]] = None
    time_windows: Optional[np.ndarray] = None
    service_times: Optional[np.ndarray] = None
    metric: str = "euclidean"  # "euclidean" | "haversine" | "custom"

    def __post_init__(self) -> None:
        n = self.n_customers
        K = self.n_vehicles
        if self.distances.shape != (n + 1, n + 1):
            raise ValueError(
                f"distances must be ({n + 1},{n + 1}); got {self.distances.shape}"
            )
        if self.demands is None:
            self.demands = np.ones(n, dtype=float)
        else:
            self.demands = np.asarray(self.demands, dtype=float)
            if self.demands.shape != (n,):
                raise ValueError(
                    f"demands must be shape ({n},); got {self.demands.shape}"
                )
        if self.capacity is None:
            self.capacity = float(math.ceil(n / K))
        if self.coords is not None and self.coords.shape != (n + 1, 2):
            raise ValueError(
                f"coords must be ({n + 1},2); got {self.coords.shape}"
            )
        if isinstance(self.capacity, (int, float)):
            self._capacities = [float(self.capacity)] * K
        else:
            if len(self.capacity) != K:
                raise ValueError("capacity list length must equal n_vehicles")
            self._capacities = [float(c) for c in self.capacity]

    @property
    def is_mvrp(self) -> bool:
        """True iff this instance was constructed with the m-VRP defaults."""
        return (
            np.allclose(self.demands, 1.0)
            and len(set(self._capacities)) == 1
            and self._capacities[0] == float(math.ceil(self.n_customers / self.n_vehicles))
        )

    # --- factories -------------------------------------------------------
    @classmethod
    def from_coords(
        cls,
        depot_coord: tuple[float, float],
        customer_coords: list[tuple[float, float]],
        n_vehicles: int,
        *,
        demands: Optional[list[float] | np.ndarray] = None,
        capacity: Optional[float | list[float]] = None,
        names: Optional[list[str]] = None,
        metric: str = "euclidean",
    ) -> "VRPInstance":
        """Construct from depot + customer coordinates. Defaults to m-VRP.

        Pass `demands` and `capacity` to construct a CVRP instead.
        """
        coords = np.array([depot_coord, *customer_coords], dtype=float)
        if metric == "haversine":
            D = _pairwise_geodesic_km(coords)
        elif metric == "euclidean":
            D = _pairwise_euclidean(coords)
        else:
            raise ValueError(f"unknown metric {metric!r}")
        return cls(
            n_customers=len(customer_coords),
            n_vehicles=n_vehicles,
            capacity=capacity,
            demands=np.asarray(demands, dtype=float) if demands is not None else None,
            distances=D,
            coords=coords,
            names=names,
            metric=metric,
        )

    @classmethod
    def random(
        cls,
        n_customers: int,
        n_vehicles: int,
        *,
        demands: Optional[list[float] | np.ndarray] = None,
        capacity: Optional[float | list[float]] = None,
        seed: Optional[int] = None,
        area: float = 100.0,
    ) -> "VRPInstance":
        """Uniform random instance on a square area. Defaults to m-VRP.

        Pass `demands` (or `demand_max=` via a wrapper) and `capacity` to
        construct a CVRP instead.
        """
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, area, size=(n_customers + 1, 2))
        coords[0] = np.array([area / 2, area / 2])
        D = _pairwise_euclidean(coords)
        return cls(
            n_customers=n_customers,
            n_vehicles=n_vehicles,
            demands=np.asarray(demands, dtype=float) if demands is not None else None,
            capacity=capacity,
            distances=D,
            coords=coords,
            metric="euclidean",
        )

    @classmethod
    def strasbourg_markets(
        cls,
        n_vehicles: int = 3,
        *,
        demands: Optional[list[float] | np.ndarray] = None,
        capacity: Optional[float | list[float]] = None,
    ) -> "VRPInstance":
        """The five-market m-VRP instance used by the demo (default `n_vehicles=3`).

        With defaults: uncapacitated multi-vehicle TSP from the Gare Centrale.
        Pass `demands` and `capacity` to construct a CVRP instead.
        """
        names_full = ["Gare Centrale"] + [m[0] for m in STRASBOURG_MARKETS]
        customer_coords = [(lat, lon) for _, lat, lon in STRASBOURG_MARKETS]
        return cls.from_coords(
            depot_coord=STRASBOURG_DEPOT,
            customer_coords=customer_coords,
            n_vehicles=n_vehicles,
            demands=demands,
            capacity=capacity,
            names=names_full,
            metric="haversine",
        )

    # --- properties ------------------------------------------------------
    @property
    def n_nodes(self) -> int:
        return self.n_customers + 1

    @property
    def capacities(self) -> list[float]:
        """Per-vehicle capacity list (homogeneous fleet broadcasts the scalar)."""
        return list(self._capacities)

    # --- evaluation ------------------------------------------------------
    def total_distance(self, routes: list[list[int]]) -> float:
        """Sum of route lengths, depot-anchored at both ends."""
        total = 0.0
        for route in routes:
            if not route:
                continue
            prev = 0
            for c in route:
                total += float(self.distances[prev, c])
                prev = c
            total += float(self.distances[prev, 0])
        return total

    def is_feasible(self, routes: list[list[int]]) -> tuple[bool, str]:
        """Check capacity and visit-once constraints. Returns (ok, reason)."""
        if len(routes) > self.n_vehicles:
            return False, f"too many routes ({len(routes)} > {self.n_vehicles})"
        seen: set[int] = set()
        for k, route in enumerate(routes):
            cap = self._capacities[k]
            load = 0.0
            for c in route:
                if c < 1 or c > self.n_customers:
                    return False, f"vehicle {k}: customer index {c} out of range"
                if c in seen:
                    return False, f"customer {c} visited more than once"
                seen.add(c)
                load += float(self.demands[c - 1])
            if load > cap + 1e-9:
                return False, f"vehicle {k}: load {load:.3f} > capacity {cap:.3f}"
        missing = set(range(1, self.n_customers + 1)) - seen
        if missing:
            return False, f"customers not visited: {sorted(missing)}"
        return True, "feasible"

    # --- visualisation ---------------------------------------------------
    def plot(self, routes: Optional[list[list[int]]] = None, ax=None):
        """Scatter the depot and customers, optionally drawing a set of routes.

        Returns the matplotlib Axes for further customisation.
        """
        import matplotlib.pyplot as plt

        if self.coords is None:
            raise ValueError("plot requires coords to be set")
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        depot = self.coords[0]
        cust = self.coords[1:]
        ax.scatter(cust[:, 1], cust[:, 0], c="tab:blue", s=60, zorder=3, label="customer")
        ax.scatter([depot[1]], [depot[0]], c="tab:red", s=120, marker="s", zorder=4, label="depot")
        if self.names is not None:
            for i, nm in enumerate(self.names):
                ax.annotate(nm, (self.coords[i, 1], self.coords[i, 0]),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
        if routes is not None:
            cmap = plt.get_cmap("tab10")
            for k, route in enumerate(routes):
                if not route:
                    continue
                pts = [0, *route, 0]
                xs = [self.coords[i, 1] for i in pts]
                ys = [self.coords[i, 0] for i in pts]
                ax.plot(xs, ys, "-", color=cmap(k % 10), label=f"vehicle {k}")
        ax.set_xlabel("lon" if self.metric == "haversine" else "x")
        ax.set_ylabel("lat" if self.metric == "haversine" else "y")
        from .strasbourg import place_legend_outside
        place_legend_outside(ax, fontsize=8)
        return ax


# -----------------------------------------------------------------------------
# Solvers
# -----------------------------------------------------------------------------
def nearest_neighbour(instance: VRPInstance) -> list[list[int]]:
    """Greedy nearest-neighbour with capacity feasibility.

    Iterates: pick the cheapest unvisited customer that fits in the current
    vehicle's residual capacity; if none fits, close the route and start the
    next vehicle. May return fewer routes than `n_vehicles`. Raises
    `RuntimeError` if the demand totals exceed total fleet capacity.
    """
    D = instance.distances
    demands = instance.demands
    caps = instance.capacities
    unvisited = set(range(1, instance.n_customers + 1))
    if sum(demands) > sum(caps) + 1e-9:
        raise RuntimeError("infeasible: total demand exceeds total fleet capacity")
    routes: list[list[int]] = []
    for k in range(instance.n_vehicles):
        if not unvisited:
            break
        load = 0.0
        cap = caps[k]
        route: list[int] = []
        cur = 0
        while True:
            candidates = [c for c in unvisited if load + demands[c - 1] <= cap + 1e-9]
            if not candidates:
                break
            nxt = min(candidates, key=lambda c: D[cur, c])
            route.append(nxt)
            load += demands[nxt - 1]
            unvisited.remove(nxt)
            cur = nxt
        if route:
            routes.append(route)
    if unvisited:
        raise RuntimeError(
            f"nearest-neighbour could not place customers {sorted(unvisited)}; "
            "increase n_vehicles or capacity"
        )
    return routes


def clarke_wright_savings(instance: VRPInstance) -> list[list[int]]:
    """Clarke & Wright (1964) savings heuristic, parallel variant.

    Starts with single-stop routes `[0, i, 0]`, then repeatedly merges the
    endpoint pair `(i, j)` with maximum saving `s_ij = d(0,i) + d(0,j)
    - d(i,j)` while capacity holds. If more than `n_vehicles` routes remain
    at the end, the smallest are bin-packed into the first `n_vehicles`.
    """
    n = instance.n_customers
    D = instance.distances
    demands = instance.demands
    cap = max(instance.capacities)  # homogeneous fleet assumption

    routes: list[list[int]] = [[i] for i in range(1, n + 1)]
    loads: list[float] = [float(demands[i - 1]) for i in range(1, n + 1)]
    route_of = {c: idx for idx, c in enumerate(range(1, n + 1))}

    savings = []
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            s = D[0, i] + D[0, j] - D[i, j]
            savings.append((s, i, j))
    savings.sort(reverse=True)

    for s, i, j in savings:
        if s <= 0:
            break
        ri = route_of[i]
        rj = route_of[j]
        if ri == rj:
            continue
        Ri = routes[ri]
        Rj = routes[rj]
        # Both i and j must sit on a route endpoint (adjacent to depot).
        if i == Ri[-1] and j == Rj[0]:
            new_route = Ri + Rj
        elif j == Rj[-1] and i == Ri[0]:
            new_route = Rj + Ri
        elif i == Ri[0] and j == Rj[0]:
            new_route = list(reversed(Ri)) + Rj
        elif i == Ri[-1] and j == Rj[-1]:
            new_route = Ri + list(reversed(Rj))
        else:
            continue
        if loads[ri] + loads[rj] > cap + 1e-9:
            continue
        # Reuse slot ri; clear rj.
        routes[ri] = new_route
        loads[ri] = loads[ri] + loads[rj]
        routes[rj] = []
        loads[rj] = 0.0
        for c in new_route:
            route_of[c] = ri

    out = [r for r in routes if r]
    out.sort(key=lambda r: -sum(demands[c - 1] for c in r))
    if len(out) > instance.n_vehicles:
        # First-fit bin-pack the surplus routes into the first n_vehicles.
        head = out[: instance.n_vehicles]
        tail = out[instance.n_vehicles :]
        head_loads = [sum(demands[c - 1] for c in r) for r in head]
        for r in tail:
            r_load = sum(demands[c - 1] for c in r)
            placed = False
            for k in range(len(head)):
                if head_loads[k] + r_load <= cap + 1e-9:
                    head[k] = head[k] + r
                    head_loads[k] += r_load
                    placed = True
                    break
            if not placed:
                head.append(r)  # caller will see infeasibility
        out = head
    return out


def m_vrp_brute_force(
    instance: VRPInstance,
    *,
    require_all_vehicles_used: bool = True,
    respect_capacity: bool = True,
    max_n: int = 10,
) -> tuple[list[list[int]], float]:
    """Exact m-VRP solver via exhaustive enumeration.

    Iterates over every assignment of customers to vehicles (`K ** n` total)
    and, for each cluster, brute-forces the optimal TSP visit order.

    By default both the "all vehicles used" constraint (natural m-VRP framing)
    and the per-vehicle capacity constraint (which defaults to `ceil(n/K)` on
    the instance) are enforced — making this give the same answer as
    `or_tools_solve` and `clarke_wright_savings` on the same instance.

    Pass `respect_capacity=False` for pure uncapacitated m-VRP, where any
    customer-to-vehicle partition is allowed. Pass `require_all_vehicles_used=False`
    to allow idle vehicles (the optimum then collapses to a single TSP).

    Tractable up to roughly `n_customers = 10`; raises `ValueError` above
    `max_n`. Returns `(routes, total_distance)` where each route is a list of
    customer indices in visit order (depot 0 implicit at both ends).
    """
    D = instance.distances
    n = instance.n_customers
    K = instance.n_vehicles
    demands = instance.demands
    capacities = instance._capacities

    if n > max_n:
        raise ValueError(
            f"brute-force m-VRP is intractable for n_customers={n} > {max_n}; "
            "use a heuristic or OR-tools."
        )
    if require_all_vehicles_used and K > n:
        raise ValueError(
            f"cannot require all {K} vehicles used with only {n} customers."
        )

    customers = list(range(1, n + 1))
    best_total = float("inf")
    best_routes: Optional[list[list[int]]] = None

    for assignment in itertools.product(range(K), repeat=n):
        if require_all_vehicles_used and len(set(assignment)) < K:
            continue
        clusters: list[list[int]] = [[] for _ in range(K)]
        for ci, vk in enumerate(assignment):
            clusters[vk].append(customers[ci])

        if respect_capacity:
            over_capacity = False
            for k_, cluster in enumerate(clusters):
                if sum(float(demands[c - 1]) for c in cluster) > capacities[k_] + 1e-9:
                    over_capacity = True
                    break
            if over_capacity:
                continue

        total = 0.0
        per_vehicle: list[list[int]] = []
        prune = False
        for cluster in clusters:
            if not cluster:
                per_vehicle.append([])
                continue
            best_tsp = float("inf")
            best_order: list[int] = []
            for perm in itertools.permutations(cluster):
                length = float(D[0, perm[0]]) + float(D[perm[-1], 0])
                length += sum(float(D[perm[i], perm[i + 1]])
                              for i in range(len(perm) - 1))
                if length < best_tsp:
                    best_tsp = length
                    best_order = list(perm)
            per_vehicle.append(best_order)
            total += best_tsp
            if total >= best_total:
                prune = True
                break

        if not prune and total < best_total:
            best_total = total
            best_routes = per_vehicle

    assert best_routes is not None, "no feasible m-VRP solution found"
    return best_routes, best_total


def or_tools_solve(
    instance: VRPInstance,
    time_limit_s: int = 5,
    first_solution: str = "PATH_CHEAPEST_ARC",
    metaheuristic: str = "GUIDED_LOCAL_SEARCH",
) -> tuple[list[list[int]], float]:
    """Google OR-tools CVRP solver (RoutingModel + capacity dimension).

    Returns `(routes, total_distance)`. The OR-tools model uses integer
    distances internally, so we scale the distance matrix by 1000 before
    handing it over and rescale the cost back at the end.
    """
    from ortools.constraint_solver import pywrapcp, routing_enums_pb2

    n_nodes = instance.n_nodes
    n_vehicles = instance.n_vehicles
    scale = 1000

    int_distances = np.rint(instance.distances * scale).astype(np.int64)
    int_demands = np.concatenate([[0], instance.demands]).astype(np.int64)
    int_capacities = [int(np.rint(c)) for c in instance.capacities]

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_cb(from_index: int, to_index: int) -> int:
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        return int(int_distances[i, j])

    transit_idx = routing.RegisterTransitCallback(distance_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    def demand_cb(from_index: int) -> int:
        i = manager.IndexToNode(from_index)
        return int(int_demands[i])

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx,
        0,                       # null capacity slack
        int_capacities,          # vehicle maximum capacities
        True,                    # start cumul at zero
        "Capacity",
    )

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(
        routing_enums_pb2.FirstSolutionStrategy, first_solution
    )
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic
    )
    params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-tools failed to find a solution")

    routes: list[list[int]] = []
    for k in range(n_vehicles):
        idx = routing.Start(k)
        route: list[int] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if node != 0:
                route.append(node)
            idx = solution.Value(routing.NextVar(idx))
        routes.append(route)

    total = instance.total_distance(routes)
    return routes, total


# -----------------------------------------------------------------------------
# QAOA / Ising encoding — full one-hot (m-VRP / VRP)
# -----------------------------------------------------------------------------
def to_qubo(
    instance: VRPInstance,
    *,
    penalty: Optional[float] = None,
    n_positions: Optional[int] = None,
) -> "QUBO":
    """Encode `instance` as a QUBO with the full one-hot formulation.

    Variables: `x_{i, t, k}` — customer `i ∈ 0..n-1`, position `t ∈ 0..L-1`,
    vehicle `k ∈ 0..K-1`. Linear index: `q = i · L · K + t · K + k`. Total
    `n · L · K` qubits.

    `n_positions` (`L`) defaults to `instance.capacity` (rounded up); for
    the m-VRP defaults this is `ceil(n / K)` — the minimum that still
    allows every customer to be placed.

    Cost — for each vehicle k:
        depot → pos 0:                   `d(0, i) · x_{i, 0, k}`
        pos t → pos t+1   (t < L-1):     `d(i, j) · x_{i, t, k} · x_{j, t+1, k}`
        early return to depot:           `d(i, 0) · x_{i, t, k} · (1 - Σ_j x_{j, t+1, k})`
        pos L-1 → depot:                 `d(i, 0) · x_{i, L-1, k}`

    The "early return" term keeps the encoding cost-exact when a vehicle
    fills fewer than `L` positions — a depot return is added at the actual
    last filled position rather than always at `L-1`.

    Constraints (squared penalties):
        each customer in exactly one slot: `Σ_{t,k} x_{i, t, k} = 1`
        each slot at most one customer:    `Σ_i x_{i, t, k} ≤ 1`

    Returns a `QUBO` whose `labels[q]` is `(customer_name, position, vehicle)`.
    """
    from .qubo import QUBO

    n = instance.n_customers
    K = instance.n_vehicles
    if n == 0 or K == 0:
        raise ValueError("instance has no customers or no vehicles")

    if n_positions is None:
        L = max(1, int(round(float(instance._capacities[0]))))
    else:
        L = int(n_positions)

    D = instance.distances  # (n+1, n+1), depot at index 0
    if penalty is None:
        # Must dominate any single edge cost; override if QAOA leaks feasibility.
        penalty = 2.0 * float(D.max())

    n_qubits = n * L * K
    Q = np.zeros((n_qubits, n_qubits), dtype=float)
    offset = 0.0

    def q_idx(i: int, t: int, k: int) -> int:
        return i * L * K + t * K + k

    def add_quad(a: int, b: int, c: float) -> None:
        if a == b:
            Q[a, a] += c
        else:
            Q[a, b] += 0.5 * c
            Q[b, a] += 0.5 * c

    # Cost — depot-anchored route per vehicle, with early-return correction.
    for k in range(K):
        for i in range(n):
            d0i = float(D[0, i + 1])
            di0 = float(D[i + 1, 0])
            # depot → position 0
            Q[q_idx(i, 0, k), q_idx(i, 0, k)] += d0i
            # position L-1 → depot
            Q[q_idx(i, L - 1, k), q_idx(i, L - 1, k)] += di0
        # Internal transitions + early-return correction
        for t in range(L - 1):
            for i in range(n):
                di0 = float(D[i + 1, 0])
                Q[q_idx(i, t, k), q_idx(i, t, k)] += di0  # linear part of d(i,0)·x·(1-Σ)
                for j in range(n):
                    coeff = float(D[i + 1, j + 1]) - di0  # internal d(i,j) - cancellation -d(i,0)
                    if coeff == 0.0:
                        continue
                    add_quad(q_idx(i, t, k), q_idx(j, t + 1, k), coeff)

    # Penalty: each customer in exactly one (t, k)
    for i in range(n):
        offset += penalty
        slots = [(t, k) for t in range(L) for k in range(K)]
        for s_idx, (t, k) in enumerate(slots):
            qi = q_idx(i, t, k)
            Q[qi, qi] -= penalty
            for (t2, k2) in slots[s_idx + 1:]:
                qj = q_idx(i, t2, k2)
                Q[qi, qj] += penalty
                Q[qj, qi] += penalty

    # Penalty: each slot has at most one customer (column at-most-one)
    for k in range(K):
        for t in range(L):
            for i in range(n):
                for j in range(i + 1, n):
                    qi = q_idx(i, t, k)
                    qj = q_idx(j, t, k)
                    Q[qi, qj] += penalty
                    Q[qj, qi] += penalty

    names = (instance.names[1:] if instance.names is not None
             else [str(i + 1) for i in range(n)])
    labels = tuple(
        (names[i], t, k) for i in range(n) for t in range(L) for k in range(K)
    )
    return QUBO(Q=Q, offset=offset, labels=labels)


def routes_from_histogram(
    qubo: "QUBO",
    histogram: dict,
    n_customers: int,
    n_vehicles: int,
    n_positions: Optional[int] = None,
) -> Optional[tuple[list[list[int]], int, float]]:
    """Walk `histogram` by descending count and return the first feasible
    `(routes, count, energy)` triple, or `None` if no bitstring decodes.
    """
    sorted_out = sorted(histogram.items(), key=lambda kv: -kv[1])
    for bs, count in sorted_out:
        decoded = routes_from_bitstring(
            qubo, bs, n_customers, n_vehicles, n_positions=n_positions
        )
        if decoded is not None:
            return decoded, int(count), float(qubo.evaluate(bs))
    return None


def routes_from_bitstring(
    qubo: "QUBO",
    bitstring,
    n_customers: int,
    n_vehicles: int,
    n_positions: Optional[int] = None,
) -> Optional[list[list[int]]]:
    """Decode a full-one-hot VRP bitstring into per-vehicle routes (lists
    of customer indices in visit order, depot implicit). Returns `None` if
    any slot has > 1 customer, or any customer is visited 0 or > 1 times."""
    from .qubo import bitstring_to_array

    n = n_customers
    K = n_vehicles
    if n_positions is None:
        L = qubo.n_qubits // (n * K)
    else:
        L = n_positions
    if n * L * K != qubo.n_qubits:
        raise ValueError(
            f"bitstring shape {qubo.n_qubits} doesn't match n={n}, L={L}, K={K}"
        )

    x = bitstring_to_array(bitstring, n * L * K).reshape(n, L, K)
    visits = x.sum(axis=(1, 2))  # times each customer appears
    if not np.all(visits == 1):
        return None
    slot_counts = x.sum(axis=0)  # (L, K) – customers in each slot
    if np.any(slot_counts > 1):
        return None

    routes: list[list[int]] = [[] for _ in range(K)]
    for k in range(K):
        for t in range(L):
            present = np.where(x[:, t, k] == 1)[0]
            if len(present) == 1:
                routes[k].append(int(present[0]) + 1)  # +1: depot is index 0
    return routes
