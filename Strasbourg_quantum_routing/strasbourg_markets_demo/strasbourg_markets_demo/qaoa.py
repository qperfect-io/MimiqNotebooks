# Copyright © QPerfect
# SPDX-License-Identifier: MIT
"""QAOA ansatz builders for MIMIQ.

Three utilities used by the demo notebooks:

* `linear_ramp(p, ...)` — adiabatic-style warm-start angles
  $(\\beta=\\beta_\\text{max},\\, \\gamma=0)$ at the first layer,
  $(\\beta=0,\\, \\gamma=\\gamma_\\text{max})$ at the last.
* `build_qaoa(h, J, gammas, betas, ...)` — concrete-angle QAOA circuit
  from Ising parameters. Used in client-side optimisation loops.
* `parametric_qaoa(qubo, p, ...)` — symbolic-angle QAOA with the cost
  Hamiltonian's expectation value already pushed into a z-register,
  ready for `mc.OptimizationExperiment` and server-side optimisation
  via `conn.optimize(...)`.

For client-side optimisation: build numerically with `build_qaoa`,
sample with `conn.submit`. For server-side: build symbolically with
`parametric_qaoa`, submit via `OptimizationExperiment` + `conn.optimize`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Sequence

import numpy as np


def linear_ramp(
    p: int,
    gamma_max: float = math.pi,
    beta_max: float = math.pi / 2,
) -> tuple[list[float], list[float]]:
    """Adiabatic-style linear ramp.

    For `p > 1`: `gammas[k] = (k / (p - 1)) * gamma_max` and
    `betas[k] = (1 - k / (p - 1)) * beta_max`. So the first layer is
    pure mixer (γ=0, β=β_max) and the last layer is pure cost
    (γ=γ_max, β=0). For `p = 1`, both default to half their maxima.
    """
    if p == 1:
        return [0.5 * gamma_max], [0.5 * beta_max]
    s = np.linspace(0.0, 1.0, p)
    return list(s * gamma_max), list((1.0 - s) * beta_max)


def _normalise(h: np.ndarray, J: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Divide (h, J) by max|coupling|. Returns (h_n, J_n, scale)."""
    scale = float(max(np.abs(h).max(initial=0.0),
                      np.abs(J).max(initial=0.0),
                      1e-12))
    return h / scale, J / scale, scale


def build_qaoa(
    h: np.ndarray,
    J: np.ndarray,
    gammas: Sequence[float],
    betas: Sequence[float],
    *,
    normalise: bool = False,
):
    """Build a QAOA circuit with concrete-valued angles.

    `h` is the Ising linear vector (length `n`). `J` is the upper-
    triangular quadratic matrix. `gammas` and `betas` have length `p`,
    in layer order.

    With `normalise=True`, divides `(h, J)` by `max|coupling|` so the
    cost-layer rotations stay O(1). The optimal bitstring is unchanged,
    but optimal angles shift with the coupling scale.
    """
    import mimiqcircuits as mc

    n = len(h)
    if normalise:
        h, J, _ = _normalise(np.asarray(h), np.asarray(J))
    c = mc.Circuit()
    for q in range(n):
        c.push(mc.GateH(), q)
    for gamma, beta in zip(gammas, betas):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                c.push(mc.GateRZ(2.0 * gamma * h[i]), i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    c.push(mc.GateRZZ(2.0 * gamma * J[i, j]), i, j)
        for i in range(n):
            c.push(mc.GateRX(2.0 * beta), i)
    return c


def parametric_qaoa(
    qubo,
    p: int,
    *,
    gamma_max: float = math.pi,
    beta_max: float = math.pi / 2,
    gamma_prefix: str = "g",
    beta_prefix: str = "b",
    normalise: bool = True,
    with_expval: bool = True,
):
    """Build a parametric QAOA ansatz from a `QUBO`.

    Angles are `symengine.Symbol`s named `g0…g{p-1}` and `b0…b{p-1}`
    (prefixes customisable). With `with_expval=True`, the cost
    Hamiltonian's expectation value is appended via `push_expval` into
    z-register 0 — the shape `mc.OptimizationExperiment(..., zregister=0)`
    expects.

    Returns `(circuit, initial_params)`. `initial_params` is an
    `OrderedDict` keyed by the `Symbol` objects (not strings),
    pre-populated with linear-ramp values.

    With `normalise=True` (default), `(h, J)` are scaled to
    `max|coupling| = 1` before being baked into gate angles.
    """
    import mimiqcircuits as mc
    from symengine import Symbol

    n = qubo.n_qubits
    h, J, _c = qubo.to_ising()
    if normalise:
        h, J, _ = _normalise(h, J)

    gamma_syms = [Symbol(f"{gamma_prefix}{k}") for k in range(p)]
    beta_syms  = [Symbol(f"{beta_prefix}{k}")  for k in range(p)]

    circuit = mc.Circuit()
    for q in range(n):
        circuit.push(mc.GateH(), q)
    for g, b in zip(gamma_syms, beta_syms):
        for i in range(n):
            if abs(h[i]) > 1e-12:
                circuit.push(mc.GateRZ(2 * g * float(h[i])), i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-12:
                    circuit.push(mc.GateRZZ(2 * g * float(J[i, j])), i, j)
        for i in range(n):
            circuit.push(mc.GateRX(2 * b), i)

    if with_expval:
        H_C = qubo.to_hamiltonian()
        circuit.push_expval(H_C, *range(n))

    g_init, b_init = linear_ramp(p, gamma_max=gamma_max, beta_max=beta_max)
    init: OrderedDict = OrderedDict()
    for sym, val in zip(gamma_syms, g_init):
        init[sym] = val
    for sym, val in zip(beta_syms, b_init):
        init[sym] = val
    return circuit, init


__all__ = ["linear_ramp", "build_qaoa", "parametric_qaoa"]
