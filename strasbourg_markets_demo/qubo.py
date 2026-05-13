# Copyright (c) 2026 QPerfect
# SPDX-License-Identifier: MIT
"""QUBO problem container and Ising / MIMIQ-Hamiltonian conversion.

A QUBO is the canonical intermediate representation for combinatorial
optimisation on quantum hardware:

    minimise   x^T Q x + offset    over    x ∈ {0, 1}^n

This module provides the `QUBO` dataclass plus two reductions:

* `QUBO.to_ising()` — returns `(h, J, c)` where the Ising Hamiltonian
  `H = Σ h_i Z_i + Σ_{i<j} J_ij Z_i Z_j + c` has the same argmin (up to the
  bijection `x_i = (1 - z_i) / 2`).
* `QUBO.to_hamiltonian()` — wraps the Ising form as a `mimiqcircuits.Hamiltonian`
  so it can drive QAOA via `circuit.push_suzukitrotter(...)` or
  `circuit.push_expval(...)`.

Higher-level encoders (`tsp.to_qubo`, `vrp.to_qubo`) build problem-specific
QUBOs and attach a `labels` tuple so consumers can decode measurement
bitstrings back into tours / routes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


def bitstring_to_array(bs: Any, n: int) -> np.ndarray:
    """Convert a MIMIQ `BitString`, a "01..." str, or an iterable of 0/1 to
    a length-`n` int ndarray.
    """
    if hasattr(bs, "to01"):
        bs = bs.to01()
    if isinstance(bs, str):
        return np.array([int(c) for c in bs[:n]], dtype=int)
    arr = np.asarray(bs, dtype=int).ravel()
    return arr[:n] if arr.size >= n else np.pad(arr, (0, n - arr.size))


@dataclass(frozen=True)
class QUBO:
    """Container for a QUBO problem and conversions to Ising / MIMIQ-Hamiltonian.

    `Q` is treated as SYMMETRIC. The form `x^T Q x` counts each off-diagonal
    pair twice, so a coefficient `c · x_i · x_j` (`i ≠ j`) is split as
    `Q[i, j] += c/2` and `Q[j, i] += c/2`. Diagonal `Q[i, i]` is the linear
    coefficient for `x_i` (since `x_i² = x_i`).

    `offset` is a constant — irrelevant to the argmin, relevant to
    absolute energy.

    `labels` is opaque to this module; encoders attach per-qubit metadata
    (e.g. `(city, position)` for TSP) for the decoders to consume.
    """

    Q: np.ndarray
    offset: float = 0.0
    labels: Optional[tuple] = None

    @property
    def n_qubits(self) -> int:
        return int(self.Q.shape[0])

    def evaluate(self, bitstring: Any) -> float:
        """Energy of `bitstring` under this QUBO, including `offset`."""
        x = bitstring_to_array(bitstring, self.n_qubits)
        return float(x @ self.Q @ x + self.offset)

    def to_ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return `(h, J, c)` for the Ising form `Σ h_i Z_i + Σ J_ij Z_i Z_j + c`.

        Mapping: `x_i = (1 - z_i) / 2`, so `z_i ∈ {-1, +1}`, with `z_i = +1`
        encoding `x_i = 0`. `J` is upper-triangular (`J[i, j]` for `i < j`).
        The returned `c` includes both the conversion constant and `self.offset`.
        """
        Q = (self.Q + self.Q.T) / 2.0
        n = Q.shape[0]
        h = -0.5 * Q.sum(axis=1)
        J = 0.5 * np.triu(Q, k=1)
        c = (Q.sum() + np.trace(Q)) / 4.0
        return h, J, float(c) + self.offset

    def to_hamiltonian(self, *, tol: float = 1e-12):
        """Build a `mimiqcircuits.Hamiltonian` from the Ising form.

        Carries the Z and ZZ terms; the constant `c` from `to_ising()` is
        dropped (irrelevant for QAOA dynamics). Coefficients with
        magnitude below `tol` are skipped.
        """
        from mimiqcircuits import Hamiltonian, PauliString

        h, J, _c = self.to_ising()
        H = Hamiltonian()
        n = self.n_qubits
        for i in range(n):
            if abs(h[i]) > tol:
                H.push(float(h[i]), PauliString("Z"), i)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > tol:
                    H.push(float(J[i, j]), PauliString("ZZ"), i, j)
        return H

    def add_linear(self, idx: int, coeff: float) -> None:
        """In-place: add `coeff · x_idx` to the QUBO."""
        self.Q[idx, idx] += coeff

    def add_quadratic(self, i: int, j: int, coeff: float) -> None:
        """In-place: add `coeff · x_i · x_j` (`i ≠ j`) to the QUBO."""
        if i == j:
            self.Q[i, i] += coeff
            return
        half = 0.5 * coeff
        self.Q[i, j] += half
        self.Q[j, i] += half


def cvar_from_histogram(
    qubo: "QUBO",
    histogram: dict,
    *,
    alpha: float = 0.2,
) -> float:
    """Conditional value-at-risk of the QUBO energy.

    Walks `histogram = {bitstring: count}`, evaluates `qubo` on each
    bitstring, and returns the count-weighted mean of the energies in
    the bottom-`alpha` fraction of samples.

    CVaR-QAOA objective (Barkoutsos et al., Quantum 2020).
    """
    items: list[tuple[float, int]] = []
    for bs, count in histogram.items():
        x = bitstring_to_array(bs, qubo.n_qubits)
        items.append((float(qubo.evaluate(x)), int(count)))
    items.sort(key=lambda kv: kv[0])
    n_total = sum(c for _, c in items)
    target = max(1, int(alpha * n_total))
    accum = 0
    weighted = 0.0
    for e, c in items:
        take = min(c, target - accum)
        weighted += e * take
        accum += take
        if accum >= target:
            break
    return weighted / accum


def plot_q_matrix(
    qubo: "QUBO",
    ax=None,
    *,
    title: str | None = None,
    block_size: int | None = None,
    figsize: tuple[float, float] = (7, 6),
):
    """Plot the QUBO `Q` matrix as a centered diverging heatmap.

    Pass `block_size` to draw faint white guides between blocks of that
    width — useful for one-hot encodings where the rows/columns of the
    Q matrix have a natural per-class block structure (e.g. Lucas-2014
    TSP with `block_size = n_cities`).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    M = qubo.Q
    vmax = float(np.abs(M).max()) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="QUBO coefficient")
    if title:
        ax.set_title(title)
    ax.set_xlabel("qubit")
    ax.set_ylabel("qubit")
    if block_size:
        for k in range(1, qubo.n_qubits // block_size):
            ax.axhline(k * block_size - 0.5, color="white", lw=0.5, alpha=0.5)
            ax.axvline(k * block_size - 0.5, color="white", lw=0.5, alpha=0.5)
    return ax


__all__ = ["QUBO", "bitstring_to_array",
           "cvar_from_histogram", "plot_q_matrix"]
