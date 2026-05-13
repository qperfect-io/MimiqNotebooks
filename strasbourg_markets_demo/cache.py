# Copyright (c) 2026 QPerfect
# SPDX-License-Identifier: MIT
"""On-disk cache for MIMIQ executions, using MIMIQ's native protobuf format.

Wraps `conn.submit` / `conn.optimize` / `conn.get_result` so a
`with mimiq_cache(conn, key=...)` block looks like ordinary MIMIQ usage,
but on subsequent runs the cloud round-trip is skipped and the previous
result is loaded from disk.

Cache files use MIMIQ's own protobuf serialisation (`saveproto` /
`loadproto`). Two file suffixes are used:

- `<key>__<hash>.pb`     — `QCSResults` from `conn.submit` + `get_result`
- `<key>__<hash>.optpb`  — `OptimizationResults` from `conn.optimize` + `get_result`

Usage:

    from strasbourg_markets_demo.cache import mimiq_cache

    conn = mc.MimiqConnection(mc.QPERFECT_CLOUD)
    conn.connect()

    # Standard sample-based job
    with mimiq_cache(conn, key="qaoa-strasbourg-100q"):
        job = conn.submit(circuit, algorithm="mps", bonddim=256, nsamples=4096)
        res = conn.get_result(job)

    # Server-side optimisation
    with mimiq_cache(conn, key="qaoa-server-toy"):
        job = conn.optimize(exp, algorithm="mps", bonddim=64, history=True)
        res = conn.get_result(job)

The cache key is `f"{key}__{hash(inputs)}"`, so changing one parameter
only invalidates the corresponding entry. Files live in `.nb_cache/mimiq/`
by default.
"""

from __future__ import annotations

import hashlib
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator

DEFAULT_CACHE_DIR: Path = Path(".nb_cache") / "mimiq"
QCS_SUFFIX: str = ".pb"
OPT_SUFFIX: str = ".optpb"
ALL_SUFFIXES: tuple[str, ...] = (QCS_SUFFIX, OPT_SUFFIX, ".pkl")  # .pkl = legacy


class _CachedJob:
    """Sentinel returned by patched `submit`/`optimize` when the result is on disk."""
    __slots__ = ("key", "result")

    def __init__(self, key: str, result: Any) -> None:
        self.key = key
        self.result = result


def _hash_circuit(circuit) -> str:
    """16-char hex digest of a `Circuit`'s gate sequence (op type, parameters,
    target qubits, classical bits where applicable)."""
    h = hashlib.sha256()
    for inst in circuit:
        op = inst.operation
        h.update(type(op).__name__.encode())
        for attr in ("theta", "phi", "lambda", "lam", "value", "params"):
            if hasattr(op, attr):
                h.update(f"{attr}={getattr(op, attr)!r}".encode())
        h.update(repr(tuple(inst.qubits)).encode())
        if hasattr(inst, "bits"):
            h.update(repr(tuple(inst.bits)).encode())
    return h.hexdigest()[:16]


def _hash_inputs(circuit, args: tuple, kwargs: dict) -> str:
    """Stable 16-char hex hash of a Circuit + submit args."""
    h = hashlib.sha256()
    h.update(_hash_circuit(circuit).encode())
    for a in args:
        h.update(repr(a).encode())
    for k in sorted(kwargs):
        h.update(f"{k}={kwargs[k]!r}".encode())
    return h.hexdigest()[:16]


def _hash_experiments(experiments, args: tuple, kwargs: dict) -> str:
    """Stable 16-char hex hash for `conn.optimize(experiments, ...)`.

    `experiments` may be a single `OptimizationExperiment` or an iterable
    of them. Each contributes its symbolic circuit hash + optimiser /
    maxiters / zregister / initparam values.
    """
    h = hashlib.sha256()
    if not isinstance(experiments, (list, tuple)):
        experiments = [experiments]
    for exp in experiments:
        h.update(_hash_circuit(exp.circuit).encode())
        h.update(f"opt={getattr(exp, 'optimizer', None)!r}".encode())
        h.update(f"max={getattr(exp, 'maxiters', None)!r}".encode())
        h.update(f"zr={getattr(exp, 'zregister', None)!r}".encode())
        for sym, val in sorted(exp.initparams.items(), key=lambda kv: str(kv[0])):
            h.update(f"{sym}={val}".encode())
    for a in args:
        h.update(repr(a).encode())
    for k in sorted(kwargs):
        h.update(f"{k}={kwargs[k]!r}".encode())
    return h.hexdigest()[:16]


@contextmanager
def mimiq_cache(
    conn,
    key: str,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Iterator[None]:
    """Context manager that caches `conn.submit` / `conn.optimize` results on disk.

    Inside the `with` block:
      - `conn.submit(circuit, ...)` is hashed (circuit + kwargs) into a key
        and looked up under `<key>__<hash>.pb`. On a hit, no cloud call;
        the subsequent `conn.get_result(job)` returns a `QCSResults`
        loaded from disk via `loadproto`. On a miss, the real cloud
        round-trip happens and the result is written via `saveproto`.
      - `conn.optimize(experiments, ...)` is hashed (each experiment's
        circuit + initparams + optimiser / maxiters / zregister + kwargs)
        and looked up under `<key>__<hash>.optpb`. Same hit/miss logic
        but the result is an `OptimizationResults`.

    Pass `force_refresh=True` to bypass the cache for one block.
    Multiple submits / optimizes in the same block are fine — each lands
    under its own input hash.
    """
    from mimiqcircuits import QCSResults
    from mimiqcircuits import OptimizationResults

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Pending jobs awaiting save on get_result; split by result type so we
    # know which loader and suffix to use.
    pending_qcs: dict[int, str] = {}
    pending_opt: dict[int, str] = {}

    def _full_key_qcs(circuit, args: tuple, kwargs: dict) -> str:
        return f"{key}__{_hash_inputs(circuit, args, kwargs)}"

    def _full_key_opt(experiments, args: tuple, kwargs: dict) -> str:
        return f"{key}__{_hash_experiments(experiments, args, kwargs)}"

    def patched_submit(self, circuit, *args, **kwargs):
        k = _full_key_qcs(circuit, args, kwargs)
        path = cache_dir / f"{k}{QCS_SUFFIX}"
        if path.exists() and not force_refresh:
            try:
                cached = QCSResults.loadproto(str(path))
                if verbose:
                    kb = path.stat().st_size / 1024
                    print(f"[mimiq_cache] hit  '{k}'  ({kb:.1f} kB, QCSResults)")
                return _CachedJob(k, cached)
            except Exception as exc:
                warnings.warn(
                    f"[mimiq_cache] '{k}{QCS_SUFFIX}' unreadable ({exc}); "
                    "resubmitting.", stacklevel=2,
                )
        if verbose:
            print(f"[mimiq_cache] miss '{k}', submitting to MIMIQ...")
        job = original_submit(self, circuit, *args, **kwargs)
        pending_qcs[id(job)] = k
        return job

    def patched_optimize(self, experiments, *args, **kwargs):
        k = _full_key_opt(experiments, args, kwargs)
        path = cache_dir / f"{k}{OPT_SUFFIX}"
        if path.exists() and not force_refresh:
            try:
                cached = OptimizationResults.loadproto(str(path))
                if verbose:
                    kb = path.stat().st_size / 1024
                    print(f"[mimiq_cache] hit  '{k}'  ({kb:.1f} kB, OptimizationResults)")
                return _CachedJob(k, cached)
            except Exception as exc:
                warnings.warn(
                    f"[mimiq_cache] '{k}{OPT_SUFFIX}' unreadable ({exc}); "
                    "resubmitting.", stacklevel=2,
                )
        if verbose:
            print(f"[mimiq_cache] miss '{k}', dispatching optimize to MIMIQ...")
        job = original_optimize(self, experiments, *args, **kwargs)
        pending_opt[id(job)] = k
        return job

    def patched_get_result(self, job, *args, **kwargs):
        if isinstance(job, _CachedJob):
            return job.result
        result = original_get_result(self, job, *args, **kwargs)
        if id(job) in pending_qcs:
            k = pending_qcs.pop(id(job))
            suffix = QCS_SUFFIX
        elif id(job) in pending_opt:
            k = pending_opt.pop(id(job))
            suffix = OPT_SUFFIX
        else:
            return result  # foreign job; not ours to cache
        path = cache_dir / f"{k}{suffix}"
        try:
            result.saveproto(str(path))
            if verbose:
                kb = path.stat().st_size / 1024
                print(f"[mimiq_cache] saved '{k}'  ({kb:.1f} kB)")
        except Exception as exc:
            warnings.warn(
                f"[mimiq_cache] could not save '{k}': {exc}",
                stacklevel=2,
            )
        return result

    original_submit = type(conn).submit
    original_optimize = type(conn).optimize
    original_get_result = type(conn).get_result

    conn.submit = types.MethodType(patched_submit, conn)
    conn.optimize = types.MethodType(patched_optimize, conn)
    conn.get_result = types.MethodType(patched_get_result, conn)
    try:
        yield
    finally:
        conn.__dict__.pop("submit", None)
        conn.__dict__.pop("optimize", None)
        conn.__dict__.pop("get_result", None)


def list_cached(cache_dir: Path | str = DEFAULT_CACHE_DIR) -> list[str]:
    """Return the cache keys (file stems) currently on disk, sorted.

    Includes both `.pb` (QCSResults) and `.optpb` (OptimizationResults) entries.
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    out: set[str] = set()
    for suffix in (QCS_SUFFIX, OPT_SUFFIX):
        for p in cache_dir.glob(f"*{suffix}"):
            out.add(p.stem)
    return sorted(out)


def clear_cache(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    key: str | None = None,
    *,
    prefix: str | None = None,
) -> int:
    """Delete cached entries. Returns the number of files removed.

    With both `key=None` and `prefix=None`: removes everything in `cache_dir`
    (`.pb`, `.optpb`, and legacy `.pkl`).

    With `key="X"`: matches `f"X__*"` — the exact namespace produced by
    `mimiq_cache(..., key="X")`.

    With `prefix="X"`: matches `f"X*"` — loose prefix, useful when a
    notebook spans several related keys (e.g. `prefix="qaoa-tsp4-"`).
    """
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return 0
    if key is not None:
        glob_root = f"{key}__*"
    elif prefix is not None:
        glob_root = f"{prefix}*"
    else:
        glob_root = "*"
    removed = 0
    for ext in ALL_SUFFIXES:
        for p in cache_dir.glob(f"{glob_root}{ext}"):
            p.unlink()
            removed += 1
    return removed


__all__ = ["mimiq_cache", "list_cached", "clear_cache",
           "DEFAULT_CACHE_DIR", "QCS_SUFFIX", "OPT_SUFFIX"]
