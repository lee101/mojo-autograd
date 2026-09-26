"""Correctness-gated benchmark for mojo-autograd.

Every case verifies numerical agreement with the reference before timing, so a
regression in the Mojo kernels shows up as a correctness failure rather than a
suspiciously good number. The elementwise VJP is compared against the three-pass
NumPy formula autograd itself evaluates; the gemm against NumPy's BLAS.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_autograd as ma  # noqa: E402
import mojo_autograd.npy as anp  # noqa: E402
from mojo_autograd import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_tanh_vjp(n: int = 1 << 22):
    """One fused pass over g, x, y against autograd's three-pass cosh formula."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    g = rng.standard_normal(n)
    y = np.tanh(x)
    theirs = g / np.cosh(x) ** 2
    assert np.allclose(_lib.vjp_unary(10, g, x, y), theirs, rtol=1e-12), "vjp mismatch"

    numpy_time = _time(lambda: g / np.cosh(x) ** 2, 3)
    mojo_time = _time(lambda: _lib.vjp_unary(10, g, x, y), 3)
    return f"tanh vjp n={n}", numpy_time, mojo_time


def bench_exp_vjp(n: int = 1 << 22):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(n)
    g = rng.standard_normal(n)
    y = np.exp(x)
    assert np.allclose(_lib.vjp_unary(0, g, x, y), g * y, rtol=1e-12)
    numpy_time = _time(lambda: g * y, 3)
    mojo_time = _time(lambda: _lib.vjp_unary(0, g, x, y), 3)
    return f"exp vjp n={n}", numpy_time, mojo_time


def bench_gemm(m: int = 600, k: int = 300, n: int = 200):
    """Compute-bound and large enough to be worth the thread fan-out."""
    rng = np.random.default_rng(2)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    assert np.allclose(_lib.gemm(a, b), a @ b, rtol=1e-11), "gemm mismatch"
    numpy_time = _time(lambda: a @ b, 3)
    mojo_time = _time(lambda: _lib.gemm(a, b), 3)
    return f"gemm {m}x{k}x{n}", numpy_time, mojo_time


def bench_small_gemm(m: int = 64, k: int = 64, n: int = 64):
    """Below the threading threshold the kernel stays serial; report the loss."""
    rng = np.random.default_rng(3)
    a = rng.standard_normal((m, k))
    b = rng.standard_normal((k, n))
    assert np.allclose(_lib.gemm(a, b), a @ b, rtol=1e-12)
    return (
        f"gemm {m}x{k}x{n} (serial)",
        _time(lambda: a @ b, 5),
        _time(lambda: _lib.gemm(a, b), 5),
    )


def bench_model(iters: int = 20):
    """End to end: a two-layer model, this port against the real autograd."""
    import autograd.numpy as aanp
    from autograd import grad as agrad

    rng = np.random.default_rng(4)
    w = rng.standard_normal((3, 2)) + 4.0
    b = rng.standard_normal(2)
    x = rng.standard_normal((4096, 3))

    def mine(w, b, x):
        return anp.mean(anp.log(anp.sum(anp.exp(anp.matmul(x, w)), axis=1)))

    def theirs(w, b, x):
        return aanp.mean(aanp.log(aanp.sum(aanp.exp(aanp.matmul(x, w)), axis=1)))

    gm = ma.grad(mine, 0)
    gt = agrad(theirs, 0)
    a = gm(w, b, x)
    c = gt(w, b, x)
    assert np.allclose(a, c, rtol=1e-9), "model gradient mismatch"

    def run(f):
        for _ in range(iters):
            f(w, b, x)

    return (
        f"model grad x{iters}",
        _time(lambda: run(gt), 3),
        _time(lambda: run(gm), 3),
    )


def main():
    print(f"{'case':<28}{'reference':>12}{'mojo-autograd':>20}{'ratio':>10}")
    print("-" * 70)
    for fn in (bench_tanh_vjp, bench_exp_vjp, bench_gemm, bench_small_gemm, bench_model):
        label, ref, got = fn()
        ratio = ref / got if got else float("nan")
        print(f"{label:<28}{ref*1e3:>10.2f}ms{got*1e3:>18.2f}ms{ratio:>9.2f}x")


if __name__ == "__main__":
    main()
