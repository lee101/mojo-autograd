"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory: the Python layer allocates every array and
passes a 64-bit address. Addresses are declared `c_int64`; `c_int` truncates
them and segfaults.
"""

from __future__ import annotations

import ctypes
import pathlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-autograd.so"

_i64 = ctypes.c_int64
_f64 = ctypes.c_double

# Chunked gemms fan out across this many workers. Below THREAD_MIN_WORK the
# call stays serial: a threaded bandwidth-bound loop is slower than a serial
# one, and the brief says so.
MAX_WORKERS = 8
THREAD_MIN_WORK = 1 << 17

_SUM, _MAX, _BCAST = 0, 1, 2

# The odometer scratch buffer is owned here; it only ever holds per-axis digit
# counters for at most 64 axes (numpy itself tops out at 32).
_SCRATCH = np.zeros(64, dtype=np.int64)


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    lib.ag_vjp_unary.restype = None
    lib.ag_vjp_unary.argtypes = [_i64] * 6

    lib.ag_vjp_binary.restype = None
    lib.ag_vjp_binary.argtypes = [_i64] * 11

    for name in ("ag_bmm_batch", "ag_bmm_rows"):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = [_i64] * 15 + [_f64, _f64, _i64, _i64]

    lib.ag_reduce.restype = None
    lib.ag_reduce.argtypes = [_i64] * 9

    lib.ag_exp_shift_scale.restype = None
    lib.ag_exp_shift_scale.argtypes = [_i64] * 15

    lib.ag_accum.restype = None
    lib.ag_accum.argtypes = [_i64] * 5 + [_f64, _f64]
    return lib


lib = _load()


def _a(x: np.ndarray) -> int:
    return x.ctypes.data


def _f64(x: np.ndarray) -> np.ndarray:
    """C-contiguous float64 that stays 0-d.

    np.ascontiguousarray promotes a 0-d array to 1-d, which would change the
    shape a caller sees coming back out of these helpers.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a).reshape(a.shape)


def _es(a: np.ndarray) -> tuple:
    """Element strides; numpy reports strides in bytes."""
    return tuple(s // a.itemsize for s in a.strides)


def _pad(shape: tuple, ndim: int) -> tuple:
    return (1,) * (ndim - len(shape)) + tuple(shape)


def _prod(shape: tuple) -> int:
    n = 1
    for s in shape:
        n *= s
    return n


def _kept_strides(shape: tuple) -> list:
    """Row-major output strides over the axes of `shape` that survive.

    An axis of extent 1 carries no information, so it gets weight 0 and is
    summed / maxed over rather than iterated.
    """
    ndim = len(shape)
    strides = [0] * ndim
    acc = 1
    for a in range(ndim - 1, -1, -1):
        if shape[a] != 1:
            strides[a] = acc
            acc *= shape[a]
    return strides


def _in_strides(shape: tuple) -> list:
    """Row-major input strides for broadcasting; extent-1 axes get weight 0."""
    return _kept_strides(shape)


# --------------------------------------------------------------------------
# unary vector-Jacobian product: out = g * f'(x)
# --------------------------------------------------------------------------
def vjp_unary(op: int, g: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    g = _f64(g)
    x = _f64(x)
    y = _f64(y)
    n = g.size
    dst = np.empty(n, dtype=np.float64)
    lib.ag_vjp_unary(op, n, _a(g), _a(x), _a(y), _a(dst))
    return dst.reshape(g.shape)


# --------------------------------------------------------------------------
# binary vector-Jacobian product: out = g * df/darg(x, y)
#
# `sy` is the element stride of the right operand, so a scalar right operand
# (a 0-d array, or a constant) costs nothing: pass sy = 0.
# --------------------------------------------------------------------------
def vjp_binary(
    op: int, arg: int, g: np.ndarray, x: np.ndarray, y: np.ndarray, sy: int = 1
) -> np.ndarray:
    g = _f64(g)
    x = _f64(x)
    y = _f64(y)
    dst = np.empty(g.shape, dtype=np.float64)
    lib.ag_vjp_binary(op, arg, g.size, _a(g), _a(x), _a(y), _a(dst), 1, 1, sy, 1)
    return dst


# --------------------------------------------------------------------------
# reductions, maxima and broadcasts, all via the same mixed-radix mapping
# --------------------------------------------------------------------------
def reduce_sum(src: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum `src` down to `shape` under numpy broadcasting rules."""
    src = _f64(src)
    shape = tuple(shape)
    if src.shape == shape:
        return src.copy()
    ndim = max(len(src.shape), len(shape), 1)
    dims = _pad(src.shape, ndim)
    tgt = _pad(shape, ndim)
    dst = np.zeros(_prod(tgt), dtype=np.float64)
    _call_reduce(_SUM, ndim, src.size, dst.size, dims, _kept_strides(tgt), src, dst)
    return dst.reshape(shape)


def reduce_max(src: np.ndarray, shape: tuple) -> np.ndarray:
    """Max-reduce `src` down to `shape` (the log-sum-exp shift)."""
    src = _f64(src)
    shape = tuple(shape)
    if src.shape == shape:
        return src.copy()
    ndim = max(len(src.shape), len(shape), 1)
    dims = _pad(src.shape, ndim)
    tgt = _pad(shape, ndim)
    dst = np.full(_prod(tgt), -np.inf, dtype=np.float64)
    _call_reduce(_MAX, ndim, src.size, dst.size, dims, _kept_strides(tgt), src, dst)
    return dst.reshape(shape)


def broadcast(src: np.ndarray, shape: tuple) -> np.ndarray:
    """Repeat `src` up to `shape`, writing a fresh C-contiguous array."""
    src = _f64(src)
    shape = tuple(shape)
    if src.shape == shape:
        return src.copy()
    ndim = max(len(src.shape), len(shape), 1)
    dims = _pad(shape, ndim)
    in_shape = _pad(src.shape, ndim)
    dst = np.empty(_prod(dims), dtype=np.float64)
    _call_reduce(_BCAST, ndim, dst.size, src.size, dims, _in_strides(in_shape), src, dst)
    return dst.reshape(shape)


def _call_reduce(mode, ndim, n_iter, n_other, dims, weights, src, dst):
    if ndim > _SCRATCH.size:  # pragma: no cover - 64 axes is past numpy's own 32
        raise ValueError("more axes than the odometer scratch buffer holds")
    d = np.array(dims, dtype=np.int64)
    w = np.array(weights, dtype=np.int64)
    lib.ag_reduce(
        mode, ndim, n_iter, n_other, _a(d), _a(w), _a(src), _a(dst), _a(_SCRATCH)
    )


# --------------------------------------------------------------------------
# out = x * exp(d - m) * inv, with m broadcast every `ms` elements
# --------------------------------------------------------------------------
def exp_shift_scale(x, d, m, r, geo_m, geo_r=(1, 1, 1), geo_x=(1, 1, 1)) -> np.ndarray:
    """out[i] = x[ix] * exp(d[i] - m[im]) * r[ir], three independent strided
    broadcasts. Each geometry is (lo, mid, hi); see the kernel comment."""
    x = _f64(x)
    d = _f64(d)
    m = _f64(m)
    r = _f64(r)
    dst = np.empty(d.size, dtype=np.float64)
    lib.ag_exp_shift_scale(
        dst.size, _a(x), _a(d), _a(m), _a(r),
        *geo_x, *geo_m, *geo_r, _a(dst),
    )
    return dst.reshape(d.shape)


# --------------------------------------------------------------------------
# gradient accumulation
# --------------------------------------------------------------------------
def accum_first(src: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """First contribution for a node: beta == 0 writes a fresh buffer, so a
    stored gradient never aliases an array a parent VJP still needs."""
    src = _f64(src)
    dst = np.empty(src.size, dtype=np.float64)
    lib.ag_accum(
        src.size, _a(src), _a(dst), 1, 1, ctypes.c_double(alpha), ctypes.c_double(0.0)
    )
    return dst.reshape(src.shape)


def accum_add(dst: np.ndarray, src: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    src = _f64(src)
    lib.ag_accum(
        src.size, _a(src), _a(dst), 1, 1, ctypes.c_double(alpha), ctypes.c_double(1.0)
    )
    return dst


# --------------------------------------------------------------------------
# matmul with arbitrary strides; a batch stride of 0 broadcasts that operand
# --------------------------------------------------------------------------
def _bmm_call(entry, m, k, n, a, b, c, sab, sbb, scb, alpha, beta, lo, hi, ov=None):
    sa, sb, sc = _es(a), _es(b), _es(c)
    sam, sak = (ov or {}).get("sam", sa[1]), (ov or {}).get("sak", sa[2])
    sbk, sbn = (ov or {}).get("sbk", sb[1]), (ov or {}).get("sbn", sb[2])
    entry(
        m, k, n, _a(a), _a(b), _a(c),
        sam, sak, sbk, sbn, sc[1], sc[2], sab, sbb, scb,
        ctypes.c_double(alpha), ctypes.c_double(beta), lo, hi,
    )


def _fanout(entry, nitems, work, args, alpha, beta, ov=None):
    if nitems > 1 and work >= THREAD_MIN_WORK:
        step = max(1, (nitems + MAX_WORKERS - 1) // MAX_WORKERS)
        parts = [(i, min(i + step, nitems)) for i in range(0, nitems, step)]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            list(
                ex.map(
                    lambda p: _bmm_call(entry, *args, alpha, beta, p[0], p[1], ov), parts
                )
            )
    else:
        _bmm_call(entry, *args, alpha, beta, 0, nitems, ov)


def bmm_batch(a, b, c, m, k, n, sab, sbb, scb, alpha=1.0, beta=0.0, ov=None):
    """c = alpha * a @ b + beta * c over batches; a is (nb, m, k), b is (nb, k, n).

    `ov` may override the derived element strides, which is how a size-1 axis
    is broadcast (a stride of 0) without materialising anything.
    """
    args = (m, k, n, a, b, c, sab, sbb, scb)
    _fanout(lib.ag_bmm_batch, c.shape[0], c.shape[0] * m * k * n, args, alpha, beta, ov)
    return c


def bmm_rows(a, b, c, m, k, n, sab, sbb, scb, alpha=1.0, beta=0.0):
    """Single-batch c = alpha * a @ b + beta * c, rows [0, m)."""
    args = (m, k, n, a, b, c, sab, sbb, scb)
    _fanout(lib.ag_bmm_rows, m, m * k * n, args, alpha, beta)
    return c


def gemm(a2, b2, c2=None, alpha=1.0, beta=0.0):
    """Plain 2-D `alpha * a @ b + beta * c` with a (m, k) and b (k, n)."""
    a2 = _f64(a2)
    b2 = _f64(b2)
    m, k = a2.shape
    n = b2.shape[1]
    if c2 is None:
        c2 = np.zeros((m, n), dtype=np.float64)
    bmm_rows(
        a2.reshape(1, m, k), b2.reshape(1, k, n), c2.reshape(1, m, n),
        m, k, n, 0, 0, 0, alpha, beta,
    )
    return c2
