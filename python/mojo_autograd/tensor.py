"""Reverse-mode tape: graph bookkeeping in Python, numeric work in Mojo.

A `Tensor` is a node on a `Tape`. Each node stores its value plus one VJP
closure per input. `Tape.backward` walks the nodes in reverse creation order (a
node can only read values that already exist, so reverse creation order is a
valid reverse topological order) and calls every closure, which lands in the
Mojo kernels in `_lib`.

The forward pass runs in NumPy, exactly as upstream autograd does: the point
of a reverse-mode AD library is the backward pass, and a fused Mojo VJP is one
memory pass where autograd's elementwise formula is three.
"""

from __future__ import annotations

import numpy as np

from . import _lib

# Op codes shared with src/kernels.mojo. Keep in step with the kernel comments.
UNARY_OPS = {
    "exp": 0,
    "expm1": 1,
    "exp2": 2,
    "log": 3,
    "log2": 4,
    "log10": 5,
    "log1p": 6,
    "sin": 7,
    "cos": 8,
    "tan": 9,
    "tanh": 10,
    "sinh": 11,
    "cosh": 12,
    "arcsin": 13,
    "arccos": 14,
    "arctan": 15,
    "arcsinh": 16,
    "arccosh": 17,
    "arctanh": 18,
    "square": 19,
    "sqrt": 20,
    "reciprocal": 21,
    "fabs": 22,
    "sigmoid": 23,
    "softplus": 24,
    "relu": 25,
}

BINARY_OPS = {
    "add": 0,
    "subtract": 1,
    "multiply": 2,
    "divide": 3,
    "power": 4,
    "maximum": 5,
    "minimum": 6,
    "logaddexp": 7,
    "arctan2": 8,
    "hypot": 9,
}


def value_of(x) -> np.ndarray:
    return x.value if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)


def _resolve_shape(shape, old) -> tuple:
    if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    else:
        shape = tuple(int(s) for s in shape)
    if -1 in shape:
        known = 1
        for s in shape:
            if s != -1:
                known *= s
        shape = tuple(int(np.prod(old)) // known if s == -1 else s for s in shape)
    return shape


def _mid_shape(shape, axis):
    """`shape` with a 1 at every reduced axis: where the seed belongs."""
    if axis is None:
        return (1,) * len(shape)
    a = int(axis)
    if a < 0:
        a += len(shape)
    return shape[:a] + (1,) + shape[a + 1 :]


def _contig(x):
    """C-contiguous float64 that stays 0-d.

    np.ascontiguousarray promotes a 0-d array to 1-d, which would silently
    change the shape of a scalar value or gradient.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.flags.c_contiguous:
        return a
    return np.ascontiguousarray(a).reshape(a.shape)


def _unshape(c, shape):
    out = np.reshape(c, tuple(shape))
    return _contig(out)


def _fit(c, shape):
    """Reshape a computed contribution to `shape`, summing the broadcast axes.

    A contribution can only be larger than its target in *leading* axes (that
    is where a batch dimension was broadcast), so the contribution is folded to
    (prod(leading), prod(trailing)) and the leading axis is summed away by the
    Mojo reduce kernel.
    """
    shape = tuple(shape)
    c = _contig(c)
    if tuple(c.shape) == shape:
        return c
    if c.ndim < len(shape):
        raise ValueError(f"contribution {c.shape} cannot be reshaped to {shape}")
    keep = int(np.prod(shape)) if shape else 1
    flat = np.reshape(c, (-1, keep))
    return np.reshape(_lib.reduce_sum(flat, (1, keep)), shape)


class Tensor:
    """A node on a tape. `value` is always C-contiguous float64."""

    __slots__ = ("value", "parents", "grad", "tape", "is_leaf", "name")

    def __init__(self, value, parents=(), tape=None, is_leaf=False, name=None):
        self.value = value
        self.parents = list(parents)
        self.tape = tape
        self.is_leaf = is_leaf
        self.name = name
        self.grad = None

    # -- plumbing ---------------------------------------------------------
    @property
    def shape(self):
        return self.value.shape

    @property
    def ndim(self):
        return self.value.ndim

    @property
    def size(self):
        return self.value.size

    def __len__(self):
        return len(self.value)

    def __array__(self, dtype=None, copy=None):
        if dtype is not None and np.dtype(dtype) != np.dtype(np.float64):
            raise TypeError("mojo_autograd tensors are float64 only")
        return np.array(self.value, copy=True)

    def _accum(self, contrib: np.ndarray) -> None:
        if tuple(contrib.shape) != tuple(self.value.shape):
            raise ValueError(
                f"gradient shape {contrib.shape} does not match value shape "
                f"{self.value.shape}"
            )
        if self.grad is None:
            self.grad = _lib.accum_first(contrib)
        else:
            _lib.accum_add(self.grad, contrib)

    def _node(self, value, parents):
        return self.tape._node(value, parents)

    # -- elementwise ------------------------------------------------------
    def _unary(self, op: int, npfn) -> "Tensor":
        y = _contig(npfn(self.value))
        xv, yv = self.value, y
        return self._node(
            y, [(self, lambda g, o=op, a=xv, b=yv: _lib.vjp_unary(o, g, a, b))]
        )

    def _binary(self, other, op: int, npfn) -> "Tensor":
        ov = np.asarray(value_of(other), dtype=np.float64)
        v = _contig(npfn(self.value, ov))
        if ov.shape == self.value.shape:
            xv, yv, sy = self.value, ov, 1
        elif ov.size == 1:
            # Scalar right operand: the kernel's stride-0 read handles it free.
            xv, yv, sy = self.value, ov, 0
        else:
            xv = _lib.broadcast(self.value, v.shape)
            yv = _lib.broadcast(ov, v.shape)
            sy = 1
        parents = []
        sshape = self.value.shape
        parents.append(
            (
                self,
                lambda g, o=op, a=xv, b=yv, s=sshape, k=sy: _lib.reduce_sum(
                    _lib.vjp_binary(o, 0, g, a, b, k), s
                ),
            )
        )
        if isinstance(other, Tensor):
            oshape = other.value.shape
            parents.append(
                (
                    other,
                    lambda g, o=op, a=xv, b=yv, s=oshape, k=sy: _lib.reduce_sum(
                        _lib.vjp_binary(o, 1, g, a, b, k), s
                    ),
                )
            )
        return self._node(v, parents)

    def _r_binary(self, other, op: int, npfn) -> "Tensor":
        """Reversed operand order: the differentiated side is the right one."""
        ov = np.asarray(value_of(other), dtype=np.float64)
        xv = self.value
        v = _contig(npfn(ov, xv))
        if ov.shape == xv.shape:
            a, b = ov, xv
        else:
            a = _lib.broadcast(ov, v.shape)
            b = _lib.broadcast(xv, v.shape)
        parents = [
            (
                self,
                lambda g, o=op, p=a, q=b, s=xv.shape: _lib.reduce_sum(
                    _lib.vjp_binary(o, 1, g, p, q), s
                ),
            )
        ]
        if isinstance(other, Tensor):
            oshape = other.value.shape
            parents.append(
                (
                    other,
                    lambda g, o=op, p=a, q=b, s=oshape: _lib.reduce_sum(
                        _lib.vjp_binary(o, 0, g, p, q), s
                    ),
                )
            )
        return self._node(v, parents)

    # -- reductions -------------------------------------------------------
    def sum(self, axis=None, keepdims=False) -> "Tensor":
        y = np.asarray(self.value.sum(axis=axis, keepdims=keepdims), dtype=np.float64)
        xv = self.value
        mid = _mid_shape(xv.shape, axis)
        return self._node(
            y, [(self, lambda g, a=xv, m=mid: _lib.broadcast(np.reshape(g, m), a.shape))]
        )

    def mean(self, axis=None, keepdims=False) -> "Tensor":
        y = np.asarray(self.value.mean(axis=axis, keepdims=keepdims), dtype=np.float64)
        xv = self.value
        mid = _mid_shape(xv.shape, axis)
        n = xv.size / y.size if y.size else 1.0
        return self._node(
            y,
            [
                (
                    self,
                    lambda g, a=xv, m=mid, k=n: _lib.broadcast(np.reshape(g, m), a.shape)
                    / k,
                )
            ],
        )

    def logsumexp(self, axis=None, keepdims=False) -> "Tensor":
        return logsumexp(self, axis, keepdims)

    # -- views (the VJP of a view is the inverse view) --------------------
    def reshape(self, *shape) -> "Tensor":
        shape = _resolve_shape(shape, self.value.shape)
        y = np.reshape(self.value, shape)
        return self._node(y, [(self, lambda g, s=self.value.shape: np.reshape(g, s))])

    def transpose(self, *axes) -> "Tensor":
        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = axes[0]
        elif not axes:
            axes = None
        y = np.transpose(self.value, axes)
        inv = None if axes is None else np.argsort(axes)
        return self._node(
            y, [(self, lambda g, i=inv: np.transpose(g, i) if i is not None else g.T)]
        )

    def swapaxes(self, a1, a2) -> "Tensor":
        y = np.swapaxes(self.value, a1, a2)
        return self._node(y, [(self, lambda g, p=a1, q=a2: np.swapaxes(g, q, p))])

    def squeeze(self, axis=None) -> "Tensor":
        y = np.squeeze(self.value, axis)
        return self._node(y, [(self, lambda g, s=self.value.shape: np.reshape(g, s))])

    def expand_dims(self, axis) -> "Tensor":
        y = np.expand_dims(self.value, axis)
        return self._node(y, [(self, lambda g, s=self.value.shape: np.reshape(g, s))])

    # -- operators --------------------------------------------------------
    def __add__(self, o):
        return self._binary(o, BINARY_OPS["add"], np.add)

    __radd__ = __add__

    def __sub__(self, o):
        return self._binary(o, BINARY_OPS["subtract"], np.subtract)

    def __rsub__(self, o):
        return self._r_binary(o, BINARY_OPS["subtract"], np.subtract)

    def __mul__(self, o):
        return self._binary(o, BINARY_OPS["multiply"], np.multiply)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return self._binary(o, BINARY_OPS["divide"], np.divide)

    def __rtruediv__(self, o):
        return self._r_binary(o, BINARY_OPS["divide"], np.divide)

    def __pow__(self, o):
        return self._binary(o, BINARY_OPS["power"], np.power)

    def __rpow__(self, o):
        return self._r_binary(o, BINARY_OPS["power"], np.power)

    def __neg__(self):
        return self._binary(-1.0, BINARY_OPS["multiply"], np.multiply)

    def __matmul__(self, o):
        return matmul(self, o)

    def __repr__(self):
        tag = f"leaf {self.name!r}" if self.is_leaf else "node"
        shape = None if self.value is None else self.value.shape
        return f"<mojo_autograd.Tensor {tag} shape={shape}>"


# --------------------------------------------------------------------------
# log-sum-exp
# --------------------------------------------------------------------------
def logsumexp(node: Tensor, axis=None, keepdims: bool = False) -> Tensor:
    """log(sum(exp(x))), computed as max + log(sum(exp(x - max))).

    The max-reduction, the shifted exponential and the softmax-weighted VJP are
    Mojo kernels; only the final scalar logarithm and the adds stay in NumPy.
    """
    if isinstance(axis, (tuple, list)):
        raise NotImplementedError(
            "mojo_autograd.logsumexp takes a single axis or None, not a tuple"
        )
    x = node.value
    if axis is None:
        red_shape: tuple = ()
        geo = (1, 1, 1)
    else:
        a = int(axis)
        if a < 0:
            a += x.ndim
        red_shape = x.shape[:a] + (1,) + x.shape[a + 1 :]
        geo = (
            int(np.prod(x.shape[a + 1 :])) if x.shape[a + 1 :] else 1,
            x.shape[a],
            int(np.prod(x.shape[:a])) if x.shape[:a] else 1,
        )
    m = _lib.reduce_max(x, red_shape)
    one = np.ones(1, dtype=np.float64)
    shifted = _lib.exp_shift_scale(one, x, m, one, geo)
    if axis is None:
        s = np.asarray(shifted.sum(), dtype=np.float64).reshape(())
    else:
        s = np.asarray(shifted.sum(axis=axis, keepdims=True), dtype=np.float64)
    y = m + np.log(s)
    if axis is not None and not keepdims:
        y = np.squeeze(y, axis=axis)
    return node._node(
        _contig(y),
        [
            (
                node,
                lambda g, a=x, mm=m, ss=s, ge=geo: _lib.exp_shift_scale(
                    g, a, mm, 1.0 / ss, ge, ge, ge
                ),
            )
        ],
    )


# --------------------------------------------------------------------------
# matmul
#
# Every adjoint is expressed as a strided batched gemm, mirroring
# autograd/numpy/numpy_vjps.py: matmul_adjoint_0 / matmul_adjoint_1. A 1-D
# operand becomes a 1-row or 1-column matrix, which the kernel handles with a
# zero batch stride, so nothing is ever copied to promote it.
# --------------------------------------------------------------------------
def _batch(a, m, k):
    """3-D (nb, m, k) view plus its element batch stride."""
    a3 = np.reshape(a, (-1, m, k))
    nb = a3.shape[0]
    return a3, (a3.strides[0] // 8 if nb > 1 else 0)


def matmul(a, b) -> Tensor:
    av = _contig(value_of(a))
    bv = _contig(value_of(b))
    if av.ndim == 0 or bv.ndim == 0:
        raise ValueError("matmul operands need at least one dimension")
    da, db = av.ndim, bv.ndim
    ap = av.reshape(1, -1) if da == 1 else av
    bp = bv.reshape(-1, 1) if db == 1 else bv
    m, k = ap.shape[-2], ap.shape[-1]
    k2, n = bp.shape[-2], bp.shape[-1]
    if k != k2:
        raise ValueError(f"matmul shape mismatch: {av.shape} @ {bv.shape}")
    batch = np.broadcast_shapes(ap.shape[:-2], bp.shape[:-2])
    nb = int(np.prod(batch)) if batch else 1
    a3, sab = _batch(ap, m, k)
    b3, sbb = _batch(bp, k, n)
    if a3.shape[0] == 1:
        sab = 0
    if b3.shape[0] == 1:
        sbb = 0
    if a3.shape[0] not in (1, nb) or b3.shape[0] not in (1, nb):
        raise NotImplementedError("per-axis batch broadcasting is not supported")
    c3 = np.empty((nb, m, n), dtype=np.float64)
    _lib.bmm_batch(a3, b3, c3, m, k, n, sab, sbb, c3.strides[0] // 8, 1.0, 0.0)
    y = c3.reshape(batch + (m, n))
    if da == 1:
        # A promoted to (1, k) contributes the row axis, which sits at
        # len(batch) in the reshaped result.
        y = np.squeeze(y, len(batch))
    if db == 1:
        y = np.squeeze(y, -1)
    y = _contig(y)
    at = a if isinstance(a, Tensor) else None
    bt = b if isinstance(b, Tensor) else None
    parents = []
    if at is not None:
        parents.append((at, lambda g, A=av, B=bv: matmul_vjp_0(g, A, B)))
    if bt is not None:
        parents.append((bt, lambda g, A=av, B=bv: matmul_vjp_1(g, A, B)))
    tape = at.tape if at is not None else bt.tape
    return tape._node(y, parents)


def _gemm(A3, B3, C3, m, k, n, sab, sbb, scb, ov=None):
    _lib.bmm_batch(A3, B3, C3, m, k, n, sab, sbb, scb, 1.0, 0.0, ov)
    return C3


def matmul_vjp_0(g, av, bv):
    """d/dA of A @ B: G @ swap(B, -1, -2), with A promoted if it was 1-D."""
    da, db = av.ndim, bv.ndim
    g = _contig(g)
    if da == 1 and db == 1:
        return float(g) * bv
    if da == 1:
        # A is (k,): dA = G (nb, 1, n) @ B^T (nb, n, k) -> (nb, 1, k)
        k, n = bv.shape[-2], bv.shape[-1]
        b3, sbb = _batch(bv, k, n)
        nb = b3.shape[0]
        g3 = np.reshape(g, (nb, 1, n))
        c3 = np.empty((nb, 1, k), dtype=np.float64)
        _gemm(
            g3, b3.swapaxes(1, 2), c3, 1, n, k,
            g3.strides[0] // 8, sbb, c3.strides[0] // 8,
        )
        return _fit(c3, av.shape)
    m, k = av.shape[-2], av.shape[-1]
    if db == 1:
        # B is (k,): dA is the outer product G[..., :, None] * B
        nb = g.size // m
        g3 = np.reshape(g, (nb, m, 1))
        b3 = bv.reshape(1, 1, k)
        c3 = np.empty((max(nb, 1), m, k), dtype=np.float64)
        _gemm(
            g3, b3, c3, m, 1, k,
            g3.strides[0] // 8, 0, c3.strides[0] // 8,
        )
        return _fit(c3, av.shape)
    n = bv.shape[-1]
    nb = g.size // (m * n)
    g3 = np.reshape(g, (nb, m, n))
    b3, sbb = _batch(bv, k, n)
    c3 = np.empty((max(nb, b3.shape[0]), m, k), dtype=np.float64)
    _gemm(
        g3, b3.swapaxes(1, 2), c3, m, n, k,
        g3.strides[0] // 8, sbb, c3.strides[0] // 8,
    )
    return _fit(c3, av.shape)


def matmul_vjp_1(g, av, bv):
    """d/dB of A @ B: swap(A, -1, -2) @ G, with B promoted if it was 1-D."""
    da, db = av.ndim, bv.ndim
    g = _contig(g)
    if da == 1 and db == 1:
        return float(g) * av
    if da == 1:
        # A is (k,): dB = A[:, None] (1, 1, k) @ G (nb, 1, n) -> (nb, k, n)
        k = av.shape[0]
        n = bv.shape[-1]
        nb = g.size // n
        g3 = np.reshape(g, (nb, 1, n))
        c3 = np.empty((nb, k, n), dtype=np.float64)
        # (k, 1) @ (1, n): the contraction is over the length-1 axis, so the
        # kernel is m = k, k = 1, n = n and nothing needs broadcasting.
        a3 = av.reshape(1, k, 1)
        _gemm(a3, g3, c3, k, 1, n, 0, g3.strides[0] // 8, c3.strides[0] // 8)
        return _fit(c3, bv.shape)
    m, k = av.shape[-2], av.shape[-1]
    if db == 1:
        # B is (k,): dB[j] = sum_i A[i, j] * G[i]  ->  A^T (nb, k, m) @ G (nb, m, 1)
        nb = g.size // m
        g3 = np.reshape(g, (nb, m, 1))
        a3, sab = _batch(av, m, k)
        c3 = np.empty((max(nb, a3.shape[0]), k, 1), dtype=np.float64)
        _gemm(
            a3.swapaxes(1, 2), g3, c3, k, m, 1,
            sab, g3.strides[0] // 8, c3.strides[0] // 8,
        )
        return _fit(c3, bv.shape)
    n = bv.shape[-1]
    nb = g.size // (m * n)
    g3 = np.reshape(g, (nb, m, n))
    a3, sab = _batch(av, m, k)
    c3 = np.empty((max(nb, a3.shape[0]), k, n), dtype=np.float64)
    _gemm(
        a3.swapaxes(1, 2), g3, c3, k, m, n,
        sab, g3.strides[0] // 8, c3.strides[0] // 8,
    )
    return _fit(c3, bv.shape)


# --------------------------------------------------------------------------
# the tape
# --------------------------------------------------------------------------
class Tape:
    """Records nodes and runs the reverse pass.

    Reverse creation order is a valid reverse topological order, because a node
    is only ever built from values that already exist: every consumer of a node
    has a higher index and is therefore visited first, so a node's gradient is
    complete before it is used.
    """

    def __init__(self):
        self.nodes: list[Tensor] = []

    def backward(self, out: Tensor, seed=None) -> Tensor:
        """Seed `out` with ones (autograd's convention) and run the tape."""
        if out.grad is None:
            s = np.ones_like(out.value) if seed is None else np.asarray(seed, np.float64)
            out.grad = _lib.accum_first(s)
        for node in reversed(self.nodes):
            g = node.grad
            if g is None:
                continue
            for parent, vjp in node.parents:
                parent._accum(vjp(g))
            if not node.is_leaf:
                # Only intermediates are recycled: a leaf's gradient is what the
                # caller asked for, and the output node's value is the result.
                node.grad = None
                if node is not out:
                    # Every consumer has run, so the value is dead now.
                    node.value = None
        return out

    def leaf(self, value, name=None) -> Tensor:
        v = np.array(value, dtype=np.float64, order="C", copy=True)
        t = Tensor(v, tape=self, is_leaf=True, name=name)
        self.nodes.append(t)
        return t

    def _node(self, value, parents) -> Tensor:
        t = Tensor(value, parents=parents, tape=self)
        self.nodes.append(t)
        return t

    def gradients(self) -> list:
        return [n.grad for n in self.nodes if n.is_leaf]

    def gradient(self, out: Tensor, argnum: int = 0):
        grads = self.gradients()
        if isinstance(argnum, (list, tuple)):
            return [grads[i] for i in argnum]
        return grads[argnum]
