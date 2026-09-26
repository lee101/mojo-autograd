"""Parity against the real autograd, plus analytic and kernel-level checks.

Tolerances are explicit everywhere. Mojo emits FMA, so a VJP that ends in a
multiply-add differs from NumPy's separate operations in the last bits; only
pure index and copy work is asserted exactly.
"""

import numpy as np
import pytest

import mojo_autograd as ma
import mojo_autograd.npy as anp
from mojo_autograd import _lib

autograd = pytest.importorskip("autograd")
import autograd.numpy as aanp  # noqa: E402
from autograd import grad as agrad  # noqa: E402

RTOL = 1e-9
ATOL = 1e-11


def parity(mine, theirs, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(mine, theirs, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# analytic ground truth
# ---------------------------------------------------------------------------
def test_gradient_of_sum_exp_is_exp():
    rng = np.random.default_rng(0)
    v = rng.uniform(0.2, 1.5, size=64)
    np.testing.assert_allclose(
        ma.grad(lambda t: anp.sum(anp.exp(t)))(v), np.exp(v), rtol=1e-12
    )


def test_gradient_of_sum_squares_is_two_v():
    v = np.linspace(-3.0, 3.0, 101)
    np.testing.assert_allclose(
        ma.grad(lambda t: anp.sum(t * t))(v), 2.0 * v, rtol=1e-12, atol=1e-13
    )


def test_gradient_of_mean_is_uniform():
    x = np.random.default_rng(1).standard_normal((5, 7))
    np.testing.assert_allclose(
        ma.grad(lambda t: anp.mean(t))(x), np.full((5, 7), 1.0 / 35.0), rtol=1e-12
    )


def test_value_and_grad_returns_both():
    v = np.array([0.5, 1.5])
    val, g = ma.value_and_grad(lambda t: anp.sum(anp.log(t)))(v)
    assert val == pytest.approx(float(np.log(v).sum()))
    np.testing.assert_allclose(g, 1.0 / v, rtol=1e-12)


# ---------------------------------------------------------------------------
# parity with the real autograd
# ---------------------------------------------------------------------------
UNARY_DOMAINS = {
    "arcsin": (0.0, 0.9),
    "arccos": (0.0, 0.9),
    "arcsinh": (-3.0, 3.0),
    "arccosh": (1.2, 3.0),
    "arctanh": (-0.9, 0.9),
}
# numpy 2 dropped these three names, so autograd cannot be the reference; the
# derivative of each is spelled out here instead.
CUSTOM_DERIV = {
    "sigmoid": lambda t: (lambda s: s * (1.0 - s))(1.0 / (1.0 + np.exp(-t))),
    "softplus": lambda t: 1.0 / (1.0 + np.exp(-t)),
    "relu": lambda t: (t > 0.0).astype(np.float64),
}


@pytest.mark.parametrize(
    "name", sorted(ma.UNARY_OPS), ids=lambda n: f"unary-{n}"
)
def test_unary_vjp_matches_autograd(name):
    lo, hi = UNARY_DOMAINS.get(name, (0.3, 1.5))
    rng = np.random.default_rng(hash(name) % 2**32)
    x = rng.uniform(lo, hi, size=32)
    mine = ma.grad(lambda t: getattr(anp, name)(t).sum())(x)
    custom = CUSTOM_DERIV.get(name)
    if custom is not None:
        parity(mine, custom(x))
        return
    parity(mine, agrad(lambda t: getattr(aanp, name)(t).sum())(x))


@pytest.mark.parametrize(
    "name", sorted(ma.BINARY_OPS), ids=lambda n: f"binary-{n}"
)
def test_binary_vjp_matches_autograd(name):
    rng = np.random.default_rng(1000 + hash(name) % 2**31)
    x = rng.uniform(0.5, 2.0, size=32)
    y = rng.uniform(0.5, 2.0, size=32)
    if name == "arctan2":
        y = rng.uniform(-2.0, 2.0, size=32)
    parity(
        ma.grad(lambda t, s: getattr(anp, name)(t, s).sum())(x, y),
        agrad(lambda t, s: getattr(aanp, name)(t, s).sum())(x, y),
        rtol=1e-8,
    )


def test_binary_vjp_unbroadcasts_broadcast_operands():
    """A (3, 1) * (1, 4) product must sum each contribution back to its own
    operand shape, or the accumulated gradient has the wrong shape."""
    rng = np.random.default_rng(3)
    p = rng.standard_normal((3, 1))
    q = rng.standard_normal((1, 4))
    gp, gq = ma.grad(lambda t, s: anp.sum(anp.multiply(t, s)), argnum=(0, 1))(p, q)
    assert gp.shape == (3, 1)
    assert gq.shape == (1, 4)
    # d(sum(p*q))/dp[i, 0] sums the whole q row; d/dq[0, j] sums the whole p column
    parity(gp[:, 0], q.sum() * np.ones(3))
    parity(gq[0, :], p.sum() * np.ones(4))


def test_reductions_match_autograd():
    x = np.random.default_rng(4).standard_normal((3, 4, 5))
    cases = [
        (lambda t: anp.sum(t), lambda t: aanp.sum(t)),
        (lambda t: anp.sum(anp.sum(t, axis=0)), lambda t: aanp.sum(aanp.sum(t, axis=0))),
        (lambda t: anp.sum(anp.mean(t, axis=1)),
         lambda t: aanp.sum(aanp.mean(t, axis=1))),
        (lambda t: anp.sum(anp.mean(t, axis=2, keepdims=True) ** 2),
         lambda t: aanp.sum(aanp.mean(t, axis=2, keepdims=True) ** 2)),
    ]
    for mine, theirs in cases:
        parity(ma.grad(mine)(x), agrad(theirs)(x))


def test_view_ops_match_autograd():
    x = np.random.default_rng(5).standard_normal((2, 3, 4))
    for mine, theirs in [
        (lambda t: anp.sum(anp.transpose(t)), lambda t: aanp.sum(aanp.transpose(t))),
        (lambda t: anp.sum(anp.swapaxes(t, 0, 2)),
         lambda t: aanp.sum(aanp.swapaxes(t, 0, 2))),
        (lambda t: anp.sum(anp.reshape(t, (6, 4)) * 2.0),
         lambda t: aanp.sum(aanp.reshape(t, (6, 4)) * 2.0)),
        (lambda t: anp.sum(anp.squeeze(anp.expand_dims(t, 1))),
         lambda t: aanp.sum(aanp.squeeze(aanp.expand_dims(t, 1)))),
    ]:
        parity(ma.grad(mine)(x), agrad(theirs)(x))


MATMUL_CASES = [
    ((3, 2), (2,)),
    ((4,), (4, 2)),
    ((3, 4), (4, 2)),
    ((5, 3, 4), (4, 6)),
    ((2, 3, 4), (4,)),
    ((4,), (2, 4, 3)),
    ((3, 3), (3, 3)),
]


@pytest.mark.parametrize("ashape,bshape", MATMUL_CASES, ids=lambda s: "x".join(map(str, s)))
def test_matmul_vjp_matches_autograd(ashape, bshape):
    rng = np.random.default_rng(hash((ashape, bshape)) % 2**32)
    a = rng.standard_normal(ashape)
    b = rng.standard_normal(bshape)
    k = ashape[-1]
    if len(bshape) == 1:
        b = rng.standard_normal((k,)) + 3.0
    for argnum in (0, 1):
        mine = ma.grad(
            lambda t, s: anp.sum(anp.matmul(t, s) ** 2), argnum=argnum
        )(a, b)
        theirs = agrad(
            lambda t, s: aanp.sum(aanp.matmul(t, s) ** 2), argnum=argnum
        )(a, b)
        parity(mine, theirs, rtol=1e-8)


def test_matmul_broadcast_batch_matches_autograd():
    rng = np.random.default_rng(6)
    a = rng.standard_normal((2, 3, 4))
    b = rng.standard_normal((4, 6))
    mine = ma.grad(lambda t, s: anp.sum(anp.matmul(t, s)), argnum=(0, 1))(a, b)
    theirs = agrad(lambda t, s: aanp.sum(aanp.matmul(t, s)), argnum=(0, 1))(a, b)
    for m, t in zip(mine, theirs):
        parity(m, t, rtol=1e-8)


def test_logsumexp_value_and_vjp():
    rng = np.random.default_rng(7)
    x = rng.standard_normal((3, 4))
    for axis in (0, 1, None):
        mine = ma.grad(lambda t: anp.sum(anp.logsumexp(t, axis=axis) ** 2))(x)
        ref = x.max(axis=axis, keepdims=True)
        stable = np.log(np.exp(x - ref).sum(axis=axis, keepdims=True)) + ref
        theirs = agrad(
            lambda t: aanp.sum((stable_ref(t, axis)) ** 2)
        )(x)
        parity(mine, theirs, rtol=1e-7)
    val = ma.value_and_grad(lambda t: anp.logsumexp(t))(x)[0]
    assert float(val) == pytest.approx(float(np.log(np.exp(x).sum())), rel=1e-12)


def stable_ref(t, axis):
    m = aanp.max(t, axis=axis, keepdims=True)
    return m + aanp.log(aanp.sum(aanp.exp(t - m), axis=axis, keepdims=True))


def test_combined_model_matches_autograd():
    rng = np.random.default_rng(8)
    w = rng.standard_normal((3, 2)) + 4.0
    b = rng.standard_normal(2)
    x = rng.standard_normal((5, 3))

    def mine(w, b, x):
        h = anp.tanh(anp.matmul(x, w)) + b
        return anp.mean(anp.log(anp.sum(anp.exp(h), axis=1)))

    def theirs(w, b, x):
        h = aanp.tanh(aanp.matmul(x, w)) + b
        return aanp.mean(aanp.log(aanp.sum(aanp.exp(h), axis=1)))

    for argnum in (0, 1, 2):
        parity(
            ma.grad(mine, argnum)(w, b, x),
            agrad(theirs, argnum)(w, b, x),
            rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# tape behaviour
# ---------------------------------------------------------------------------
def test_tape_frees_intermediates_but_keeps_leaf_gradients():
    from mojo_autograd.tensor import Tape

    tape = Tape()
    v = tape.leaf(np.ones(4))
    y = v * v
    z = anp.matmul(y, np.eye(4))
    out = anp.sum(z)
    tape.backward(out)
    assert np.allclose(tape.gradients()[0], 2.0 * np.ones(4))
    assert y.value is None  # every consumer has run
    assert tape.nodes[0].value is not None  # leaves keep their input


def test_unsupported_op_raises():
    from mojo_autograd.tensor import Tape

    t = Tape()
    with pytest.raises(TypeError, match="no gradient rule"):
        anp.sinc(t.leaf(np.ones(3)))


def test_npy_passes_through_to_numpy_without_tensors():
    x = np.arange(4.0)
    np.testing.assert_array_equal(anp.exp(x), np.exp(x))
    assert not isinstance(anp.sum(x), ma.Tensor)


# ---------------------------------------------------------------------------
# kernel level
# ---------------------------------------------------------------------------
def test_vjp_unary_matches_central_differences():
    rng = np.random.default_rng(9)
    x = rng.uniform(0.3, 1.2, size=256)
    g = rng.standard_normal(256)
    h = 1e-6
    cases = {
        0: np.exp,
        3: np.log,
        7: np.sin,
        10: np.tanh,
        19: np.square,
        20: np.sqrt,
        22: np.fabs,
    }
    for op, fn in cases.items():
        num = (fn(x + h) - fn(x - h)) / (2 * h)
        parity(_lib.vjp_unary(op, g, x, fn(x)), g * num, rtol=1e-5, atol=1e-7)


def test_vjp_unary_relu_and_softplus():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    g = np.ones(5)
    np.testing.assert_array_equal(_lib.vjp_unary(25, g, x, np.maximum(x, 0)),
                                  [0.0, 0.0, 0.0, 1.0, 1.0])
    sig = 1.0 / (1.0 + np.exp(-x))
    parity(_lib.vjp_unary(24, g, x, sig), 1.0 / (1.0 + np.exp(-x)))


def test_vjp_binary_scalar_stride_matches_explicit_broadcast():
    """A 0-d right operand read with stride 0 must equal the broadcast form."""
    rng = np.random.default_rng(10)
    x = rng.uniform(0.5, 2.0, size=128)
    s = np.array(1.7)
    g = rng.standard_normal(128)
    parity(_lib.vjp_binary(2, 0, g, x, s, 0), _lib.vjp_binary(2, 0, g, x, np.full(128, 1.7)))
    parity(_lib.vjp_binary(3, 1, g, x, s, 0), _lib.vjp_binary(3, 1, g, x, np.full(128, 1.7)))


def test_vjp_binary_power_second_argument_uses_the_result():
    """d/dy of x**y is g*log(x)*x**y, not g*log(x)*y."""
    x = np.array([2.0, 3.0])
    y = np.array([4.0, 5.0])
    g = np.ones(2)
    parity(_lib.vjp_binary(4, 1, g, x, y), np.log(x) * x**y)


def test_vjp_binary_add_gives_plus_to_both_operands():
    g = np.arange(3.0)
    x = np.arange(3.0)
    np.testing.assert_array_equal(_lib.vjp_binary(0, 0, g, x, x), g)
    np.testing.assert_array_equal(_lib.vjp_binary(0, 1, g, x, x), g)
    np.testing.assert_array_equal(_lib.vjp_binary(1, 1, g, x, x), -g)


def test_vjp_binary_maximum_is_balanced_at_a_tie():
    x = np.array([1.0, 1.0, 2.0])
    y = np.array([1.0, 0.5, 1.0])
    g = np.ones(3)
    parity(_lib.vjp_binary(5, 0, g, x, y), [0.5, 1.0, 1.0])
    parity(_lib.vjp_binary(5, 1, g, x, y), [0.5, 0.0, 0.0])


def test_reduce_sum_matches_numpy_on_many_shapes():
    rng = np.random.default_rng(11)
    # (source shape, reduced shape with the summed axes kept as 1, the axes)
    cases = [
        ((3, 1, 5), (3, 1, 1), (2,)),
        ((2, 3, 4, 5), (2, 3, 4, 1), (3,)),
        ((2, 3, 4, 5), (2, 1, 4, 5), (1,)),
        ((2, 3, 4, 5), (2, 3, 4, 5), ()),
        ((2, 3, 4, 5), (1, 3, 1, 1), (0, 2, 3)),
    ]
    for src_shape, out_shape, axes in cases:
        src = rng.standard_normal(src_shape)
        expect = src.sum(axis=axes, keepdims=True)
        # The kernel accumulates in its own order, so only tolerance applies.
        parity(_lib.reduce_sum(src, out_shape), expect, rtol=1e-12, atol=1e-12)


def test_broadcast_matches_numpy():
    rng = np.random.default_rng(12)
    for src, shape in [
        (rng.standard_normal(3), (2, 3)),
        (rng.standard_normal(4), (2, 3, 4)),
        (rng.standard_normal(1), (2, 2)),
        (rng.standard_normal(3), (3,)),
    ]:
        parity(_lib.broadcast(src, shape), np.broadcast_to(src, shape), rtol=0, atol=0)


def test_reduce_max_matches_numpy():
    rng = np.random.default_rng(13)
    src = rng.standard_normal((4, 5, 6))
    parity(_lib.reduce_max(src, (4, 1, 6)), src.max(axis=1, keepdims=True))
    parity(_lib.reduce_max(src, (1, 5, 1)), src.max(axis=(0, 2), keepdims=True))
    parity(_lib.reduce_max(src, ()), src.max().reshape(()))


def test_gemm_matches_numpy_and_accumulates():
    rng = np.random.default_rng(14)
    a = rng.standard_normal((17, 9))
    b = rng.standard_normal((9, 5))
    parity(_lib.gemm(a, b), a @ b, rtol=1e-12, atol=1e-12)
    c = rng.standard_normal((17, 5))
    parity(_lib.gemm(a, b, c.copy(), 1.0, 1.0), a @ b + c, rtol=1e-12, atol=1e-12)


def test_gemm_threaded_batch_matches_numpy():
    rng = np.random.default_rng(15)
    a = rng.standard_normal((24, 40, 60))
    b = rng.standard_normal((24, 60, 30))
    c = np.zeros((24, 40, 30))
    _lib.bmm_batch(
        a, b, c, 40, 60, 30,
        a.strides[0] // 8, b.strides[0] // 8, c.strides[0] // 8,
    )
    parity(c, a @ b, rtol=1e-11, atol=1e-12)


def test_exp_shift_scale_matches_broadcast_exponentials():
    rng = np.random.default_rng(16)
    # The axis=None geometry (1, 1, 1) is covered by the logsumexp test above.
    for shape, axis in [((3, 4), 0), ((3, 4), 1), ((2, 3, 4), 2)]:
        x = rng.standard_normal(shape)
        m = _lib.reduce_max(x, _mid(shape, axis))
        geo = _geo(shape, axis)
        one = np.ones(1)
        parity(
            _lib.exp_shift_scale(one, x, m, one, geo),
            np.exp(x - np.broadcast_to(m, shape)),
            rtol=1e-12,
            atol=1e-12,
        )
        shifted = np.exp(x - np.broadcast_to(m, shape))
        total = shifted.sum(axis=axis, keepdims=True) if axis is not None else shifted.sum()
        inv = 1.0 / total
        # The seed arrives in the reduced shape, which is what the kernel's
        # index mapping expects; it is broadcast here only for the comparison.
        seed = rng.standard_normal(m.shape)
        parity(
            _lib.exp_shift_scale(seed, x, m, inv, geo, geo, geo),
            np.broadcast_to(seed, shape) * shifted * np.broadcast_to(inv, shape),
            rtol=1e-10,
            atol=1e-12,
        )


def _mid(shape, axis):
    if axis is None:
        return ()
    a = axis if axis >= 0 else axis + len(shape)
    return shape[:a] + (1,) + shape[a + 1:]


def _geo(shape, axis):
    if axis is None:
        return (1, 1, 1)
    a = axis if axis >= 0 else axis + len(shape)
    return (
        int(np.prod(shape[a + 1:])) if shape[a + 1:] else 1,
        shape[a],
        int(np.prod(shape[:a])) if shape[:a] else 1,
    )


def test_accum_first_does_not_alias_its_input():
    src = np.arange(4.0)
    dst = _lib.accum_first(src)
    src[0] = 99.0
    np.testing.assert_array_equal(dst, [0.0, 1.0, 2.0, 3.0])


def test_accum_add_sums_contributions():
    dst = _lib.accum_first(np.ones(3))
    _lib.accum_add(dst, np.full(3, 2.0))
    np.testing.assert_allclose(dst, [3.0, 3.0, 3.0], rtol=0, atol=0)
