"""A NumPy-shaped namespace that dispatches to the tape.

Mirrors `autograd.numpy`: functions take `Tensor` arguments when any argument
is a tensor and fall through to real NumPy otherwise.

    import mojo_autograd as ma
    import mojo_autograd.npy as anp

    ma.grad(lambda w, x: anp.sum(anp.log(anp.matmul(x, w))), 0)(w, x)
"""

from __future__ import annotations

import numpy as _np

from .tensor import BINARY_OPS, UNARY_OPS, Tensor, logsumexp, matmul

# numpy 2 removed these three. They are spelled here the standard stable way;
# the VJP op codes 23-25 need only the saved input or output, so the exact
# forward spelling does not change the gradient.
_CUSTOM_FORWARD = {
    "sigmoid": lambda x: 1.0 / (1.0 + _np.exp(-x)),
    "softplus": lambda x: _np.log1p(_np.exp(-_np.abs(x))) + _np.maximum(x, 0.0),
    "relu": lambda x: _np.maximum(x, 0.0),
}


def _forward(name):
    fn = _CUSTOM_FORWARD.get(name) or getattr(_np, name, None)
    if fn is None:  # pragma: no cover - guards an op added without a forward
        raise AttributeError(f"numpy.{name} is not available in this numpy")
    return fn


_UNARY = {
    name: (lambda t, _f=_forward(name), _op=op: t._unary(_op, _f))
    for name, op in UNARY_OPS.items()
}

_BINARY = {
    name: (lambda a, b, _f=_forward(name), _op=op: a._binary(b, _op, _f))
    for name, op in BINARY_OPS.items()
}


def _no_dot(a, b):
    raise NotImplementedError(
        "mojo_autograd.npy.dot is not implemented; use matmul, whose kernel "
        "already covers the 1-D, 2-D and batched cases"
    )


_TENSOR_FUNCS = {
    "sum": lambda a, **kw: a.sum(**kw),
    "mean": lambda a, **kw: a.mean(**kw),
    "logsumexp": lambda a, **kw: logsumexp(a, **kw),
    "reshape": lambda a, *s, **kw: a.reshape(*s, **kw),
    "transpose": lambda a, *ax: a.transpose(*ax),
    "swapaxes": lambda a, x, y: a.swapaxes(x, y),
    "squeeze": lambda a, axis=None: a.squeeze(axis),
    "expand_dims": lambda a, axis: a.expand_dims(axis),
    "matmul": lambda a, b: matmul(a, b),
    "dot": _no_dot,
}
_TENSOR_FUNCS.update(_UNARY)
_TENSOR_FUNCS.update(_BINARY)


def __getattr__(name):
    impl = _TENSOR_FUNCS.get(name)
    # Looked up lazily: sigmoid, softplus and relu are gradient ops here but
    # numpy 2 no longer exports them.
    npf = getattr(_np, name, None)
    if impl is None and npf is None:
        raise AttributeError(f"module 'mojo_autograd.npy' has no attribute {name!r}")

    def f(*args, **kwargs):
        if not any(isinstance(a, Tensor) for a in args):
            if npf is None:
                raise TypeError(f"numpy.{name} needs a Tensor argument")
            return npf(*args, **kwargs)
        if impl is None:
            raise TypeError(
                f"mojo_autograd has no gradient rule for numpy.{name}; "
                "use the real autograd package for that op"
            )
        return impl(*args, **kwargs)

    f.__name__ = name
    f.__doc__ = None if npf is None else npf.__doc__
    return f
