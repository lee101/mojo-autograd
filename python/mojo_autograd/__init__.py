"""mojo-autograd: the reverse-mode numeric core of autograd, in Mojo.

Public surface mirrors the parts of `autograd` that a compiled inner loop can
actually serve: `grad`, `value_and_grad`, the `Tape`/`Tensor` tape itself and
`mojo_autograd.npy` as a drop-in `autograd.numpy` for the covered ops.
"""

from . import _lib
from .tensor import (
    BINARY_OPS,
    UNARY_OPS,
    Tape,
    Tensor,
    logsumexp,
    matmul,
    value_of,
)

__version__ = "0.1.0"

__all__ = [
    "BINARY_OPS",
    "Tape",
    "Tensor",
    "UNARY_OPS",
    "grad",
    "logsumexp",
    "matmul",
    "value_and_grad",
    "value_of",
]


def _differentiate(f, argnum, want_value):
    def wrapped(*args, **kwargs):
        tape = Tape()
        leaves = [tape.leaf(a) for a in args]
        out = f(*leaves, **kwargs)
        if not isinstance(out, Tensor):
            raise TypeError(
                "the differentiated function must return a Tensor; a constant "
                "result has no gradient path"
            )
        tape.backward(out)
        grads = tape.gradient(out, argnum)
        return (out.value, grads) if want_value else grads

    return wrapped


def grad(f, argnum=0):
    """Reverse-mode gradient of `f` with respect to argument `argnum`.

    Mirrors `autograd.grad`: `f` receives `Tensor` leaves, `argnum` selects an
    argument or a list of arguments, and the seed is a vector of ones.
    """
    return _differentiate(f, argnum, False)


def value_and_grad(f, argnum=0):
    """`f`'s value and its gradient in one pass over the tape."""
    return _differentiate(f, argnum, True)
