# mojo-autograd

`mojo-autograd` is the reverse-mode numeric core of
[autograd](https://github.com/HIPS/autograd) with the vector-Jacobian
inner loops implemented in Mojo. The graph is still built in Python — that is
control flow, and a compiled inner loop does not help it — but every piece of
arithmetic that a gradient actually costs lives in one shared library.

The Python package is `mojo_autograd`, so it installs alongside the real
`autograd` and the tests compare the two directly.

```python
import numpy as np
import mojo_autograd as ma
import mojo_autograd.npy as anp

def loss(w, b, x):
    return anp.mean(anp.log(anp.sum(anp.exp(anp.matmul(x, w) + b), axis=1)))

ma.grad(loss, 0)(w, b, x)
```

`mojo_autograd.npy` mirrors `autograd.numpy`: it takes `Tensor` arguments when
any argument is a tensor and falls through to real NumPy otherwise.

## Covered subset

| area | implemented API |
| --- | --- |
| Entry points | `grad(f, argnum)`, `value_and_grad(f, argnum)`, `Tape`, `Tensor` |
| Unary VJPs | `exp expm1 exp2 log log2 log10 log1p sin cos tan tanh sinh cosh arcsin arccos arctan arcsinh arccosh arctanh square sqrt reciprocal fabs sigmoid softplus relu` |
| Binary VJPs | `add subtract multiply divide power maximum minimum logaddexp arctan2 hypot` |
| Linearity | `matmul` (1-D, 2-D and batched operands, broadcast batches) |
| Reductions | `sum`, `mean`, `logsumexp` (single axis or all) |
| Views | `reshape`, `transpose`, `swapaxes`, `squeeze`, `expand_dims` |
| Kernels | `ag_vjp_unary`, `ag_vjp_binary`, `ag_bmm_batch`, `ag_bmm_rows`, `ag_reduce`, `ag_exp_shift_scale`, `ag_accum` |

Not implemented, and left to the real `autograd`:

- Forward-mode AD (`jvp`, `jacrev`, `make_jvp`, `elementwise_grad`,
  `holomorphic_grad`, `make_vjp`).
- `logsumexp` over a tuple of axes (a single axis or `None` only).
- `matmul` where *both* batch shapes need genuine per-axis broadcasting, e.g.
  `(2, 1, 3, 4) @ (1, 5, 4, 6)`. Equal batch counts and a size-1 batch are
  handled by the kernel's zero batch stride; a genuine per-axis broadcast raises
  `NotImplementedError` rather than computing something wrong.
- Complex numbers, integer inputs, `dot`, `kron`, `einsum`, `tensordot`,
  convolutions, optimizers, `fixed_points`, and every non-CPU device path.
- The numerical contract is C-contiguous `float64` throughout; anything wider
  or complex is rejected rather than silently narrowed.

## How the backward pass works

`Tape` records a node per operation. Each node stores its value plus one VJP
closure per input. `Tape.backward` walks the nodes in reverse creation order —
which is a valid reverse topological order, because a node is only ever built
from values that already exist, so every consumer of a node has a higher index
and is visited first — and calls each closure. A node's value is released as
soon as it has been consumed; leaf gradients are what the caller asked for and
are kept.

The forward pass runs in NumPy, exactly as upstream autograd does. The point of
a reverse-mode AD library is the backward pass, and that is where the win is:
`anp.tanh` upstream is `g / cosh(x) ** 2`, three passes over memory, while
`ag_vjp_unary` is one fused pass with its own tight loop per op.

Shape handling is where the work is:

- `ag_reduce` implements sum, max and broadcast with one mixed-radix odometer
  over a caller-owned scratch buffer, so no division is needed per element. It
  serves the reduction VJPs, the `logsumexp` shift and the unbroadcasting of a
  broadcast operand back to its own shape.
- `ag_bmm_batch` / `ag_bmm_rows` take full element strides for A, B and C plus a
  batch stride per operand, so a transposed operand is a stride swap and a
  broadcast operand is a stride of zero. A 1-D operand becomes a 1-row or
  1-column matrix without a copy, which is what makes the 1-D matmul adjoints
  work. Large gemms are chunked and fanned out over a thread pool from the
  Python shim; small and bandwidth-bound ones stay serial.

## Install

The repository pins its own Mojo toolchain and builds against the shared
`mojo 1.2.0.dev2026092605` environment:

```bash
source /nvme0n1-disk/mojo-toolchain/activate.sh
bash build/build.sh          # -> dist/libmojo-autograd.so
PYTHONPATH=python python -m pytest tests -q
```

`build/build.sh` compiles the single compilation unit `src/kernels.mojo` with
`mojo build --emit shared-lib`. Set `PYTHONPATH=python` when using the package
outside a Pixi task.

## Performance

Best-of-N wall clock, same process, benchmarked against NumPy and against the
real `autograd`. Every case verifies numerical agreement before timing.

| case | reference | mojo-autograd | result |
| --- | ---: | ---: | ---: |
| tanh vjp n=4194304 | 444.53 ms | 107.68 ms | 4.13x faster |
| exp vjp n=4194304 | 24.85 ms | 22.75 ms | 1.09x faster |
| gemm 600x300x200 | 109.32 ms | 42.45 ms | 2.58x faster |
| gemm 64x64x64 (serial) | 0.04 ms | 5.92 ms | 0.01x, 140x slower |
| model gradient x20 | 19.85 ms | 37.21 ms | 0.53x, 1.9x slower |

Two of those are losses and are reported as losses.

- The small gemm is a loss because a 64x64x64 product is 0.5 Mflop: below the
  threading threshold the kernel stays serial, and NumPy's BLAS is tuned for
  exactly that size. The threshold is `mojo_autograd._lib.THREAD_MIN_WORK`.
- The end-to-end model is a loss because at this size the run is dominated by
  Python-level tape bookkeeping, which this port does not make faster, while
  the backward arithmetic it does speed up is a small part of the total. The
  kernels win; the surrounding graph does not.

The two elementwise VJPs differ by a lot for a reason worth stating: upstream's
`tanh` VJP needs `cosh(x)`, a square and a divide, so it streams the array three
times, while the fused kernel streams it once. `exp`'s VJP is already one pass
in NumPy, so there is nothing to win and 1.09x is the honest number.

Reproduce with:

```bash
PYTHONPATH=python python bench/bench.py
```

## Numerical notes

Mojo emits FMA, so a VJP that ends in a multiply-add differs from NumPy's
separate operations in the last bits. The parity tests assert `rtol=1e-9`
against the real `autograd` and `rtol=1e-8` where a gemm is involved; exact
equality is asserted only for pure index and copy work (a broadcast, a
relu/indicator gradient), where the operation really is exact.

## License

MIT
