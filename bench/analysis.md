# mojo-autograd — measured hotspot map (ANALYZE phase)

> Every number below was produced by a run in this session (2026-11-09, Linux,
> Intel Xeon E5-2697 v4, 2 sockets × 18 cores). Nothing is estimated. `src/` and
> `tests/` were not touched; this file is the only addition. Upstream comparison
> uses the importable `autograd` package (numpy 2.5.1, Python 3.13.14); the NumPy
> column is the plain formula each VJP replaces.

## How the numbers were produced

- `pixi run bench` (flock-wrapped, per `pixi.toml`) — the five cases in `bench/bench.py`.
- A throwaway probe script (since deleted) that timed each public entry point
  (`_lib.vjp_unary`, `vjp_binary`, `reduce_sum`/`reduce_max`, `broadcast`,
  `exp_shift_scale`, `accum_first`/`accum_add`, `bmm_batch`/`bmm_rows`, `gemm`,
  and end-to-end `ma.grad`) at the sizes real callers pass, best-of-3/5 per case.
- `cProfile` over the model-gradient loop, and disassembly of the built
  `dist/libmojo-autograd.so` with `gdb` (elementwise kernels) to check whether the
  inner loops vectorise. Both are repo-local inspections, cited per-row.
- Flops/bytes are computed from the shapes: one elementwise pass reads `g`, `x`,
  `y` and writes `dst` = 4 × 8 B = **32 B/elem** (reuse factor folded into the
  flops column); a gemm reads `A` + `B` and writes `C` with `2·m·k·n` flops.
  `memory-bound vs compute-bound` is judged from flops-per-byte: < ~2 → memory,
  > ~10 → compute (docs: a CPU-only kernel below ~2 flop/byte is legitimate).

## Ranked hotspot table

Rows are ordered by measured time of the worst observed case, descending.
`SLOWER THAN UPSTREAM/REFERENCE` rows are kept, per instructions. Time is the
best measured wall clock for the case size shown; “glue vs kernel” splits the
Python/ctypes wrapper cost from the Mojo kernel cost.

| # | function | input size | time | bytes touched | flops | flops/byte | memory- or compute-bound | Python/ctypes glue vs kernel | vs upstream / baseline |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `gemm` (600×300×200) via `_lib.gemm` | m=600, k=300, n=200 | 12.2–20.3 ms | 5.76 MB | 72.0 M | 12.5 | compute | kernel ≈ 12 ms; wrapper ≈ 0.3 ms (`_f64` ×2, `np.zeros`, `_es` stride packing) | **SLOWER** — reference (NumPy BLAS) 4.6–6.3 ms, i.e. 0.31–0.33×. The gap is kernel serialisation: with `MAX_WORKERS=1` the same gemm is 77 ms, 8 workers 13 ms, so it scales, but even fully fanned out it loses 2.6× to BLAS. |
| 2 | `vjp_unary` tanh, n=4 194 304 | 4 M elems | 65–74 ms | 128 MB (64 read + 32 write + 32 read) | 12.6 M | 0.19 | **memory** (transcendental `cosh` dominates, not the stream) | kernel 49–53 ms; wrapper 13–15 ms = 3× `_f64` (asarray) + `np.empty` + ctypes call | **FASTER** — 3-pass NumPy `g/cosh(x)**2`: 104–125 ms (1.6–1.7×); upstream `make_vjp` tanh: 121 ms. The one-pass win is real but capped: upstream already has to compute `cosh` once. |
| 3 | `vjp_unary` sqrt, n=4 194 304 | 4 M elems | 143–159 ms | same 128 MB | 12.6 M | 0.19 | memory, but the cost is `pow(x,-0.5)` (software path, not a hardware sqrt) | kernel ≈ 143 ms; wrapper ≈ 15 ms | **SLOWER** — NumPy sqrt path 31–75 ms (0.47×). Kernel serialises on scalar `pow`; numpy gets the SIMD `vrsqrt`-class path. `sqrt` is the worst elementwise row. |
| 4 | `vjp_unary` sin/cos/tan/log family, n=4 194 304 | 4 M elems | sin 84 ms, tan similar; log 28 ms | 128 MB | 12.6 M | 0.19 | compute (transcendental) | kernel-side; per-op branch outside the loop is correct (each op = one tight loop) | mixed: sin 1.11× faster than NumPy's `g·cos(x)` (93 ms); log **0.98×** (28.0 vs 27.4 ms); exp 1.09–2.2× (22.7 ms vs NumPy's single-pass `g·exp`, 21.5–61 ms depending on how NumPy's exp output is written). No fusion to gain where upstream is already one pass. |
| 5 | end-to-end `ma.grad` model (4096×3 → mean∘log∘sum∘exp∘matmul) | 4096 rows, 3→2 GEMM, ×20 iters in `bench.py` | single 1.24–2.55 ms; ×20 = 20.7 ms | dominated by the 4096×3 matmul + elementwise tail | ≈ 0.2 M (GEMM 2·4096·3·2 = 49 k + elementwise) | <0.2 | memory/overhead | **Tape/glue dominates**: `cProfile` shows `_bmm_call` + `_fanout` (gemm) 5 ms/20 iters, `accum_first` 6 ms/20 iters, `_unary` 3 ms/20 iters. The forward pass (NumPy `exp`/`log`/`sum`/`mean`) is ~0.5 ms of the 1.2 ms. Per call: 3× `bmm_batch`, 7× `accum_first`, 2× `broadcast`, 2× `vjp_unary`, 2× `reduce_sum` (instrumented, see §3). | **SLOWER** — upstream `autograd.grad` 0.92 ms/iter (0.36–0.58×); `bench.py` itself reports 0.58×. At this size the graph/tape overhead swamps the kernels, and the kernels' win is on a 12 k-elem array. |
| 6 | `vjp_binary` multiply, n=4 194 304 | 4 M elems | 28.5 ms | 128 MB | 8.4 M | 0.13 | memory | kernel 28.5 ms (strided indexing `i*sg/sx/sy`) | **SLOWER** — NumPy `g·b` 25.4 ms (0.89×). The strided loads do not vectorise (no contiguous fast path for `sg=sx=sy=1`). |
| 7 | `accum_first` + `accum_add`, n=4 194 304 | 4 M elems | `accum_first` 26.9 ms; `accum_add` 9.2 ms | `first` 64 MB (r+w); `add` 96 MB (r+w+r) | 8.4 M | 0.13 | memory | kernel-only; `accum_first` already copies through a fresh `np.empty` | `accum_first` 1.05× vs `x.copy()` (28.4 ms); `accum_add` 1.01× vs in-place add (9.3 ms). Par with NumPy — this is `beta==0` fused copy; the 2.9× `first`-vs-`add` gap is the extra read of `dst` in `add`. |
| 8 | `reduce_sum` / `reduce_max` (logsumexp shift) | 4096×3 → 4096; 4096×8 → 4096 | 0.048–0.091 ms | 196–524 KB | 12–33 k | 0.06 | memory | `ag_reduce` odometer loop; per-call `_call_reduce` allocs 2× `np.array(int64)` for dims/weights (~1.6 µs each, measured) | `reduce_sum` 4096×3: **1.83×** vs NumPy sum (0.048 vs 0.088 ms); 4096×8: 1.10×; `broadcast` **0.74×** (0.093 vs 0.069 ms — odometer bookkeeping costs more than NumPy's stride-aware copy). |
| 9 | `exp_shift_scale` (logsumexp core) | 4096×3, geo (1,3,4096) | 0.28–0.31 ms | 196 KB | 49 k | 0.25 | memory | kernel odometer + `exp` per element | **SLOWER** — NumPy `exp(x-m)` 0.17 ms (0.60×). Per-element triple-odometer bookkeeping is the cost; at 4096×3 the overhead dwarfs the data. |
| 10 | small-`gemm` (64×64×64, 256×256×256) | 64³, 256³ | 1.72 ms, 8.6–11.7 ms | 0.2–3.1 MB | 0.52 M, 33.5 M | 2.7, 10.7 | compute (but tiny) | kernel serial below `THREAD_MIN_WORK`=131 072… measured: 64³ work = 262 144 > threshold → **fans out** via `ThreadPoolExecutor` (8 workers, each a fresh pool per call) and still loses | **SLOWER** — NumPy 0.031 / 0.23 ms (0.02–0.06×, 17–56× slower). This is the `bench.py` "serial" row, but the fan-out threshold is already crossed at 64³, so the loss is thread-pool + kernel serialisation, not serial execution. |
| 11 | per-call overhead, small n (1, 64, 1024) | 1–1024 elems | 9.5–9.8 µs (n≤64), 23.7 µs (n=1024) | — | — | — | overhead | pure glue: 3× `_f64` + `np.empty` + ctypes call; measured per-call ≈ 9.5 µs with a zero-length kernel | **SLOWER** — NumPy 1.9–15.8 µs (0.19–0.66×). Crossover is n ≈ 4096 (95 µs both) / n=65 536 (0.89 vs 1.02 ms); below ~4 k elements the port loses on glue alone. |

## What the numbers say about the kernels (disassembly)

`gdb disas` on `dist/libmojo-autograd.so`, per symbol:

| symbol | `ymm`/`zmm` (wide-vector) instructions | `xmm` (scalar-SSE) instructions |
|---|---|---|
| `ag_vjp_unary` | **0** | 339 |
| `ag_vjp_binary` | **0** | 338 |
| `ag_bmm_batch` | **0** | 20 |
| `ag_bmm_rows` | **0** | 20 |
| `ag_reduce` | **0** | 17 |
| `ag_exp_shift_scale` | **0** | 41 |
| `ag_accum` | **0** | 7 |

Every inner loop compiles to scalar `vmovsd`/`vmulsd`/`vdivsd` (e.g. `ag_vjp_unary`
op-0 body: `vmovsd (%r14,%rax,8)` / `vmulsd` / `vmovsd (%rbx,%rax,8)` in a 6-instruction
loop). **No kernel vectorises.** That single fact explains rows 1, 3, 6, 9 and 10:
the port is an interpreted-shape fast scalar loop competing against NumPy's
SIMD/BLAS loops. The wrapper does not help it either: `vjp_unary` measured
13–15 ms of glue on a 65 ms call (20–23% — the 49 ms kernel-only direct ctypes
call vs the 62–66 ms wrapped call), so the glue is a real secondary cost at n=4M
and the *primary* cost below n≈4096 (row 11).

## Flagged patterns (§4 of the brief)

1. **No Python loop re-enters ctypes per element.** The only `for` loops in
   `python/mojo_autograd/*.py` are over *axes* (`npy.py:78` over args, `_lib.py:109`
   over dims, `_lib.py:263` over fan-out chunks) — never over elements. All
   element work is inside the Mojo kernels. Good.
2. **Per-call allocations/copies that matter:**
   - `_lib.vjp_unary`: 3× `_f64` (asarray, cheap but 3 calls) + `np.empty(n)` per call.
     At n=4M the wrapper is 13–15 ms of the 62–66 ms; at n≤1024 the wrapper is the
     *whole* cost (9.5 µs/call vs a zero-length kernel). `_f64` is called even when
     the input is already C-contiguous float64 (asarray no-ops but still 3 calls).
   - `_lib._call_reduce`: 2× `np.array(dims/weights, int64)` per call (~1.6 µs each,
     measured) plus `np.zeros`/`np.full` for `dst` in every reduce/broadcast call.
   - `Tensor._accum` → `accum_first`: allocates a fresh 4M-element output per
     *first* contribution (7× per model backward, instrumented); `accum_add` then
     reads+writes it in place.
   - `matmul`: `np.empty((nb, m, n))` + `swapaxes`/views per VJP; `bmm_batch` is called
     3× per model backward with small (4096×3×2) shapes, each under the thread
     threshold → serial kernel + Python fan-out check per call.
   - `ThreadPoolExecutor` is created fresh per `gemm`/`bmm` call above threshold
     (60 `ex.map` spawns in the 20-iter profile) — sub-millisecond at 600×300×200
     but visible at 64³ where the 1.72 ms is mostly pool + serial kernel.
3. **Threading threshold mislabelled in `bench.py`:** the "serial" 64³ case
   (work = 262 144) is *above* `THREAD_MIN_WORK` (131 072), so it fans out over 8
   workers. Scaling check for 600×300×200: 1 worker 77 ms → 2 = 37.6 → 4 = 21.9 →
   8 = 13.2 ms — it scales, but the ceiling is the serial scalar kernel (see
   SIMD section), not the pool.

## Recommended action per hotspot (§5 of the brief)

- **Row 3 (`sqrt`, `pow(x,-0.5)`), row 4 (sin/tan/log)**: **simd** — vectorise the
  transcendental kernels; `pow(x,-0.5)` must become a SIMD sqrt+rcp, `cosh`/`sin`/`tan`
  the vendor SIMD paths NumPy uses. Flops/byte 0.19 says memory-bound on streams, but
  measured time is compute — the ceiling is the scalar `libm` call per element.
- **Row 1 & 10 (gemm)**: **simd** (inner loop → FMA blocks, `simd_width_of` /
  hard-coded `W=4` from `MOJO_NOTES.md` §3) and only then **parallel** (the
  `max.algorithm.parallelize` path from `MOJO_NOTES.md` §4 instead of the Python
  `ThreadPoolExecutor` fan-out). Flops/byte 10–43 says comfortably compute-bound;
  6.1 GF/s measured vs ~tens of GF/s for dgemm on AVX2. GPU is *not* justified here:
  the 600×300×200 case is 72 Mflop — launch + transfer would dominate; the wins are
  SIMD width and removing per-call pool creation.
- **Row 6 (`vjp_binary` strided)**: **simd** with a contiguous fast path when all
  strides are 1 and `sy=1`, falling back to the strided loop only for true
  broadcasts (the 0.89× is stride-induced scalarisation).
- **Row 9 (`exp_shift_scale`), row 8 (`reduce` odometer)**: **unroll** — replace the
  per-element triple-odometer update with block-unrolled indexing (unroll factor 4/8
  on the trailing dimension), and precompute the `_call_reduce` dims/weights arrays
  in the caller, removing two per-call `np.array` allocations.
- **Row 5 (model end-to-end, tape-bound)**: **none** of the kernel phases fixes it;
  the loss is Python tape bookkeeping (`accum_first` 7×/backward, `_bmm_call` 3×/
  backward, `broadcast` 2×/backward at 4096×3 where the GEMM is 49 kflop). Next
  phase there would be glue reduction (fuse `accum_first` into the VJP that first
  writes a node; cache `_f64` no-ops) — listed as *none* for the kernel roadmap
  because no SIMD/parallel/GPU kernel change helps a tape-bound workload.
- **GPU verdicts**: none recommended. Every elementwise kernel is 0.06–0.25
  flop/byte (memory-bound; at 4M elems ≈ 128 MB/call a GPU would be
  bandwidth-capped and launch-bound), and the gemm is ≤ 2.1 Gflop at the largest
  size benchmarked (1024³: 354 ms here, 22 ms BLAS) — the documented
  "~2 flop/byte or don't bother" rule from `MOJO_NOTES.md` §5 plus the observed
  sizes keep this CPU-side. The 1024³ row is the only one where parallel CPU
  (8 workers) is already the right shape and SIMD is the missing step, not a device.

## Correctness note

All timed cases verify against the NumPy reference before timing (inherited from
`bench/bench.py`); the probe re-checked `vjp_unary`/`vjp_binary`/`gemm` vs NumPy
with `np.allclose` (rtol 1e-9–1e-12), plus upstream `autograd.make_vjp` tanh
(121 ms vs kernel 171 ms in one cold run — the loss is `cosh` duplication, not the
wrapper: kernel-only is 49–53 ms warm).
