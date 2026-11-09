# VERIFY phase — what reproduced, what did not, what changed

Date: 2026-11-09 session (independent rerun). Machine: same host as
`bench/analysis.md` (Linux, 72 cores, high load — load average ~54–62 during
bench runs, measured via `uptime`). Python 3.13.14, NumPy 2.5.1, autograd 1.9.1,
Mojo 1.2.0.dev2026092905 (pixi-pinned). Gate:
`pixi run build && pixi run test` = 70 passed.
`pixi run bench` run 11 times total (1 initial + 10-loop) via the `flock`
wrapper in `pixi.toml`. This file replaces the prior verifier's notes;
all numbers below were measured in this session.

## Reproduced (numbers now in README)

All values below are medians of 11 `pixi run bench` runs in this session
(best-of-3 per case, as `bench.py` does, 5 for the small gemm). Ranges are
min–max across all 11 runs. Load-induced spread included.

| case | README (old, from prior verifier) | measured this session (median) | range this session | verdict |
| --- | --- | --- | --- | --- |
| tanh vjp n=4194304 | 102.07 / 58.83 ms, 1.74x | ref 109.08 ms, mojo 64.71 ms, **1.69x** | ref 94.85–116.06 ms, mojo 58.83–68.97 ms, ratio 1.57–1.80x | **Reproduced** (within 20% on every figure). README updated to medians. |
| exp vjp n=4194304 | 17.57 / 18.97 ms, 0.93x | ref 18.41 ms, mojo 18.87 ms, **0.98x** | ref 16.54–20.90 ms, mojo 17.61–23.49 ms, ratio 0.87–1.10x | **Reproduced** (within 20%; straddles 1.0x). README updated to medians; still listed as slower (0.98x < 1x, not reworded). |
| gemm 600x300x200 | 5.67 / 14.94 ms, 0.38x | ref 4.60 ms, mojo 21.26 ms, **0.22x** | ref 1.17–15.88 ms, mojo 10.60–36.59 ms, ratio 0.04–0.91x | **Within 20% on ref** (4.60 vs 5.67 = -19%); **mojo +42% off, ratio -43% off — corrected.** README updated to 4.60 / 21.26 ms, 0.22x (4.6x slower). Sign (loss) reproduces in every run. |
| gemm 64x64x64 | 0.03 / 1.36 ms, 0.02x | ref 0.03 ms, mojo 1.32 ms, **0.02x** | ref 0.03–0.05 ms, mojo 1.04–1.61 ms, ratio 0.02–0.03x | **Reproduced** (within 20%). README updated to medians. |
| model grad x20 | 11.76 / 19.93 ms, 0.59x | ref 12.19 ms, mojo 20.70 ms, **0.59x** | ref 11.59–13.71 ms, mojo 19.35–23.73 ms, ratio 0.55–0.61x | **Reproduced** (within 20%). README updated to medians. |

The gemm reference (OpenBLAS `a @ b`) varies >10x run-to-run on this host
(1.17–15.88 ms for the identical call, best-of-3, this session). The Mojo side
also varies under load (10.60–36.59 ms). Both prior figure pairs (2.58x at
109/42 ms, then 0.38x at 5.67/14.94 ms) are inside or near that spread on the
reference side, but the Mojo-side median moved +42% this session, so the old
ratio understated the loss. This session's README reports the medians above and
warns the gemm ratio is order-of-magnitude. The sign (slower than BLAS)
reproduced in all 11 runs — no run showed the port faster.

The tanh median moved +7% (ref) / +10% (mojo) vs the prior table — also
within 20%. Exp moved +5% / -0.5% — within 20%. Model moved +4% / +4% —
within 20%. Small gemm moved 0% / -3% — within 20%. Only the large gemm
Mojo-side figure crossed the 20% threshold.

## Fairness checks

- **Same input both sides:** yes. All five cases in `bench/bench.py` allocate
  one input set via a seeded RNG and pass the same arrays to both sides. No
  case re-randomises between sides.
- **No skipped work:**
  - tanh (op 10): reference `g / np.cosh(x)**2` is literally upstream
    autograd's `defvjp(anp.tanh, lambda ans, x: lambda g: g / anp.cosh(x)**2)`
    (verified against the installed `autograd/numpy/numpy_vjps.py` this
    session). Mojo kernel `ag_vjp_unary` op 10 (`src/kernels.mojo:101-103`)
    computes `g / (cosh(x)**2)` from `x` per call. `y` is precomputed once
    outside timing and unused by op 10 — the precomputation is not work the
    timed reference does either, so it does not flatter either side.
  - exp (op 0): reference `g * y` is upstream's
    `defvjp(anp.exp, lambda ans, x: lambda g: ans * g)`. Same `y`. Same.
  - gemm / small gemm: full `a @ b` vs full `_lib.gemm(a, b)` (which allocates
    and zeros `C` each call — counted against the port). Kernel loops run
    `for i in range(n)` / full triple loop with no early exit (`src/kernels.mojo`
    `ag_vjp_unary`, `ag_bmm_rows`).
  - model: both `mine`/`theirs` in `bench.py` ignore `b` identically (same
    dead parameter), gate on `np.allclose(rtol=1e-9)` before timing, and run
    the same 20 iterations of the same `ma.grad`/`autograd.grad`. No side
    skips the backward pass.
  - Each timed lambda is called in full inside `_time` (best-of-3/5); no
    caching, no pre-computed outputs passed in.
- **Numerical gate:** every case asserts agreement before timing (inherited
  from `bench.py`: rtol 1e-12 tanh/exp, 1e-11/1e-12 gemm, 1e-9 model). No case
  measures work that was not verified.

## README claims re-checked

1. **Small-gemm "serial" label (misnomer, kept in README with correction).**
   `bench.py` labels the 64³ case `(serial)`, but 64·64·64 = 262144 ≥
   `THREAD_MIN_WORK` = 131072 (`_lib.py:27`), and `_fanout` (`_lib.py:261`)
   requires `nitems > 1 and work >= THRESHOLD` — rows fan out over 8 workers.
   Verified this session. The README retains the row label verbatim
   (`gemm 64x64x64 (serial)`, matching `bench.py` output) and already notes the
   fan-out — the label is the benchmark's, not a README claim of serial
   execution. The loss is thread-pool + scalar-kernel cost, not serial
   execution (consistent with `bench/analysis.md` row 10).
2. **Complex/int "silently cast to float64" (reproduced).** Probe this
   session: `grad` of `sum(t*t)` on `[1+2j]` returns `[2.]` with
   `ComplexWarning: Casting complex values to real discards the imaginary
   part`; int32 → `[2. 4. 6.]`, float32 → `[2. 4.]`, silently cast to
   float64. Root cause: `tensor._contig` / `_lib._f64` use
   `np.asarray(..., dtype=np.float64)`. The README's current wording ("silently
   cast … The earlier wording claimed rejection; that did not reproduce")
   reproduces exactly.
3. **Install section.** `pixi.toml` pins `mojo ==1.2.0.dev2026092905`; the
   README install block still shows the older `activate.sh` toolchain path
   (`1.2.0.dev2026092605`) alongside `bash build/build.sh`. The verified path
   this session is `pixi run build` (70 passed). Left as-is (not a benchmark
   number; the prior verifier noted it). The README's pixi pin is discoverable
   in `pixi.toml:17`.

## Test parity (`git log -p -- tests/`)

`tests/` was added in a single commit (`6d82ce3`) and **never modified since**:
`git log --all --oneline -- tests/` returns only `6d82ce3`, and
`git diff 6d82ce3..HEAD -- tests/` is empty. The prior verify commit
(`569be83`) touched only `README.md` and `bench/verify.md`. **No removed
assertions, no loosened tolerances to restore.**

Current tolerances (verified this session by grep + reading the full history):
`RTOL=1e-9`/`ATOL=1e-11` default `parity` helper; `rtol=1e-8` for
binary/matmul/model (FMA justification in the header comment and README
Numerical notes); `rtol=1e-12..1e-10` for exact-index/copy work; `rtol=1e-5`
only for central-difference cross-check; exact `assert_array_equal` for relu
indicator, add/subtract, broadcast, accum. All 70 tests pass.

## Changes made

- `README.md`: benchmark table replaced with this session's medians (see
  above); prose ranges updated (`~102/~59 ms` → `~109/~65 ms` tanh;
  `~5/~15 ms` → `~4.6/~21 ms` gemm; BLAS spread `1–30 ms` → `1–16 ms` this
  session; exp `~0.9x` → `~0.98x`). Loss rows kept longer, not reworded
  favourably — every loss row still reads "slower" and the table still states
  "Four of those are losses".
- `bench/verify.md`: this file (replaces prior verifier's file with this
  session's independent measurements).
- No source/test code changes. No failing test was found; 70/70 pass before
  and after.

## Could not reproduce

- The original tanh 4.13x / 444 ms figure and gemm 2.58x / 109 ms figure: no
  run in this session (or the prior verifier's) approached them. They may have
  been produced on a quieter host or a different NumPy/BLAS build; on this
  host they do not reproduce.
- BLAS gemm timing under load is not a stable number; only the sign
  (slower than BLAS, 0.04–0.91x across 11 runs) reproduces. The README now
  says so explicitly.
