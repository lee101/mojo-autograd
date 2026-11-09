# VERIFY phase — what reproduced, what did not, what changed

Date: 2026-11-09 session. Machine: same host as `bench/analysis.md` (Linux,
72 cores, high load — load average ~55 during bench runs). Gate:
`pixi run build && pixi run test` = 70 passed. `pixi run bench` run 10+
times via `flock` wrapper per `pixi.toml`.

## Reproduced (numbers now in README)

All values below are medians of repeated `pixi run bench` runs in this
session (best-of-3 per case, as `bench.py` does). Ranges include the
load-induced spread.

| case | README (old) | measured this session | verdict |
| --- | --- | --- | --- |
| tanh vjp n=4194304 | 444.53 / 107.68 ms, 4.13x | ref 94–114 ms, mojo 58–64 ms, **~1.6–1.8x** | **Did not reproduce.** README corrected to 102.07 / 58.83 ms, 1.74x. |
| exp vjp n=4194304 | 24.85 / 22.75 ms, 1.09x faster | ref 16.5–21 ms, mojo 17.5–24 ms, **~0.88–1.0x** | **Did not reproduce (sign flipped).** README corrected to 17.57 / 18.97 ms, 0.93x (loss). |
| gemm 600x300x200 | 109.32 / 42.45 ms, 2.58x faster | ref 1–30 ms, mojo 13–36 ms, **~0.1–1.0x** | **Did not reproduce (sign flipped).** README corrected to 5.67 / 14.94 ms, 0.38x (loss). |
| gemm 64x64x64 | 0.04 / 5.92 ms, 0.01x | ref 0.03–0.04 ms, mojo 1.1–1.6 ms, **~0.02x** | Within ~20% on ratio; README corrected to measured 0.03 / 1.36 ms, 0.02x. |
| model grad x20 | 19.85 / 37.21 ms, 0.53x | ref 10.5–14 ms, mojo 17.8–25 ms, **~0.55–0.65x** | Ratio reproduces; README corrected to 11.76 / 19.93 ms, 0.59x. |

The gemm reference (OpenBLAS `a @ b`) varies >10x run-to-run on this host
(1–30 ms for the identical call, best-of-3). The old table's 109 ms ref
figure never occurred in ~13 runs; the 42 ms mojo figure occurred once.
Treat gemm as "slower, order-of-magnitude" — the README now says so.

## Fairness checks

- **Same input both sides:** yes. All five cases allocate one input set
  (seeded RNG in `bench.py`) and pass the same arrays to both sides.
- **No skipped work:**
  - tanh: reference `g / np.cosh(x)**2` is literally upstream autograd's
    `defvjp(anp.tanh, lambda ans, x: lambda g: g / anp.cosh(x)**2)` (verified
    against the installed `autograd` source). Mojo `op==10` computes the same
    from `x` per call. `y` is precomputed once outside timing on both sides.
  - exp: reference `g * y` is upstream's `defvjp(anp.exp, ... ans * g)`. Same.
  - gemm / small gemm: full `a @ b` vs full `_lib.gemm(a, b)` (which
    additionally zeros `C` each call — counted against the port, fair).
  - model: both `mine`/`theirs` in `bench.py` ignore `b` identically, gate on
    `np.allclose(rtol=1e-9)` before timing, and run the same 20 iterations.
    No side skips the backward pass.
- **Numerical gate:** every case asserts agreement before timing (inherited
  from `bench.py`). No case measures work that was not verified.

## README claims that did not reproduce

1. **Small-gemm "serial" explanation (wrong).** README said the 64³ case
   stays serial below the threading threshold. Measured: work = 64·64·64 =
   262144 ≥ `THREAD_MIN_WORK` = 131072 (`_lib.py:27`), and `_fanout`
   (`_lib.py:261`) requires `nitems > 1 and work >= THRESHOLD` — rows fan out
   over 8 workers. The loss is thread-pool + scalar-kernel cost, not serial
   execution (consistent with `bench/analysis.md` row 10). README corrected.
2. **Complex/int "rejected rather than silently narrowed" (wrong).** Probe
   in this session: `grad` on `[1+2j]` returns `[2.]` with a NumPy
   `ComplexWarning` (real part only); int and float32 inputs are silently
   cast to float64. Root cause: `tensor._contig` / `_lib._f64` use
   `np.asarray(x, dtype=np.float64)`. README corrected. No code change (rule:
   fix code only with a failing test; this is a docs bug, and the casting
   behaviour is arguably a feature — see below).
3. **Install section (stale).** README says toolchain
   `mojo 1.2.0.dev2026092605` via `source .../activate.sh` +
   `bash build/build.sh`. `pixi.toml` pins `==1.2.0.dev2026092905` and the
   task instructions use `pixi run build`. The `activate.sh` path was not
   used in this session. README's install block left as-is except noting the
   Pixi pin; `pixi run build` is the verified path.

## Test parity (`git log -p -- tests/`)

`tests/` was added in a single commit (`6d82ce3`) and never modified since —
**no removed assertions, no loosened tolerances** to restore. Current
tolerances: `RTOL=1e-9`/`ATOL=1e-11` default `parity` helper; `rtol=1e-8` for
binary/matmul/model (FMA justification in the header comment and README
Numerical notes); `rtol=1e-12..1e-10` for exact-index/copy work; `rtol=1e-5`
only for central-difference cross-check; exact `assert_array_equal` for relu
indicator, add/subtract, broadcast, accum. All 70 tests pass.

## Changes made

- `README.md`: benchmark table replaced with measured values (above); "two
  losses" → "four losses"; small-gemm serial explanation corrected;
  complex/int rejection claim corrected. Loss rows kept longer, not reworded
  favourably.
- `bench/verify.md`: this file.
- No source/test code changes. No failing test was found; the complex-cast
  behaviour is a docs discrepancy, not a test failure (tests assert with
  `importorskip("autograd")` and run green). If the project wants the
  documented "rejected rather than narrowed" contract, the fix is a short
  type check in `_contig`/`_f64` plus a test — not done here per the
  code-change rule.

## Could not reproduce

- The old tanh 4.13x / 444 ms figure and gemm 2.58x / 109 ms figure: no run
  in this session approached them. They may have been produced on a quieter
  host or a different NumPy/BLAS build; on this host they do not reproduce.
- BLAS gemm timing under load is not a stable number; only the sign
  (slower than BLAS) and rough magnitude (~3–13x slower at median) reproduce.
