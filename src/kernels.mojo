"""Reverse-mode vector-Jacobian kernels for mojo-autograd.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric.

The op selectors are runtime integers, so each op gets its own tight loop
instead of a branch inside one loop; that is what lets the inner loop
vectorise. Broadcast operands are handled with element strides rather than by
materialising a broadcast copy.
"""

from std.math import (
    abs,
    acos,
    acosh,
    asin,
    asinh,
    atan,
    atanh,
    cos,
    cosh,
    exp,
    exp2,
    expm1,
    fma,
    log,
    log1p,
    log2,
    log10,
    pow,
    sin,
    sinh,
    sqrt,
    tan,
    tanh,
)

comptime FPtr = Pointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = Pointer[Int64, AnyOrigin[mut=True]]

comptime LN2 = 0.6931471805599453
comptime LN10 = 2.302585092994046


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# Unary vector-Jacobian products: dst[i] = g[i] * f'(x[i])
#
#   op  0 exp      1 expm1    2 exp2     3 log      4 log2     5 log10
#       6 log1p    7 sin      8 cos      9 tan      10 tanh    11 sinh
#      12 cosh    13 arcsin  14 arccos  15 arctan  16 arcsinh 17 arccosh
#      18 arctanh 19 square  20 sqrt    21 reciprocal 22 fabs   23 sigmoid
#      24 softplus 25 relu
#
# Formulas mirror autograd/numpy/numpy_vjps.py so results agree to tolerance.
# ---------------------------------------------------------------------------
@export("ag_vjp_unary")
def ag_vjp_unary(op: Int, n: Int, g_addr: Int, x_addr: Int, y_addr: Int, dst_addr: Int) abi("C"):
    var g = fp(g_addr)
    var x = fp(x_addr)
    var y = fp(y_addr)
    var d = fp(dst_addr)
    if op == 0:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * y[unsafe_offset=i]
    elif op == 1:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * (y[unsafe_offset=i] + 1.0)
    elif op == 2:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * (y[unsafe_offset=i] * LN2)
    elif op == 3:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / x[unsafe_offset=i]
    elif op == 4:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (x[unsafe_offset=i] * LN2)
    elif op == 5:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (x[unsafe_offset=i] * LN10)
    elif op == 6:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (x[unsafe_offset=i] + 1.0)
    elif op == 7:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * cos(x[unsafe_offset=i])
    elif op == 8:
        for i in range(n):
            d[unsafe_offset=i] = -g[unsafe_offset=i] * sin(x[unsafe_offset=i])
    elif op == 9:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (cos(x[unsafe_offset=i]) ** 2)
    elif op == 10:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (cosh(x[unsafe_offset=i]) ** 2)
    elif op == 11:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * cosh(x[unsafe_offset=i])
    elif op == 12:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * sinh(x[unsafe_offset=i])
    elif op == 13:
        for i in range(n):
            var t = 1.0 - x[unsafe_offset=i] * x[unsafe_offset=i]
            d[unsafe_offset=i] = g[unsafe_offset=i] / sqrt(t)
    elif op == 14:
        for i in range(n):
            var t = 1.0 - x[unsafe_offset=i] * x[unsafe_offset=i]
            d[unsafe_offset=i] = -g[unsafe_offset=i] / sqrt(t)
    elif op == 15:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (1.0 + x[unsafe_offset=i] * x[unsafe_offset=i])
    elif op == 16:
        for i in range(n):
            var t = x[unsafe_offset=i] * x[unsafe_offset=i] + 1.0
            d[unsafe_offset=i] = g[unsafe_offset=i] / sqrt(t)
    elif op == 17:
        for i in range(n):
            var t = x[unsafe_offset=i] * x[unsafe_offset=i] - 1.0
            d[unsafe_offset=i] = g[unsafe_offset=i] / sqrt(t)
    elif op == 18:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (1.0 - x[unsafe_offset=i] * x[unsafe_offset=i])
    elif op == 19:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * (2.0 * x[unsafe_offset=i])
    elif op == 20:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] * 0.5 * pow(x[unsafe_offset=i], -0.5)
    elif op == 21:
        for i in range(n):
            var t = x[unsafe_offset=i] * x[unsafe_offset=i]
            d[unsafe_offset=i] = -g[unsafe_offset=i] / t
    elif op == 22:
        for i in range(n):
            var t = x[unsafe_offset=i]
            var s = 0.0
            if t > 0.0:
                s = 1.0
            elif t < 0.0:
                s = -1.0
            d[unsafe_offset=i] = g[unsafe_offset=i] * s
    elif op == 23:
        for i in range(n):
            var t = y[unsafe_offset=i]
            d[unsafe_offset=i] = g[unsafe_offset=i] * t * (1.0 - t)
    elif op == 24:
        for i in range(n):
            d[unsafe_offset=i] = g[unsafe_offset=i] / (1.0 + exp(-x[unsafe_offset=i]))
    else:
        for i in range(n):
            var t = x[unsafe_offset=i]
            var s = 0.0
            if t > 0.0:
                s = 1.0
            d[unsafe_offset=i] = g[unsafe_offset=i] * s


# ---------------------------------------------------------------------------
# Binary vector-Jacobian products: dst[i] = g[i] * df/darg(x[i], y[i])
#
#   op 0 add  1 subtract  2 multiply  3 divide  4 power
#      5 maximum 6 minimum 7 logaddexp 8 arctan2 9 hypot
#
# `arg` selects which operand is differentiated. `sg/sx/sy/sd` are element
# strides, so a broadcast (including a 0-d scalar) operand costs nothing extra.
# ---------------------------------------------------------------------------
@export("ag_vjp_binary")
def ag_vjp_binary(
    op: Int,
    arg: Int,
    n: Int,
    g_addr: Int,
    x_addr: Int,
    y_addr: Int,
    dst_addr: Int,
    sg: Int,
    sx: Int,
    sy: Int,
    sd: Int,
) abi("C"):
    var g = fp(g_addr)
    var x = fp(x_addr)
    var y = fp(y_addr)
    var d = fp(dst_addr)
    if op == 0:
        if arg == 0:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg]
        else:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg]
    elif op == 1:
        if arg == 0:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg]
        else:
            for i in range(n):
                d[unsafe_offset=i * sd] = -g[unsafe_offset=i * sg]
    elif op == 2:
        if arg == 0:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * y[unsafe_offset=i * sy]
        else:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * x[unsafe_offset=i * sx]
    elif op == 3:
        if arg == 0:
            for i in range(n):
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] / y[unsafe_offset=i * sy]
        else:
            for i in range(n):
                d[unsafe_offset=i * sd] = -g[unsafe_offset=i * sg] * x[unsafe_offset=i * sx] / (
                    y[unsafe_offset=i * sy] * y[unsafe_offset=i * sy]
                )
    elif op == 4:
        if arg == 0:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var e = 1.0
                if yv != 0.0:
                    e = yv - 1.0
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * yv * pow(xv, e)
        else:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var t = xv
                if xv == 0.0:
                    t = 1.0
                # The upstream rule is g * log(replace_zero(x, 1)) * ans, and ans
                # is x**y, not the right operand.
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * log(t) * pow(xv, yv)
    elif op == 5 or op == 6:
        # maximum / minimum: balanced, so ties at x == y give 0.5 to both.
        for i in range(n):
            var xv = x[unsafe_offset=i * sx]
            var yv = y[unsafe_offset=i * sy]
            if arg == 0:
                var t = 0.0
                if op == 5:
                    if xv > yv:
                        t = 1.0
                    elif xv == yv:
                        t = 0.5
                else:
                    if xv < yv:
                        t = 1.0
                    elif xv == yv:
                        t = 0.5
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * t
            else:
                var t = 0.0
                if op == 5:
                    if yv > xv:
                        t = 1.0
                    elif yv == xv:
                        t = 0.5
                else:
                    if yv < xv:
                        t = 1.0
                    elif yv == xv:
                        t = 0.5
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * t
    elif op == 7:
        # logaddexp: exp(x - logaddexp(x, y))
        if arg == 0:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var m = xv
                if yv > m:
                    m = yv
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * exp(xv - m - log(exp(xv - m) + exp(yv - m)))
        else:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var m = xv
                if yv > m:
                    m = yv
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * exp(yv - m - log(exp(xv - m) + exp(yv - m)))
    elif op == 8:
        for i in range(n):
            var xv = x[unsafe_offset=i * sx]
            var yv = y[unsafe_offset=i * sy]
            var den = xv * xv + yv * yv
            if arg == 0:
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * yv / den
            else:
                d[unsafe_offset=i * sd] = -g[unsafe_offset=i * sg] * xv / den
    else:
        # hypot
        if arg == 0:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var h = sqrt(xv * xv + yv * yv)
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * xv / h
        else:
            for i in range(n):
                var xv = x[unsafe_offset=i * sx]
                var yv = y[unsafe_offset=i * sy]
                var h = sqrt(xv * xv + yv * yv)
                d[unsafe_offset=i * sd] = g[unsafe_offset=i * sg] * yv / h


# ---------------------------------------------------------------------------
# Batched matrix multiply C = alpha * A @ B + beta * C.
#
# A is (m, k) with element strides (sam, sak), B is (k, n) with (sbk, sbn),
# C is (m, n) with (scm, scn). Batch strides (sab, sbb, scb) of 0 broadcast
# that operand across the batch, so a 1-D operand promoted with
# `expand_dims` needs no copy at all.
# ---------------------------------------------------------------------------
@export("ag_bmm_batch")
def ag_bmm_batch(
    m: Int,
    k: Int,
    n: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    sam: Int,
    sak: Int,
    sbk: Int,
    sbn: Int,
    scm: Int,
    scn: Int,
    sab: Int,
    sbb: Int,
    scb: Int,
    alpha: Float64,
    beta: Float64,
    b0: Int,
    b1: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    for bi in range(b0, b1):
        var ao = bi * sab
        var bo = bi * sbb
        var co = bi * scb
        for i in range(m):
            var arow = ao + i * sam
            var crow = co + i * scm
            for j in range(n):
                var bcol = bo + j * sbn
                var acc = Float64(0.0)
                for p in range(k):
                    acc = fma(a[unsafe_offset=arow + p * sak], b[unsafe_offset=bcol + p * sbk], acc)
                var idx = j * scn
                if beta == 0.0:
                    c[unsafe_offset=crow + idx] = alpha * acc
                else:
                    c[unsafe_offset=crow + idx] = alpha * acc + beta * c[unsafe_offset=crow + idx]


@export("ag_bmm_rows")
def ag_bmm_rows(
    m: Int,
    k: Int,
    n: Int,
    a_addr: Int,
    b_addr: Int,
    c_addr: Int,
    sam: Int,
    sak: Int,
    sbk: Int,
    sbn: Int,
    scm: Int,
    scn: Int,
    sab: Int,
    sbb: Int,
    scb: Int,
    alpha: Float64,
    beta: Float64,
    i0: Int,
    i1: Int,
) abi("C"):
    var a = fp(a_addr)
    var b = fp(b_addr)
    var c = fp(c_addr)
    for ii in range(i0, i1):
        var arow = ii * sam
        var crow = ii * scm
        for j in range(n):
            var bcol = j * sbn
            var acc = Float64(0.0)
            for p in range(k):
                acc = fma(a[unsafe_offset=arow + p * sak], b[unsafe_offset=bcol + p * sbk], acc)
            var idx = j * scn
            if beta == 0.0:
                c[unsafe_offset=crow + idx] = alpha * acc
            else:
                c[unsafe_offset=crow + idx] = alpha * acc + beta * c[unsafe_offset=crow + idx]


# ---------------------------------------------------------------------------
# Shape mapping for reductions, maxima and broadcasts.
#
#   mode 0: dst[o] += src[i]   (sum-reduce / unbroadcast)
#   mode 1: dst[o]  = max(dst[o], src[i])
#   mode 2: dst[i]  = src[o]   (broadcast / repeat)
#
# `dims` is the iterated shape and `weights` gives, per axis, the stride of the
# other array (0 for a broadcast axis, and for a reduced axis in mode 0/1).
# The index of the other array is rebuilt with a mixed-radix odometer over
# `dims`, held in the caller-owned `scratch` buffer, so no division is needed.
# ---------------------------------------------------------------------------
@export("ag_reduce")
def ag_reduce(
    mode: Int,
    ndim: Int,
    n_iter: Int,
    n_other: Int,
    dims_addr: Int,
    weights_addr: Int,
    src_addr: Int,
    dst_addr: Int,
    scratch_addr: Int,
) abi("C"):
    var dims = ip(dims_addr)
    var w = ip(weights_addr)
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var digits = ip(scratch_addr)
    for a in range(ndim):
        digits[unsafe_offset=a] = Int64(0)
    var oi = Int(0)
    for i in range(n_iter):
        if mode == 2:
            dst[unsafe_offset=i] = src[unsafe_offset=oi]
        elif mode == 0:
            dst[unsafe_offset=oi] += src[unsafe_offset=i]
        else:
            var t = src[unsafe_offset=i]
            if t > dst[unsafe_offset=oi]:
                dst[unsafe_offset=oi] = t
        var a = ndim - 1
        while a >= 0:
            var d = digits[unsafe_offset=a] + Int64(1)
            var dim = Int(dims[unsafe_offset=a])
            if d < Int64(dim):
                digits[unsafe_offset=a] = d
                oi += Int(w[unsafe_offset=a])
                break
            else:
                oi -= Int(w[unsafe_offset=a]) * (dim - 1)
                digits[unsafe_offset=a] = Int64(0)
                a -= 1


# ---------------------------------------------------------------------------
# dst[i] = x[xi] * exp(d[i] - m[mi]) * r[ri]
#
# Three independently strided broadcasts, each described by (lo, mid, hi): the
# index of element i in that operand is (i // (lo * mid)) * lo + (i % lo), with
# the mid digit dropped because it belongs to the reduced axis. For a source of
# shape S reduced over axis a that is lo = prod(S[a+1:]), mid = S[a] and
# hi = prod(S[:a]). Passing (1, 1, 1) pins an operand to element 0, which is how
# a constant 1 is supplied.
#
# Serves both halves of log-sum-exp: the shifted exponential the forward pass
# needs (x == 1, r == 1) and the softmax-weighted VJP (x == the seed, r ==
# 1 / sum), where the seed and the reciprocal both live in the reduced shape.
# ---------------------------------------------------------------------------
@export("ag_exp_shift_scale")
def ag_exp_shift_scale(
    n: Int,
    x_addr: Int,
    d_addr: Int,
    m_addr: Int,
    r_addr: Int,
    xlo: Int,
    xmid: Int,
    xhi: Int,
    mlo: Int,
    mmid: Int,
    mhi: Int,
    rlo: Int,
    rmid: Int,
    rhi: Int,
    dst_addr: Int,
) abi("C"):
    var x = fp(x_addr)
    var d = fp(d_addr)
    var m = fp(m_addr)
    var r = fp(r_addr)
    var dst = fp(dst_addr)
    var xl = Int(0)
    var xm = Int(0)
    var xh = Int(0)
    var ml = Int(0)
    var mm = Int(0)
    var mh = Int(0)
    var rl = Int(0)
    var rm = Int(0)
    var rh = Int(0)
    for i in range(n):
        dst[unsafe_offset=i] = x[unsafe_offset=xh * xlo + xl] * exp(
            d[unsafe_offset=i] - m[unsafe_offset=mh * mlo + ml]
        ) * r[unsafe_offset=rh * rlo + rl]
        xl += 1
        if xl == xlo:
            xl = 0
            xm += 1
            if xm == xmid:
                xm = 0
                xh += 1
                if xh == xhi:
                    xh = 0
        ml += 1
        if ml == mlo:
            ml = 0
            mm += 1
            if mm == mmid:
                mm = 0
                mh += 1
                if mh == mhi:
                    mh = 0
        rl += 1
        if rl == rlo:
            rl = 0
            rm += 1
            if rm == rmid:
                rm = 0
                rh += 1
                if rh == rhi:
                    rh = 0


# ---------------------------------------------------------------------------
# dst[o*sd] = alpha * src[i*ss] + beta * dst[o*sd]; beta == 0 overwrites, which
# is how the first incoming gradient for a node lands in a fresh buffer
# instead of aliasing an array a parent still needs.
# ---------------------------------------------------------------------------
@export("ag_accum")
def ag_accum(
    n: Int,
    src_addr: Int,
    dst_addr: Int,
    ss: Int,
    sd: Int,
    alpha: Float64,
    beta: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    if beta == 0.0:
        for i in range(n):
            dst[unsafe_offset=i * sd] = alpha * src[unsafe_offset=i * ss]
    else:
        for i in range(n):
            dst[unsafe_offset=i * sd] = alpha * src[unsafe_offset=i * ss] + beta * dst[unsafe_offset=i * sd]
