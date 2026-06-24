import math
import numpy as np

# numba is an optional accelerator for interp2linear. If it is unavailable the
# pure-NumPy reference path is used instead, so the module never hard-fails.
try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


if _HAVE_NUMBA:
    @njit(parallel=True, cache=True)
    def _interp2linear_numba(z, xi, yi, ncols, nrows, extrapval):
        """Fused bilinear-interpolation kernel (compiled, parallel over output pts).

        Reads four neighbours + two float32 weights per output point and writes the
        result in a single pass, with no full-size temporaries — the dominant cost
        of the NumPy reference is the ~dozen 133M-element arrays its expression
        graph materialises, not the arithmetic. Edge/out-of-range handling mirrors
        interp2linear exactly so the output stays equivalent to MATLAB interp2.
        """
        n = xi.size
        out = np.empty(n, dtype=z.dtype)
        ncols_m1 = np.float32(ncols - 1)
        nrows_m1 = np.float32(nrows - 1)
        one = np.float32(1.0)
        for k in prange(n):
            x = xi[k]
            y = yi[k]
            if x < 0.0 or x > ncols_m1 or y < 0.0 or y > nrows_m1:
                out[k] = extrapval
                continue
            x0 = np.intp(math.floor(x))
            y0 = np.intp(math.floor(y))
            # On the far edge the +1 neighbour would be out of bounds: step one
            # cell back and let the fractional weight reach 1 (matches interp2linear).
            on_x = (x == ncols_m1)
            on_y = (y == nrows_m1)
            if on_x:
                x0 = x0 - 1
            if on_y:
                y0 = y0 - 1
            fx = x - np.float32(x0)
            fy = y - np.float32(y0)
            if on_x:
                fx = one
            if on_y:
                fy = one
            base = y0 * ncols + x0
            z00 = z[base]
            z10 = z[base + 1]
            z01 = z[base + ncols]
            z11 = z[base + ncols + 1]
            omfx = one - fx
            omfy = one - fy
            out[k] = (z00 * omfy + z01 * fy) * omfx + (z10 * omfy + z11 * fy) * fx
        return out


def _interp2linear_numpy(z, xi, yi, ncols, nrows, extrapval):
    """Pure-NumPy reference path (also the fallback when numba is absent).

    Identical math/edge logic to _interp2linear_numba; kept as the ground truth
    for the equivalence self-check and so the module works without numba.
    """
    out_shape = xi.shape
    xi = np.ascontiguousarray(xi, dtype=np.float32).ravel()
    yi = np.ascontiguousarray(yi, dtype=np.float32).ravel()

    x_bad = (xi < 0) | (xi > ncols - 1)
    y_bad = (yi < 0) | (yi > nrows - 1)
    bad = x_bad | y_bad

    # Clamp out-of-range coords to a valid cell so the index math is well-formed;
    # we overwrite their results with extrapval at the end.
    xi = np.where(x_bad, np.float32(0.0), xi)
    yi = np.where(y_bad, np.float32(0.0), yi)

    x0 = np.floor(xi).astype(np.intp)
    y0 = np.floor(yi).astype(np.intp)

    on_x_edge = xi == ncols - 1
    on_y_edge = yi == nrows - 1
    x0 = np.where(on_x_edge, x0 - 1, x0)
    y0 = np.where(on_y_edge, y0 - 1, y0)

    fx = (xi - x0).astype(np.float32)
    fy = (yi - y0).astype(np.float32)
    fx = np.where(on_x_edge, np.float32(1.0), fx)
    fy = np.where(on_y_edge, np.float32(1.0), fy)

    zr = z.ravel()
    base = y0 * ncols + x0
    z00 = zr[base]
    z10 = zr[base + 1]
    z01 = zr[base + ncols]
    z11 = zr[base + ncols + 1]

    one_minus_fy = 1.0 - fy
    one_minus_fx = 1.0 - fx
    f = (z00 * one_minus_fy + z01 * fy) * one_minus_fx + (
        z10 * one_minus_fy + z11 * fy) * fx

    if bad.any():
        f[bad] = extrapval

    return f.reshape(out_shape)


def interp2linear(z, xi, yi, extrapval=np.nan):
    """Bilinear interpolation, equivalent to MATLAB interp2(z, xi, yi, 'linear').

    z: 2-D array sampled on the integer lattice [0..ncols) x [0..nrows) (C order).
    xi, yi: arrays of column/row coordinates at which to interpolate.
    extrapval: fill value for out-of-range coordinates (default NaN).

    Uses the numba-compiled fused kernel when numba is available (single pass,
    no full-size temporaries, parallel over output points); otherwise falls back
    to the pure-NumPy reference. Both paths stay in the input dtype throughout —
    coordinates/weights are float32 so a complex64 gather-and-weight never
    upcasts to complex128 — and the caller's xi/yi arrays are not modified.
    """
    nrows, ncols = z.shape
    if nrows < 2 or ncols < 2:
        raise ValueError("z shape is too small")
    if xi.shape != yi.shape:
        raise ValueError("sizes of X indexes and Y-indexes must match")

    extrap = z.dtype.type(extrapval)
    if _HAVE_NUMBA:
        zr = np.ascontiguousarray(z, dtype=z.dtype).ravel()
        xi_c = np.ascontiguousarray(xi, dtype=np.float32).ravel()
        yi_c = np.ascontiguousarray(yi, dtype=np.float32).ravel()
        out = _interp2linear_numba(zr, xi_c, yi_c, np.intp(ncols), np.intp(nrows), extrap)
        return out.reshape(xi.shape)

    return _interp2linear_numpy(z, xi, yi, ncols, nrows, extrap)


def _interp2linear_selfcheck():
    """Equivalence check: numba path vs NumPy reference on a small random grid.

    Covers interior points, the far-edge step-back case, and out-of-range coords
    (NaN fill). Verifies value closeness and that NaN positions match exactly.
    """
    if not _HAVE_NUMBA:
        print("numba not available — self-check skipped (NumPy path only).")
        return True

    rng = np.random.default_rng(0)
    nrows, ncols = 37, 29
    z = (rng.standard_normal((nrows, ncols)) + 1j * rng.standard_normal((nrows, ncols))).astype(np.complex64)

    # a mix of interior, edge, just-outside, and clearly-out-of-range query points
    xs = np.array([-0.5, 0.0, 0.3, 1.25, 5.9, float(ncols - 1), float(ncols - 0.25),
                   float(ncols), float(ncols + 2)], dtype=np.float32)
    ys = np.array([float(nrows - 1), 0.0, 2.7, 4.1, float(nrows - 1), 0.0,
                   float(nrows - 0.5), float(nrows), -1.0], dtype=np.float32)
    xi, yi = np.meshgrid(xs, ys)

    ref = _interp2linear_numpy(z, xi, yi, ncols, nrows, z.dtype.type(np.nan))
    got = interp2linear(z, xi, yi, extrapval=np.nan)

    # The fill value is nan+0j (real NaN, imag 0), so NaN-ness must be detected
    # via isfinite (True only when both parts are finite), not by requiring both
    # parts to be NaN.
    ref_bad = ~np.isfinite(ref)
    got_bad = ~np.isfinite(got)
    nan_match = np.array_equal(ref_bad, got_bad)

    # compare only the finite (in-range) entries
    finite = ~ref_bad
    vals_close = np.allclose(got[finite], ref[finite], rtol=1e-5, atol=1e-6)
    max_diff = np.nanmax(np.abs(got[finite] - ref[finite])) if finite.any() else 0.0

    ok = bool(nan_match and vals_close)
    print(f"interp2linear self-check: {'PASS' if ok else 'FAIL'} "
          f"(nan_mask={nan_match}, values_close={vals_close}, "
          f"max_abs_diff={max_diff:.3e})")
    if not ok:
        print("ref:\n", ref)
        print("got:\n", got)
    return ok


if __name__ == "__main__":
    _interp2linear_selfcheck()