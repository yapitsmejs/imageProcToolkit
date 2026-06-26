import numpy as np
from .interp2 import interp2linear
from .fftTranslate2d import fftTranslate2d

# cupy is an optional GPU fast-path. Only the **translation** stage of a similarity
# transform can use it (via fftTranslate2d's own guarded switch); the rotation+scale
# warp is a bilinear resample run through interp2 (numba, CPU) -- there is no cupy
# bilinear-warp path in this toolkit, so similarityRotScaleImage never dispatches to
# cupy. The gate is imported for parity with the rest of the package and so a future
# GPU warp could slot in here; the module never hard-fails on a CPU-only host.
try:
    import cupy as cp  # noqa: F401
    try:
        _HAVE_CUPY_GPU = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        _HAVE_CUPY_GPU = False
except ImportError:
    cp = None
    _HAVE_CUPY_GPU = False

'''
    Apply a 4-DOF similarity transform G = (theta, s, dy, dx) to an image: rotation by
    theta (radians), uniform scale by s, then translation (dy, dx) = (rows/y, cols/x).
    Atomic + batched, mirroring fftTranslate2d.py.

    --------------------------------------------------------------------------- #
    Convention

    G is the similarity to **apply to the image** to align it to a reference, matching
    getSimilarityTransform's per-image G_i = (theta_i, s_i, dy_i, dx_i). The forward
    map on coordinates (relative to the image centre) is

        x_out = s * R(theta) * x_src + t          (R(theta) on (x, y) = (col, row))

    so the **inverse** map (the one a backward warp needs -- "which source pixel lands
    at this output pixel?") is

        x_src = (1/s) * R(-theta) * (x_out - t)

    which is exactly `image((1/s) R(-theta) (x - t))`. The rot/scale stage implements
    the t = 0 half via a bilinear inverse-map warp (interp2linear); the translate stage
    appends t via fftTranslate2d's phase ramp (output[n] = input[n - (dy, dx)]).
    Composing rot/scale FIRST then translate gives

        out(x) = fftTranslate2d(out_rs, t)(x) = out_rs(x - t)
               = image((1/s) R(-theta) (x - t))           -- the full similarity

    i.e. G_i = T(t'_i) compose G_i^{rs} = (theta_i, s_i, t'_i), where t'_i is the
    translation getSimilarityTransform's stage-2 estimates on the de-rotated/de-scaled
    image. The reverse order (translate then rot/scale) would give image((1/s)R(-theta)x
    - t) -- the translation not rotated into the source frame -- so the order matters.

    --------------------------------------------------------------------------- #
    Dtype + NaN handling (carried over from fftTranslate2d)

    complex in -> complex out (phase preserved through BOTH the bilinear warp, which
    keeps complex64, and the phase-ramp translate); real in -> real out. interp2linear
    preserves the input dtype (float32 weights never upcast a complex64 gather), and
    fftTranslate2d casts back with .astype(image.dtype) at its return.

    NaN borders grow on every look, as in the rest of the pipeline: the warp marks
    out-of-source pixels NaN (interp2 extrapval=NaN); fftTranslate2d zero-fills input
    NaN before its FFT (an NaN anywhere in fft2 poisons the whole array) and, with
    markInvalid=True (default), NaN-fills the shifted-out border. So a similarity
    transform of an image with a NaN border grows that border -- the intended, cleaner
    behaviour for interferometry.

    Real-input sub-pixel caveat (from fftTranslate2d, carried over): the translate
    stage takes np.abs of the phase-ramp IFFT for real input, faithful for ~integer
    shifts with minor distortion for sub-pixel real translations; the complex branch
    (the interferometry case) has no such fold. Radio calibration is out of scope.
'''


# --------------------------------------------------------------------------- #
# rotation + uniform-scale warp (inverse-map bilinear)
# --------------------------------------------------------------------------- #
def similarityRotScaleImage(image, params, markInvalid=True):
    """Apply rotation theta + uniform scale s to `image` via an inverse-map bilinear
    warp, output(x) = image((1/s) R(-theta) x) with x relative to the image centre.

    This is the rot/scale half of the similarity (translation 0); the full similarity
    is similarityTransform2d (this stage followed by fftTranslate2d).

    Args:
        image: (H, W) complex or real array. NaNs propagate through the bilinear
            weights (an input NaN border stays NaN); out-of-source pixels are filled
            with the extrapval.
        params: (theta, s) -- theta in radians, s > 0 the uniform scale. (Extra
            trailing elements, e.g. a full (theta, s, dy, dx) row, are ignored so a
            caller can pass the 4-vector and only the rot/scale is applied.)
        markInvalid: if True (default), out-of-source pixels are NaN (interp2
            extrapval=NaN); if False, they are zero-filled (extrapval=0), mirroring
            fftTranslate2d's markInvalid=False shipped behaviour.

    Returns:
        (H, W) array of the same dtype as `image`, rotated by theta and scaled by s
        about the centre, with an NaN (or zero, if markInvalid=False) border where the
        source has no sample.
    """
    image = np.asarray(image)
    H, W = image.shape
    theta = float(params[0])
    s = float(params[1])
    if s <= 0:
        raise ValueError(f"scale must be > 0, got {s}")

    # The warp is a sub-pixel resampling, so the output is inherently floating point:
    # promote integer input (e.g. the uint8 estimation image) to float32 so the bilinear
    # weights are not truncated and the NaN out-of-source border is representable
    # (interp2linear cannot make an integer NaN). Complex/float dtypes are preserved.
    out_dtype = np.float32 if np.issubdtype(image.dtype, np.integer) else image.dtype
    fimg = image.astype(out_dtype, copy=False)

    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    ct = np.cos(theta)
    st = np.sin(theta)
    inv_s = 1.0 / s

    # output pixel (r, c) -> source pixel via the inverse map
    #   x_src = cx + (1/s) * ( cos theta * dx + sin theta * dy)
    #   y_src = cy + (1/s) * (-sin theta * dx + cos theta * dy)
    # with dx = c - cx, dy = r - cy (R(-theta) on (x, y) = (col, row)).
    rs = np.arange(H, dtype=np.float64)[:, None] - cy        # (H, 1) dy
    cs = np.arange(W, dtype=np.float64)[None, :] - cx        # (1, W) dx
    src_x = cx + inv_s * (ct * cs + st * rs)                 # (H, W) col coords
    src_y = cy + inv_s * (-st * cs + ct * rs)                # (H, W) row coords

    extrap = fimg.dtype.type(np.nan) if markInvalid else fimg.dtype.type(0)
    out = interp2linear(fimg, src_x, src_y, extrapval=extrap)
    return out.astype(out_dtype)


# --------------------------------------------------------------------------- #
# atomic full similarity (rot/scale then translate)
# --------------------------------------------------------------------------- #
def similarityTransform2d(image, params, markInvalid=True):
    """Apply the full 4-DOF similarity G = (theta, s, dy, dx) to `image`: rotate by
    theta, uniformly scale by s, then translate by (dy, dx).

    Composition is rot/scale FIRST, then translate (see the module docstring): this
    matches G_i = T(t'_i) compose G_i^{rs} = (theta_i, s_i, t'_i) where t'_i is the
    translation estimated by getSimilarityTransform's stage 2 on the de-rotated/
    de-scaled image. Applying similarityTransform2d(image_i, G_i) therefore
    co-registers the set.

    Args:
        image: (H, W) complex or real array, on-disk L2 convention (axis 0 = range/y,
            axis 1 = azimuth/x). May contain NaN borders.
        params: (theta, s, dy, dx) -- theta in radians, s > 0, (dy, dx) = (rows/y,
            cols/x) the translation.
        markInvalid: forwarded to both stages (default True). NaN-fills the warp's
            out-of-source border and the translate's shifted-out border; the NaN border
            grows on every look (intended, cleaner for interferometry).

    Returns:
        (H, W) array of the same dtype as `image`: complex in -> complex out (phase
        preserved through both stages), real in -> real out (fftTranslate2d's
        np.abs sub-pixel caveat carries over).
    """
    image = np.asarray(image)
    theta = float(params[0])
    s = float(params[1])
    dy = float(params[2])
    dx = float(params[3])

    # rot/scale first (inverse-map bilinear warp), then translate (FFT phase ramp).
    # The warp's NaN border becomes the translate stage's input-NaN border, which
    # fftTranslate2d zero-fills before its FFT and re-marks with markInvalid.
    warped = similarityRotScaleImage(image, (theta, s), markInvalid)
    out = fftTranslate2d(warped, (dy, dx), markInvalid)
    return out.astype(image.dtype)


# --------------------------------------------------------------------------- #
# batched similarity (step 5)
# --------------------------------------------------------------------------- #
def similarityTransformImages(images, params, markInvalid=True):
    """Apply each per-image similarity (batched step 5). `params` is the (N, 4) array
    returned by getSimilarityTransform (columns (theta, log s, dy, dx) -- NB the scale
    column is exp'd here before application, row order = input order). Delegates to the
    atomic similarityTransform2d; markInvalid=True (default) NaN-fills the borders.
    """
    images = list(images)
    params = np.asarray(params, dtype=np.float64)
    if len(images) != params.shape[0]:
        raise ValueError(
            f"{len(images)} images but params has {params.shape[0]} rows")
    return [similarityTransform2d(
                img, (float(params[i, 0]), float(np.exp(params[i, 1])),
                      float(params[i, 2]), float(params[i, 3])), markInvalid)
            for i, img in enumerate(images)]


# --------------------------------------------------------------------------- #
# self-check: rot/scale round-trip + dtype preservation
# --------------------------------------------------------------------------- #
def _similarityTransform2d_selfcheck():
    """Round-trip a rotation+scale: apply (theta, s, 0, 0) then (-theta, 1/s, 0, 0)
    and assert the finite (non-border) region matches the original. The rot/scale
    subgroup is abelian so its inverse is trivial (-theta, 1/s); this validates the
    inverse-map warp's signs. Also checks complex64 / float32 dtype preservation and
    that a full similarity returns the input shape/dtype.
    """
    rng = np.random.default_rng(0)
    H, W = 64, 64
    # band-limited pattern: a few low-frequency sinusoids + an off-centre Gaussian.
    # A sharp edge (e.g. a bright cross) would accumulate several units of bilinear
    # error across two warps and falsely fail the round-trip; this stays smooth enough
    # that the round-trip residual reflects the warp's sign correctness, not aliasing.
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    base = (np.sin(0.18 * xx + 0.11 * yy) + np.cos(0.07 * xx - 0.05 * yy)
            + 3.0 * np.exp(-((xx - 22) ** 2 + (yy - 40) ** 2) / 80.0)).astype(np.float32)

    theta = 0.12        # ~7 deg, within the warp's clean range
    s = 1.08

    # round trip: G then G^{-1} (rot/scale subgroup inverse)
    fwd = similarityRotScaleImage(base, (theta, s), markInvalid=True)
    back = similarityRotScaleImage(fwd, (-theta, 1.0 / s), markInvalid=True)

    fin0 = np.isfinite(base)
    fin1 = np.isfinite(back)
    both = fin0 & fin1
    # compare only on the central region that survives both warps' NaN borders, eroded
    # by a few pixels to avoid the residual sinc roll-off at the finite/NaN edge.
    erode = 4
    both[:erode, :] = both[-erode:, :] = both[:, :erode] = both[:, -erode:] = False
    max_diff = float(np.nanmax(np.abs(back[both] - base[both]))) if both.any() else 0.0
    # bilinear (not cubic) smoothing over two warps leaves a ~1% residual on O(1-4)
    # values; a sign flip would give an O(1)+ residual, so atol=0.1 still catches it.
    vals_close = bool(both.any()) and np.allclose(back[both], base[both], atol=0.1)

    # dtype preservation
    cplx = (rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))).astype(np.complex64)
    cplx_out = similarityTransform2d(cplx, (0.05, 1.02, 0.5, -0.5))
    dtype_cplx = cplx_out.dtype == np.complex64
    shape_cplx = cplx_out.shape == cplx.shape

    real_out = similarityTransform2d(base, (0.05, 1.02, 1.0, -1.0))
    dtype_real = real_out.dtype == np.float32
    shape_real = real_out.shape == base.shape

    ok = bool(vals_close and dtype_cplx and shape_cplx and dtype_real and shape_real)
    print(f"similarityTransform2d self-check: {'PASS' if ok else 'FAIL'} "
          f"(roundtrip_close={vals_close}, max_abs_diff={max_diff:.3e}, "
          f"dtype_cplx={dtype_cplx}, dtype_real={dtype_real}, "
          f"shape_ok={shape_cplx and shape_real})")
    return ok


if __name__ == "__main__":
    _similarityTransform2d_selfcheck()