"""
Frequency-wavenumber (f-k) band / fan filter for 2-D gathers.

The filter passes energy whose temporal frequency |f| lies inside
[f_min, f_max] *and* whose apparent velocity |f| / |k| lies inside
[v_min, v_max].  Any of the four limits may be ``None`` (or ``np.inf`` for
the upper ones), in which case that side of the band is left open.

Compared with a naive implementation, the two things this module is careful
about are:

  * an open edge is genuinely open -- it is not emulated by a huge number,
    which would blow up the taper width; and
  * the cosine taper is clamped so that it never runs past zero, so the
    -6 dB corner stays where you asked for it even when f_min or v_min is
    small relative to the band width.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift, fftfreq

__all__ = ["fk_filter", "fk_mask", "tukey"]

Limit = Optional[float]


# --------------------------------------------------------------------------
# windows and 1-D edge tapers
# --------------------------------------------------------------------------
def tukey(n: int, alpha: float) -> np.ndarray:
    """Tukey (cosine-tapered) window of length ``n``.

    ``alpha`` is the total fraction of the window that is tapered, split
    evenly between the two ends (alpha=0 -> boxcar, alpha=1 -> Hann).
    """
    if n <= 0:
        raise ValueError("window length must be positive")
    alpha = float(np.clip(alpha, 0.0, 1.0))
    w = np.ones(n, dtype=float)
    if alpha == 0.0:
        return w
    width = min(int(alpha / 2.0 * n), n // 2)   # never let the ramps overlap
    if width < 1:
        return w
    ramp = 0.5 * (1.0 - np.cos(np.pi * np.linspace(0.0, 1.0, width)))
    w[:width] = ramp
    w[-width:] = ramp[::-1]
    return w


def _ramp_up(val: np.ndarray, edge: float, width: float) -> np.ndarray:
    """Rising raised-cosine edge: 0 below ``edge - width/2``, 1 above
    ``edge + width/2``, 0.5 exactly at ``edge``.

    ``width`` is shrunk if necessary so the ramp cannot extend below zero;
    ``val`` is a non-negative quantity (|f| or |f|/|k|), so a ramp that
    started at a negative value would silently move the corner.
    """
    out = np.zeros(val.shape, dtype=float)
    width = min(float(width), 2.0 * float(edge))
    if width <= 0.0:
        out[val >= edge] = 1.0
        return out
    lo, hi = edge - width / 2.0, edge + width / 2.0
    out[val >= hi] = 1.0                       # np.inf lands here, not in the ramp
    ramp = (val > lo) & (val < hi)
    out[ramp] = 0.5 * (1.0 - np.cos(np.pi * (val[ramp] - lo) / width))
    return out


def _ramp_down(val: np.ndarray, edge: float, width: float) -> np.ndarray:
    """Falling raised-cosine edge: 1 below ``edge - width/2``, 0 above
    ``edge + width/2``, 0.5 exactly at ``edge``."""
    out = np.zeros(val.shape, dtype=float)
    width = min(float(width), 2.0 * float(edge))
    if width <= 0.0:
        out[val <= edge] = 1.0
        return out
    lo, hi = edge - width / 2.0, edge + width / 2.0
    out[val <= lo] = 1.0
    ramp = (val > lo) & (val < hi)             # np.inf fails this -> stays 0
    out[ramp] = 0.5 * (1.0 + np.cos(np.pi * (val[ramp] - lo) / width))
    return out


def _clean_limit(x: Limit, name: str, allow_zero: bool = True) -> Limit:
    """Normalise a band limit: None / inf / nan all mean 'open'."""
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None                            # np.inf is a legal 'open' spelling
    if x < 0.0 or (x == 0.0 and not allow_zero):
        raise ValueError(f"{name} must be positive (got {x})")
    return x


def _band_mask(val: np.ndarray, lo: Limit, hi: Limit, taper_frac: float,
               name: str) -> np.ndarray:
    """Raised-cosine band mask over ``val``; either edge may be open.

    The taper width is a fraction of the band span when both edges are given,
    and a fraction of the single edge value when only one is given.
    """
    lo = _clean_limit(lo, f"{name}_min")
    hi = _clean_limit(hi, f"{name}_max", allow_zero=False)

    if lo is not None and hi is not None and lo >= hi:
        raise ValueError(f"{name}_min ({lo}) must be smaller than {name}_max ({hi})")

    if lo is None and hi is None:              # both open -> no constraint
        return np.ones(val.shape, dtype=float)

    taper_frac = max(0.0, float(taper_frac))
    if lo is not None and hi is not None:
        span = hi - lo
    else:
        span = hi if lo is None else lo        # one-sided: scale on the edge itself
    width = taper_frac * span

    mask = np.ones(val.shape, dtype=float)
    if lo is not None:
        mask *= _ramp_up(val, lo, width)
    if hi is not None:
        mask *= _ramp_down(val, hi, width)
    return mask


# --------------------------------------------------------------------------
# the mask in the f-k plane
# --------------------------------------------------------------------------
def fk_mask(
    shape: Tuple[int, int],
    dt: float,
    dx: float,
    f_min: Limit = None,
    f_max: Limit = None,
    v_min: Limit = None,
    v_max: Limit = None,
    cos_taper_width: float = 0.1,
) -> np.ndarray:
    """Build the (fftshift-ed) f-k mask for a gather of ``shape = (Nt, Nx)``.

    Useful on its own for QC -- plot it against ``fftshift(fftfreq(...))``
    axes to see exactly what the filter keeps.
    """
    Nt, Nx = shape
    f = fftshift(fftfreq(Nt, d=dt))
    k = fftshift(fftfreq(Nx, d=dx))

    absF = np.abs(f)[:, None]                  # (Nt, 1)
    absK = np.abs(k)[None, :]                  # (1, Nx)

    # apparent velocity |f| / |k|; the k = 0 column is infinitely fast.
    V = np.divide(
        np.broadcast_to(absF, (Nt, Nx)),
        absK,
        out=np.full((Nt, Nx), np.inf),
        where=(absK > 0),
    )

    mask_f = _band_mask(absF, f_min, f_max, cos_taper_width, "f")
    mask_v = _band_mask(V, v_min, v_max, cos_taper_width, "v")
    return np.broadcast_to(mask_f, (Nt, Nx)) * mask_v


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------
def fk_filter(
    data: np.ndarray,
    f_min: Limit = None,
    f_max: Limit = None,
    v_min: Limit = None,
    v_max: Limit = None,
    dt: Optional[float] = None,
    dx: Optional[float] = None,
    taper_frac: float = 0.1,
    cos_taper_width: float = 0.1,
    return_mask: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Apply an f-k band / fan filter to a 2-D gather.

    Parameters
    ----------
    data : (Nt, Nx) array
        Time along axis 0, space (channel / offset) along axis 1.
    f_min, f_max : float or None
        Temporal frequency band in Hz. ``None`` (or ``np.inf`` for f_max)
        leaves that side open.
    v_min, v_max : float or None
        Apparent-velocity band in m/s, measured as |f| / |k|. ``None`` (or
        ``np.inf`` for v_max) leaves that side open. Leaving ``v_max`` open
        keeps the k = 0 column, i.e. spatially flat (infinitely fast) events.
    dt, dx : float
        Time and space sampling. Both are required; they are keyword-able
        only so the band limits can default to None.
    taper_frac : float
        Tukey taper applied to the data in t and x before the FFT, as a
        fraction of each axis. Suppresses wrap-around from the periodic FFT.
    cos_taper_width : float
        Width of the raised-cosine roll-off at each band edge, as a fraction
        of the band span. The requested limits are the -6 dB (half-amplitude)
        points.
    return_mask : bool
        Also return the fftshift-ed mask that was applied.

    Notes
    -----
    * The mask depends on |f| and |k| only, so the filter is **not**
      directional: it treats left- and right-going energy identically. For a
      one-sided fan filter you need the signed ratio f / k instead.
    * The output is the filtered version of the *tapered* data, so amplitudes
      near the four edges of the gather are reduced by ``taper_frac``.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be 2-D (Nt, Nx), got shape {data.shape}")
    if dt is None or dx is None:
        raise TypeError("dt and dx are required")
    if dt <= 0 or dx <= 0:
        raise ValueError("dt and dx must be positive")

    Nt, Nx = data.shape

    window = np.outer(tukey(Nt, taper_frac), tukey(Nx, taper_frac))
    spectrum = fftshift(fft2(data * window))

    mask = fk_mask((Nt, Nx), dt, dx, f_min, f_max, v_min, v_max, cos_taper_width)

    filtered = np.real(ifft2(ifftshift(spectrum * mask)))
    if return_mask:
        return filtered, mask
    return filtered
