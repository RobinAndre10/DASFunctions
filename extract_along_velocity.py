"""
Extract amplitudes from an f-k spectrum along a line of constant apparent
velocity, v = f / |k|.

For each frequency the wavenumber k = f/v is computed directly (no search),
then the amplitude is read off the k axis by index-snapping ('nearest') or by
linear interpolation between the two straddling samples ('linear').

Both under-read a narrow ridge, because the crest of a coherent event is about
one k sample wide and f/v rarely lands on it. For comparing examples that is
usually fine: extract every example the same way and the bias largely cancels.
It does NOT cancel between different velocities -- f/v drifts across the sample
grid at a different rate for each v -- so treat a comparison across velocities
with more caution than a comparison across events at the same velocity.

Usage
-----
    S = np.fft.fftshift(np.fft.fft2(data))          # data is (Nt, Nx)
    f = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
    k = np.fft.fftshift(np.fft.fftfreq(Nx, dx))

    f_out, amp = extract_along_velocity(S, f, k, v=26.0, f_range=(0.02, 1.0))
"""

from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np

__all__ = ["extract_along_velocity", "fv_panel", "velocity_resolution",
           "plot_extraction"]


# --------------------------------------------------------------------------
def velocity_resolution(v: float, f: float, n_ch: int, dx: float) -> float:
    """Velocity smear from one wavenumber sample: dv = v^2 / (f * L).

    The aperture limit, not an extraction artefact. Two velocities closer than
    this are not separable at that frequency however you read the grid.
    """
    return v ** 2 / (f * n_ch * dx)


def _nearest_index(k: np.ndarray, kt: np.ndarray) -> np.ndarray:
    """Index of the k sample closest to each target (k must be increasing)."""
    j = np.clip(np.searchsorted(k, kt), 1, k.size - 1)
    return np.where(kt - k[j - 1] < k[j] - kt, j - 1, j)


# --------------------------------------------------------------------------
def extract_along_velocity(
    S: np.ndarray,
    f: np.ndarray,
    k: np.ndarray,
    v: float,
    interp: str = "nearest",
    side: str = "auto",
    f_range: Optional[Tuple[float, float]] = None,
    db: bool = False,
    db_ref: Optional[float] = None,
    return_complex: bool = False,
    return_index: bool = False,
):
    """Amplitude along the line k = -f/v (or +f/v) through an f-k spectrum.

    Parameters
    ----------
    S : (n_f, n_k) complex array
        fftshift-ed f-k spectrum of an (Nt, Nx) gather.
    f, k : 1-D strictly increasing arrays
        Frequency (Hz) and wavenumber (1/m) axes matching S.
    v : float
        Apparent velocity in m/s, positive. `side` picks the branch.
    interp : {'nearest', 'linear'}
        'nearest' snaps to the closest k sample; 'linear' interpolates between
        the two straddling samples.
    side : {'auto', 'negative', 'positive', 'both'}
        Sign of k. With the fftshift convention above, an event moving toward
        increasing channel number sits at negative k -- which is where your
        ridge is. 'auto' picks whichever branch carries more energy over the
        band; 'both' averages the two magnitudes.
    f_range : (lo, hi), optional
        Frequency band. Default: every f > 0.
    db : bool
        Return 20*log10(amp / db_ref). db_ref defaults to the trace maximum,
        so pass a shared reference when comparing examples on one axis.
    return_complex : bool
        Return the complex value rather than the magnitude.
    return_index : bool
        Also return the k index used at each frequency (nearest only; for
        'linear' it is the lower of the two samples). Handy for checking how
        far f/v drifts from the grid.

    Returns
    -------
    f_out, amp             (or f_out, amp, idx if return_index)
        Frequencies whose k = f/v falls inside the grid, and the amplitudes.
    """
    S = np.asarray(S)
    f = np.asarray(f, dtype=float)
    k = np.asarray(k, dtype=float)

    if S.shape != (f.size, k.size):
        raise ValueError(
            f"S has shape {S.shape}, expected {(f.size, k.size)}. S must be "
            "(n_freq, n_wavenumber) and fftshift-ed, with f and k shifted to "
            "match.")
    if np.any(np.diff(f) <= 0) or np.any(np.diff(k) <= 0):
        raise ValueError("f and k must be strictly increasing -- apply "
                         "np.fft.fftshift to the spectrum and to both axes")
    if v <= 0:
        raise ValueError("v must be positive -- use `side` to pick the branch")
    if interp not in ("nearest", "linear"):
        raise ValueError("interp must be 'nearest' or 'linear'")

    # frequencies to use
    lo, hi = f_range if f_range is not None else (f[f > 0].min(), f.max())
    sel = (f >= lo) & (f <= hi) & (f > 0)
    if not sel.any():
        raise ValueError("no frequencies in the requested range")
    f_out, rows = f[sel], S[sel, :]

    # the wavenumber wanted at each frequency
    k_mag = f_out / v
    k_lim = min(abs(k[0]), k[-1])
    inside = k_mag <= k_lim
    if not inside.any():
        raise ValueError(
            f"the v = {v} m/s line leaves the k grid over the whole band "
            f"(needs |k| up to {k_mag.max():.4g}, grid reaches {k_lim:.4g}). "
            "Check the units of k, or lower f_range.")
    if not inside.all():
        warnings.warn(
            f"the v = {v} m/s line crosses k-Nyquist at "
            f"{f_out[inside].max():.3g} Hz; higher frequencies are dropped "
            "(spatial aliasing limit v / (2 dx)).", stacklevel=2)
    f_out, rows, k_mag = f_out[inside], rows[inside], k_mag[inside]

    i = np.arange(rows.shape[0])

    def sample(signed_k):
        if interp == "nearest":
            j = _nearest_index(k, signed_k)
            return rows[i, j], j
        j = np.clip(np.searchsorted(k, signed_k) - 1, 0, k.size - 2)
        w = (signed_k - k[j]) / (k[j + 1] - k[j])
        return rows[i, j] * (1.0 - w) + rows[i, j + 1] * w, j

    if side == "auto":
        en_neg = np.abs(sample(-k_mag)[0]).sum()
        en_pos = np.abs(sample(+k_mag)[0]).sum()
        side = "negative" if en_neg >= en_pos else "positive"

    if side == "negative":
        vals, idx = sample(-k_mag)
    elif side == "positive":
        vals, idx = sample(+k_mag)
    elif side == "both":
        a, idx = sample(-k_mag)
        b, _ = sample(+k_mag)
        vals = 0.5 * (np.abs(a) + np.abs(b))
    else:
        raise ValueError("side must be 'auto', 'negative', 'positive', 'both'")

    amp = vals if (return_complex and not db) else np.abs(vals)
    if db:
        ref = db_ref if db_ref is not None else np.max(np.abs(amp))
        amp = 20.0 * np.log10(np.abs(amp) / ref + 1e-300)

    return (f_out, amp, idx) if return_index else (f_out, amp)


# --------------------------------------------------------------------------
def fv_panel(S, f, k, velocities, **kw):
    """(n_f, n_v) amplitude panel -- one column per velocity.

    Returns (f_out, panel). Useful for scanning which velocity a ridge sits at,
    or for comparing a handful of examples on one image.
    """
    f_out, cols = None, []
    for v in np.atleast_1d(velocities):
        f_out, a = extract_along_velocity(S, f, k, float(v), **kw)
        cols.append(np.abs(a))
    return f_out, np.column_stack(cols)


def plot_extraction(S, f, k, v, f_range=None, vmin_db=-60, interp="nearest"):
    """QC figure: f-k panel with the extraction line, plus the spectrum."""
    import matplotlib.pyplot as plt

    f_out, amp = extract_along_velocity(S, f, k, v, interp=interp,
                                        f_range=f_range)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    A = 20 * np.log10(np.abs(S) / np.abs(S).max() + 1e-300)
    a1.pcolormesh(k, f, A, vmin=vmin_db, vmax=0, shading="auto")
    a1.plot(-f_out / v, f_out, "r--", lw=1.0, label=f"v = {v:g} m/s")
    a1.plot(+f_out / v, f_out, "r--", lw=1.0, alpha=0.4)
    a1.set_xlabel("k (1/m)")
    a1.set_ylabel("f (Hz)")
    a1.set_ylim(0, f_out.max())
    a1.legend(loc="upper right", fontsize=8)

    a2.plot(f_out, 20 * np.log10(amp / amp.max() + 1e-300), "k-", lw=1.0)
    a2.set_xlabel("f (Hz)")
    a2.set_ylabel("amplitude (dB re peak)")
    a2.grid(alpha=0.3)
    a2.set_title(f"along v = {v:g} m/s ({interp}), "
                 f"peak {f_out[amp.argmax()]:.3f} Hz")
    fig.tight_layout()
    return fig