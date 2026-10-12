"""Time-domain windows and tapers."""
import numpy as np
import scipy as sp
import scipy.signal

__all__ = [
    "timeDomainWindow",
    "fade_in_taper",
]


def time_Domain_Window(type,
                     length,
                     alpha
                     ):
    """
    Function to generate a time domain window to be used in further, e.g., frequency domain operations.

    Parameters:
    type is a string that specifies the type of window to be generated.
    length is the length of the window to be generated.
    alpha is the taper rate of the window, only applicable for type: 'tukey' .

    Returns:
    window is the window of the specified type and length.
    """
    if type == 'tukey':
        window = sp.signal.windows.tukey(length, alpha)
    elif type == 'hamming':
        window = sp.signal.windows.hamming(length)
    elif type == 'hanning':
        window = sp.signal.windows.hanning(length)
    elif type == 'blackman':
        window = sp.signal.windows.blackman(length)

    return window


def fade_in_taper(x, n_taper, kind="half_cosine"):
    """
    Apply a fade-in taper to the first n_taper samples of x.
    The taper goes from 0 -> 1, so x[0] becomes 0.

    Parameters
    ----------
    x : array_like
        1D time series (real or complex).
    n_taper : int
        Number of samples to taper at the start.
    kind : str
        "half_cosine" (default) or "linear" or "tukey".

    Returns
    -------
    y : ndarray
        Tapered signal (copy).
    w : ndarray
        The taper window used (length len(x)).
    """
    x = np.asarray(x)
    n = x.shape[0]
    n_taper = int(np.clip(n_taper, 0, n))
    w = np.ones(n, dtype=float)

    if n_taper == 0:
        return x.copy(), w

    if kind == "linear":
        w[:n_taper] = np.linspace(0.0, 1.0, n_taper, endpoint=True)

    elif kind == "half_cosine":
        # 0 -> 1 smoothly
        k = np.arange(n_taper)
        w[:n_taper] = 0.5 * (1.0 - np.cos(np.pi * (k / (n_taper - 1 if n_taper > 1 else 1))))

    elif kind == "tukey":
        # Equivalent to a Tukey-like rise portion (cosine)
        k = np.arange(n_taper)
        w[:n_taper] = 0.5 * (1.0 - np.cos(np.pi * (k / (n_taper - 1 if n_taper > 1 else 1))))

    else:
        raise ValueError("kind must be 'half_cosine', 'linear', or 'tukey'")

    y = x * w
    return y, w
