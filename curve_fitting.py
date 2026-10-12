"""Smoothing and robust fitting of 1-D curves."""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline

__all__ = [
    "smooth_curve",
    "robust_polyfit",
]


def smooth_curve(
    x,
    y,
    method="moving_average",
    window_size=11,
    polyorder=3,
    sigma=2,
    spline_smooth=None,
):
    """
    Smooth a 1D curve using several possible methods.

    Parameters
    ----------
    x : array_like
        1D array of x-values (must be sorted).
    y : array_like
        1D array of y-values to smooth.
    method : str
        Smoothing method: 'moving_average', 'gaussian', 'savgol', 'spline'
    window_size : int
        Window length (odd number recommended for moving/savgol).
    polyorder : int
        Polynomial order for Savitzky–Golay filter.
    sigma : float
        Standard deviation for Gaussian filter.
    spline_smooth : float or None
        Smoothing factor for spline (None = automatic / exact fit).
    show_plot : bool
        If True, plot the original and smoothed curves.

    Returns
    -------
    y_smooth : ndarray
        Smoothed version of y.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Ensure proper input
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")

    # Choose method
    if method == "moving_average":
        kernel = np.ones(window_size) / window_size
        y_smooth = np.convolve(y, kernel, mode="same")

    elif method == "gaussian":
        y_smooth = gaussian_filter1d(y, sigma=sigma, mode="nearest")

    elif method == "savgol":
        if window_size % 2 == 0:
            window_size += 1  # must be odd
        y_smooth = savgol_filter(y, window_length=window_size, polyorder=polyorder)

    elif method == "spline":
        spline = UnivariateSpline(x, y, s=spline_smooth)
        y_smooth = spline(x)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from 'moving_average', 'gaussian', 'savgol', 'spline'.")

    return y_smooth


def robust_polyfit(x,
                   y,
                   degree=1,
                   norm=None
                   ):
    """
    Robust polynomial fit using statsmodels.RLM.

    Parameters:
        x, y: 1D arrays of data points
        degree: Degree of polynomial (default = 1)
        norm: Robust norm (default = TukeyBiweight)

    Returns:
        coeffs: Polynomial coefficients (highest degree first, like np.polyfit)
        y_fit: Fitted y values at x
    """
    import statsmodels.api as sm
    x = np.asarray(x)
    y = np.asarray(y)

    # Construct design matrix: [1, x, x^2, ..., x^degree]
    X = np.vstack([x**i for i in range(degree + 1)]).T

    # Choose robust norm (TukeyBiweight ~ MATLAB 'bisquare')
    if norm == 'TukeyBiweight' or norm is None:
        norm = sm.robust.norms.TukeyBiweight()
    elif norm == 'Huber':
        norm = sm.robust.norms.HuberT()
    elif norm == 'Hampel':
        norm = sm.robust.norms.Hampel()

    # Fit robust model
    model = sm.RLM(y, X, M=norm)
    results = model.fit()

    coeffs = results.params[::-1]  # Reverse to match np.polyfit (highest degree first)
    y_fit = results.predict(X)

    return coeffs, y_fit
