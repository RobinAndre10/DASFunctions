"""Basic DAS data processing: demeaning, RMS, unit conversion,
integration, gauge length and concatenation."""
import numpy as np
import scipy as sp
import scipy.ndimage
import scipy.signal
from scipy.integrate import cumulative_trapezoid

__all__ = [
    "demean_array",
    "compute_RMS",
    "strain2strainRate",
    "strainRate2strain",
    "numerical_integration",
    "changeGaugeLength",
    "changeGaugeLength_OLD",
    "concatenate_DAS_timeaxis",
]


def demean_array(data):
    """
    Subtract the mean value from each element in the data array.

    Parameters:
    data (numpy array): The input data array to be demeaned.

    Returns:
    numpy array: The demeaned data array.
    """
    mean_value = np.mean(data, axis=0)
    demeaned_data = data - mean_value
    return demeaned_data


def compute_RMS(data,
                axisNum
                ):
    """
    Compute the root mean square (RMS) of the input data along a specified axis.
    Parameters:
    data (numpy array): The input data array.
    axisNum (int): The axis along which to compute the RMS. 0 for rows, 1 for columns.
    Returns:
    rms (numpy array): The RMS of the input data along the specified axis.
    """

    dataInRMS = np.sqrt(np.mean(np.square(data), axis=axisNum))

    return dataInRMS


def strain_2_strainRate(data,
                      dt
                      ):
    """
    Convert strain data to strain rate data by temporal differentiation.

    Parameters:
    data (matrix): The input strain data matrix [distance, time].
    dt (float): The time step [s].

    Returns:
    strainRate (numpy array): The converted strain rate data matrix.
    """

    strainRate = np.diff( data, 1 ) / dt

    return strainRate


def strainRate_2_strain(data,
                      dt
                      ):
    """
    Convert strain rate data to strain data by temporal integration.

    Parameters:
    data (matrix): The input strain rate data matrix [distance, time].
    dt (float): The time step [s].

    Returns:
    strain (numpy array): The converted strain data matrix.
    """

    strain = np.cumsum( data, axis=1 ) * dt

    return strain


def numerical_integration(vel, dt, axis=0, method="trapz", demean=True):
    """
    Convert particle velocity to displacement by time integration.

    vel : np.ndarray
        2D array with time along `axis`.
    dt : float
        Sampling interval [s].
    axis : int
        Axis corresponding to time.
    method : {"trapz", "rect"}
        Numerical integration method.
    demean : bool
        If True, subtract mean along time before integrating.
    """
    v = np.asarray(vel)

    if demean:
        mean_v = np.mean(v, axis=axis, keepdims=True)
        v = v - mean_v

    if method == "rect":
        disp = np.cumsum(v, axis=axis) * dt
    elif method == "trapz":
        nt = v.shape[axis]
        t = np.arange(nt) * dt
        disp = cumulative_trapezoid(v, t, axis=axis, initial=0.0)
    else:
        raise ValueError("method must be 'trapz' or 'rect'")

    return disp


def change_Gauge_Length(data,
                       oldROIDec,
                       newGL,
                       newROIDec,
                       windowType,
                       oldGL=None):
    """
    Synthetically increase the gauge length of DAS strain(-rate) data by
    convolving along the spatial axis with a smoothing window, then
    optionally re-decimating the spatial sampling.

    INPUT:
    data       = Original data, size [time, space], recorded/derived with
                 gauge length 'oldGL'.
    oldROIDec  = Spatial sampling (dx) of the original data, in the same
                 units as newGL/oldGL (e.g. metres).
    newGL      = The new (larger) gauge length that is wanted, in the same
                 units as oldROIDec.
    newROIDec  = Desired spatial sampling of the output. Only used for the
                 decimation step; must be an integer multiple of
                 oldROIDec, otherwise decimation is skipped (with a
                 warning) and the data is returned at the original
                 sampling.
    windowType = Name of the smoothing window. Passed straight to
                 scipy.signal.windows.get_window, so anything scipy
                 supports works ('boxcar', 'triang', 'hann', 'hamming',
                 'blackman', ('kaiser', beta), ...). The old aliases
                 'rectwin' / 'rectangularwindow' / 'squarewindow' /
                 'squarwin' -> 'boxcar' and 'triangle' -> 'triang' are
                 kept for backwards compatibility with the original code.
    oldGL      = Gauge length already embedded in 'data' (same units as
                 newGL). Defaults to oldROIDec, i.e. assumes the common
                 DAS configuration where the hardware gauge length equals
                 the channel spacing. Pass 0 to reproduce the original
                 function's behaviour exactly (which implicitly assumed
                 oldGL = 0 / a point-like input).

    -------------------------------------------------------------------------
    Return:
    newData  = Data with the new gauge length applied (and, if applicable,
               decimated spatial sampling).
    newGL    = The gauge length that was targeted (echoed back, as in the
               original function).
    actualDx = The spatial sampling actually present in newData. Equals
               newROIDec if decimation was applied, otherwise oldROIDec.
               (New — the original function returned newGL alone, which
               didn't tell the caller whether decimation had silently been
               skipped.)
    """

    # ------------------------------------------------------------------
    # Aaccount for the gauge length already present
    # in the input data instead of always convolving with a window of the
    # full width newGL. Convolving two averaging windows roughly adds
    # their widths, so to end up with an effective gauge length of newGL
    # (rather than newGL + oldGL), the conversion kernel should span
    # (newGL - oldGL), not newGL.
    #
    # NOTE: this changes numerical results relative to the original
    # implementation whenever oldGL > 0. Pass oldGL=0 to reproduce the
    # exact old behaviour.
    # ------------------------------------------------------------------
    if oldGL is None:
        oldGL = oldROIDec  # common case: hardware GL == channel spacing

    convWidth = newGL - oldGL
    if convWidth <= 0:
        raise ValueError(
            f"newGL ({newGL}) must be larger than oldGL ({oldGL}) to "
            "increase the gauge length."
        )

    # Number of taps needed to span convWidth at the original spatial
    # sampling (oldROIDec).
    nKernel = len(np.arange(-round(convWidth / 2), round(convWidth / 2) + 1, oldROIDec))

    # ------------------------------------------------------------------
    # Build the window via scipy's own
    # get_window instead of hand-rolled if/elif branches, so any window
    # scipy supports can be used. Old aliases are mapped to their scipy
    # equivalents so existing calls (e.g. windowType='triangle') keep
    # working unchanged.
    # ------------------------------------------------------------------
    aliasMap = {
        'rectwin': 'boxcar',
        'rectangularwindow': 'boxcar',
        'squarewindow': 'boxcar',
        'squarwin': 'boxcar',
        'triang': 'triang',
        'triangle': 'triang',
    }
    windowKey = windowType.lower() if isinstance(windowType, str) else windowType
    windowName = aliasMap.get(windowKey, windowKey)

    try:
        sigma = sp.signal.windows.get_window(windowName, nKernel)
    except ValueError:
        raise ValueError(f"Unknown windowType: {windowType}")

    # ------------------------------------------------------------------
    # Normalize the kernel to unit sum so the
    # convolution preserves signal amplitude/units. Without this, changing
    # the gauge length also silently rescaled the data by sum(sigma),
    # which depends on window length and type.
    # ------------------------------------------------------------------
    sigma = sigma / np.sum(sigma)

    # ------------------------------------------------------------------
    # Cconvolve the whole [time, space] array along
    # the spatial axis in one vectorized call instead of looping over
    # every time sample in Python.
    #
    # Use mode='reflect' instead of the implicit
    # zero-padding of scipy.signal.convolve(..., mode='same'). Zero-padding
    # tapers the amplitude of channels near the start/end of the fiber;
    # reflecting the data at the edges avoids that artificial rolloff.
    # ------------------------------------------------------------------
    newData = sp.ndimage.convolve1d(data, sigma, axis=1, mode='reflect')

    # ------------------------------------------------------------------
    # The original code computed
    #   if oldROIDec != newROIDec: decRate = ...
    #   if newROIDec % oldROIDec == 0: decRate = int(decRate); ...
    # as two SEPARATE if-statements. When oldROIDec == newROIDec, the
    # first branch is skipped (decRate never assigned), but the second
    # condition (x % x == 0) is still True, so `int(decRate)` raised
    # UnboundLocalError. Nesting the second check inside the first fixes
    # this, and decimation is now simply skipped (no-op) when the
    # sampling is already unchanged.
    # ------------------------------------------------------------------
    actualDx = oldROIDec
    if oldROIDec != newROIDec:
        if newROIDec % oldROIDec == 0:
            decRate = int(newROIDec / oldROIDec)
            newData = newData[:, ::decRate]
            actualDx = oldROIDec * decRate
        else:
            print('Warning: New spatial sampling (dx) is not an integer '
                  'multiple of the old. The old sampling is kept.')

    return newData, newGL, actualDx


def change_Gauge_Length_OLD(data,
                      oldROIDec,
                      newGL,
                      newROIDec,
                      windowType
                      ):
    """
    INPUT:
    data = Original data with gauge length equal to 'oldGL' (size [time,space])
    olddx = Spatial sampling of original data
    newGL = The new gauge length that is wanted [int]
    filterType = The wanted filter type (normally from -newGL/2 tp +newGL/2).
    Options: 'triangle'
    -------------------------------------------------------------------------
    Return:
    newData = Data with new gauge length
    newGL = The new gauge length that is wanted
    """

    if windowType.lower() in ['rectwin', 'rectangularwindow', 'squarewindow', 'squarwin']:
        sigma = np.ones(len(np.arange(-round(newGL/2), round(newGL/2)+1, oldROIDec)))
        sigma = sigma / np.sum(sigma)
    elif windowType.lower() in ['triang', 'triangle', 'triangle']:
        sigma = sp.signal.window.triang(len(np.arange(-round(newGL/2), round(newGL/2)+1, oldROIDec)))
        sigma = sigma / np.sum(sigma)
    else:
        raise ValueError("Unknown windowType")

    newData = np.zeros_like(data)
    for it in range(data.shape[0]):
        newData[it, :] = sp.signal.convolve(data[it, :], sigma, mode='same')

    if oldROIDec != newROIDec:
        decRate = newROIDec / oldROIDec
    if newROIDec % oldROIDec == 0:
        decRate = int(decRate)
        newData = newData[:, ::decRate]
    else:
        print('Warning: New spatial sampling (dx) is not an integer multiple of the old. The old is kept.')


    return newData, newGL


def concatenate_DAS_timeaxis(signal_part1,
                             signal_part2
                             ):
    """
    Function that will concatenate DAS data along the time axis.
    As DAS data always start at 0 strain at the first time indes,
    we need to shift the signal being concatetanated (part 2) to
    the first part 1. This also incude removing any non strain value
    in the start (if you do not start at t = 0)

    Parameters:
    signal_part1 (numpy matrix, (nx, nt)): The first part to be concatenated.
    signal_part2 (numpy matrix, (nx, nt)): The second part to be concatenated. This one will be shifted.

    Return:
    signal: The concatenated matrix.
    """

    if signal_part2.shape[0] != signal_part1.shape[0]:
        raise ValueError("signal_part1 and signal_part2 must have the same number of rows (spatial channels).")

    signal = np.concatenate((signal_part1, signal_part2-signal_part2[:,[0]]+signal_part1[:,[-1]]),1)

    return signal
