"""Frequency filters (Butterworth, Ormsby, simple f-k) and gain control.

For the more careful f-k band/fan filter with open edges, see fk_filter.py.
"""
import numpy as np
import scipy as sp
import scipy.signal
import scipy.signal as sig
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift, fftfreq

from .tapers import timeDomainWindow

__all__ = [
    "filter_butterworth",
    "filter_butterworth_vClaude",
    "filter_ormsby_vClaude",
    "agc",
    "filter_time_domain",
]


def filter_butterworth(data,
                       filterOrder,
                       filter_type,
                       cutoff_freq,
                       Fs,
                       tprRate,
                       zeroPhase,
                       ):
    """
    Filter the input data in the time domain using a specified filter type and cutoff frequency.

    Parameters:
    data (numpy array): The input data array to be filtered.
    filterOrder (int): The order of the filter to be used.
    filter_type (str): The type of filter to be used. Options: 'lowpass', 'highpass', 'bandpass', 'bandstop
    cutoff_freq (float or list): The cutoff frequency of the filter. If filter_type is 'bandpass' or 'bandstop', this should be a list of two values
    Fs (float): The sample rate of the data [Hz].

    Return:
    data_filtered (numpy array): The filtered data array.
    """
    window = timeDomainWindow('tukey', len(data), tprRate)
    data = data * window

    # ### High-pass filter ###
    if filter_type == 'highpass' or filter_type == 'hp':
        if len(cutoff_freq) != 1:
            raise ValueError("The cutoff frequency should be one value contaning the cut-off frequency, when doing low-pass filtering")
        sos_HP = sp.signal.butter(filterOrder, cutoff_freq,  btype='highpass', analog=False, output = 'sos', fs = Fs)
        if zeroPhase:
            data_filtered = sp.signal.sosfiltfilt(sos_HP, data)
        else:
            data_filtered = sp.signal.sosfilt(sos_HP, data)

    elif filter_type == 'lowpass' or filter_type == 'lp':
        if len(cutoff_freq) != 1:
            raise ValueError("The cutoff frequency should be one value contaning the cut-off frequency, when doing low-pass filtering")
        sos_LP = sp.signal.butter(filterOrder, cutoff_freq,  btype='lowpass', analog=False, output = 'sos', fs = Fs)
        if zeroPhase:
            data_filtered = sp.signal.sosfiltfilt(sos_LP, data)
        else:
            data_filtered = sp.signal.sosfilt(sos_LP, data)

    elif filter_type == 'bandpass' or filter_type == 'bp':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        sos_BP = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandpass', analog=False, output = 'sos', fs = Fs)
        if zeroPhase:
            data_filtered = sp.signal.sosfiltfilt(sos_BP, data)
        else:
            data_filtered = sp.signal.sosfilt(sos_BP, data)

    elif filter_type == 'bandstop' or filter_type == 'bs':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        sos_BS = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandstop', analog=False, output = 'sos', fs = Fs)
        if zeroPhase:
            data_filtered = sp.signal.sosfiltfilt(sos_BS, data)
        else:
            data_filtered = sp.signal.sosfilt(sos_BS, data)

    return data_filtered


def filter_butterworth_vClaude(data, filterOrder, filter_type, cutoff_freq, Fs, tprRate, zeroPhase, axis=-1):
    """
    Taper then Butterworth-filter `data` along `axis`.

    Parameters
    ----------
    data : ndarray
        Input array; the Tukey taper and the filter are both applied along `axis`.
    filterOrder : int
        Butterworth filter order.
    filter_type : {'lowpass'/'lp', 'highpass'/'hp', 'bandpass'/'bp', 'bandstop'/'bs'}
    cutoff_freq : float or sequence of float
        One corner frequency for lowpass/highpass, two (low, high) for bandpass/bandstop [Hz].
    Fs : float
        Sample rate [Hz].
    tprRate : float
        Tukey taper fraction along `axis` before filtering (0 = rectangular/no taper,
        1 = full cosine), used to suppress edge transients.
    zeroPhase : bool
        True: sosfiltfilt (zero phase; effective order doubles and the -3 dB edge
        shifts inward relative to `cutoff_freq`). False: sosfilt (causal, single pass).
    axis : int, default -1
        Axis to taper/filter along (the time axis, for a (channels, samples) gather).

    Returns
    -------
    ndarray, same shape as `data`.
    """
    filter_types = {
        'highpass': 'highpass', 'hp': 'highpass',
        'lowpass': 'lowpass', 'lp': 'lowpass',
        'bandpass': 'bandpass', 'bp': 'bandpass',
        'bandstop': 'bandstop', 'bs': 'bandstop',
    }
    n_corners = {'highpass': 1, 'lowpass': 1, 'bandpass': 2, 'bandstop': 2}

    key = filter_type.lower() if isinstance(filter_type, str) else filter_type
    resolved_type = filter_types.get(key)
    if resolved_type is None:
        raise ValueError(f"Unknown filter_type {filter_type!r}; expected one of "
                          f"{sorted(set(filter_types))}")

    cutoff_freq = np.atleast_1d(np.asarray(cutoff_freq, dtype=float))
    n_expected = n_corners[resolved_type]
    if cutoff_freq.size != n_expected:
        raise ValueError(f"{resolved_type} needs {n_expected} corner frequency(ies), "
                          f"got {cutoff_freq.size}")
    # scipy.signal.butter raises on a length-1 Wn array for lowpass/highpass; give it a scalar
    Wn = cutoff_freq[0] if n_expected == 1 else cutoff_freq

    n_taper = data.shape[axis]
    window_shape = [1] * data.ndim
    window_shape[axis] = n_taper
    window = sig.windows.tukey(n_taper, alpha=tprRate).reshape(window_shape)
    data = data * window

    sos = sig.butter(filterOrder, Wn, btype=resolved_type, analog=False, output='sos', fs=Fs)
    if zeroPhase:
        return sig.sosfiltfilt(sos, data, axis=axis)
    return sig.sosfilt(sos, data, axis=axis)


def filter_ormsby_vClaude(data, numtaps, filter_type, corner_freq, Fs, tprRate, zeroPhase, axis=-1):
    """
    Taper then Ormsby-filter `data` along `axis`.

    The Ormsby filter is a trapezoidal filter in the frequency domain, specified by
    corner frequencies rather than an order — the classic exploration-seismology
    parameterization. It's built here as a linear-phase FIR (frequency-sampling
    method, `scipy.signal.firwin2`), which is the standard digital implementation
    of the trapezoid.

    Parameters
    ----------
    data : ndarray
        Input array; the Tukey taper and the filter are both applied along `axis`.
    numtaps : int
        FIR filter length. Use an odd value for an exact integer-sample delay
        ((numtaps - 1) / 2 samples). Longer = sharper corners, more compute/delay.
    filter_type : {'lowpass'/'lp', 'highpass'/'hp', 'bandpass'/'bp', 'bandstop'/'bs'}
    corner_freq : sequence of float [Hz], strictly increasing
        lowpass:  (f_pass, f_stop)        — gain tapers 1 -> 0 between them
        highpass: (f_stop, f_pass)        — gain tapers 0 -> 1 between them
        bandpass: (f1, f2, f3, f4)        — gain 0,0,1,1,0,0 at 0,f1,f2,f3,f4,Nyquist
        bandstop: (f1, f2, f3, f4)        — gain 1,1,0,0,1,1 at 0,f1,f2,f3,f4,Nyquist
    Fs : float
        Sample rate [Hz].
    tprRate : float
        Tukey taper fraction along `axis` before filtering (0 = rectangular/no taper,
        1 = full cosine), used to suppress edge transients.
    zeroPhase : bool
        True: scipy.signal.filtfilt (zero delay; the filter is effectively applied
        twice, so the magnitude response is squared and the transition shifts
        inward relative to the corner frequencies). False: lfilter (causal, single
        pass, constant delay of (numtaps - 1) / 2 samples — exactly known and
        shiftable, unlike an IIR filter's frequency-dependent group delay).
    axis : int, default -1
        Axis to taper/filter along (the time axis, for a (channels, samples) gather).

    Returns
    -------
    ndarray, same shape as `data`.
    """
    filter_types = {
        'highpass': 'highpass', 'hp': 'highpass',
        'lowpass': 'lowpass', 'lp': 'lowpass',
        'bandpass': 'bandpass', 'bp': 'bandpass',
        'bandstop': 'bandstop', 'bs': 'bandstop',
    }
    # number of corner frequencies each type needs, and what they're called
    corners_spec = {
        'lowpass':  ('f_pass, f_stop', 2),
        'highpass': ('f_stop, f_pass', 2),
        'bandpass': ('f1, f2, f3, f4', 4),
        'bandstop': ('f1, f2, f3, f4', 4),
    }

    key = filter_type.lower() if isinstance(filter_type, str) else filter_type
    resolved_type = filter_types.get(key)
    if resolved_type is None:
        raise ValueError(f"Unknown filter_type {filter_type!r}; expected one of "
                          f"{sorted(set(filter_types))}")

    names, n_expected = corners_spec[resolved_type]
    corner_freq = np.atleast_1d(np.asarray(corner_freq, dtype=float))
    if corner_freq.size != n_expected:
        raise ValueError(f"{resolved_type} needs {n_expected} corner frequencies "
                          f"({names}), got {corner_freq.size}")
    if np.any(np.diff(corner_freq) <= 0):
        raise ValueError(f"corner_freq must be strictly increasing ({names}), got {tuple(corner_freq)}")

    nyq = Fs / 2
    if resolved_type == 'lowpass':
        f_pass, f_stop = corner_freq
        freqs, gains = [0, f_pass, f_stop, nyq], [1, 1, 0, 0]
    elif resolved_type == 'highpass':
        f_stop, f_pass = corner_freq
        freqs, gains = [0, f_stop, f_pass, nyq], [0, 0, 1, 1]
    elif resolved_type == 'bandpass':
        f1, f2, f3, f4 = corner_freq
        freqs, gains = [0, f1, f2, f3, f4, nyq], [0, 0, 1, 1, 0, 0]
    else:  # bandstop
        f1, f2, f3, f4 = corner_freq
        freqs, gains = [0, f1, f2, f3, f4, nyq], [1, 1, 0, 0, 1, 1]
    if freqs[-2] >= nyq:
        raise ValueError(f"highest corner frequency ({freqs[-2]} Hz) must be below "
                          f"Nyquist ({nyq} Hz)")

    n_taper = data.shape[axis]
    window_shape = [1] * data.ndim
    window_shape[axis] = n_taper
    window = sig.windows.tukey(n_taper, alpha=tprRate).reshape(window_shape)
    data = data * window

    taps = sig.firwin2(numtaps, freqs, gains, fs=Fs)
    if zeroPhase:
        return sig.filtfilt(taps, 1, data, axis=axis)
    return sig.lfilter(taps, 1, data, axis=axis)

def agc(data, window_samples, axis=1):
    """
    Apply automatic gain control to 2D data.

    Parameters
    ----------
    data : np.ndarray
        Input 2D array.
    window_samples : int
        Length of the smoothing window in samples.
    axis : int
        Axis along which to compute and apply the gain.
        0 = along channels (normalize in space)
        1 = along time (normalize in time)  [default]
    """
    import numpy as np
    from scipy.ndimage import uniform_filter1d
    rms = np.sqrt(uniform_filter1d(data**2, size=window_samples, axis=axis))
    rms = np.where(rms < 1e-12, 1e-12, rms)
    return data / rms


def filter_time_domain(data, filterOrder, filter_type, cutoff_freq, Fs, tprRate,
                       zeroPhase=False):
    """
    Deprecated name for filter_butterworth, kept so older scripts still run.

    The original filter_time_domain filtered causally (sosfilt), so zeroPhase
    defaults to False here to reproduce that behaviour.
    """
    return filter_butterworth(data, filterOrder, filter_type, cutoff_freq,
                              Fs, tprRate, zeroPhase)
