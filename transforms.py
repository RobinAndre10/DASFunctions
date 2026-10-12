"""Spectral transforms: FFT, f-x, f-k, f-v, tau-p and spectrograms."""
import numpy as np
from scipy.fft import fft
from scipy.fftpack import fft2, fftshift, fftfreq
from scipy.signal import spectrogram, get_window

from .tapers import timeDomainWindow

__all__ = [
    "nextpow2",
    "my_fft",
    "my_fft_OLD",
    "fx_domain",
    "fk_domain",
    "fk_domain_conversion",
    "fk_inverse",
    "fk_to_fv",
    "taup_transform",
    "compute_spectrogram",
]


def nextpow2(n):
    """
    Returns the exponent of the next power of 2 greater than or equal to n. Often used to improve the fft function.
    """
    return np.ceil(np.log2(n)).astype(int)


def my_fft(signal,
           Fs,
           padding,
           return_psd=False):
    """
    Compute amplitude, phase, and optionally power spectral density (PSD)
    with both positive and negative frequencies (fftshifted).

    Normalized by dt = 1/Fs so that the amplitude spectrum approximates
    the continuous Fourier transform F(f) ≈ dt * Σ f(t) * e^(-i2π...),
    consistent with fk_domain which normalizes by dx*dt.

    Parameters
    ----------
    signal : array_like
        Input time series (nt,).
    Fs : float
        Sampling rate [Hz].
    padding : bool
        If True, zero-pad to the next power of two before FFT.
    return_psd : bool, optional
        If True, compute and return two-sided fftshifted PSD [x^2/Hz].

    Returns
    -------
    absSIGNAL : ndarray
        Amplitude spectrum (|FFT| * dt), fftshifted.  [input units × s]
    ff : ndarray
        Two-sided frequency vector [Hz], fftshifted.
    phaSIGNAL : ndarray
        Phase spectrum [radians], fftshifted.
    psdSIGNAL : ndarray (optional)
        Two-sided PSD [input units^2 / Hz], fftshifted.
    """

    signal = np.asarray(signal)
    N = len(signal)
    dt = 1.0 / Fs

    # Determine FFT length
    if padding:
        nfft = 2 ** nextpow2(N)
    else:
        nfft = N

    # FFT and normalization by dt (continuous transform convention)
    X = np.fft.fft(signal, nfft)
    Xn = X * dt  # normalized amplitude spectrum [input units × s]

    # Compute amplitude and phase (fftshifted)
    Xn_fftshifted = np.fft.fftshift(Xn)
    absSIGNAL = np.abs(Xn_fftshifted)
    phaSIGNAL = np.angle(Xn_fftshifted)
    ff = Fs * np.arange(-nfft / 2, nfft / 2) / nfft

    # ----- PSD computation -----
    # PSD scaling: |X(k)|^2 * dt / nfft  [input units^2 / Hz]
    PSD = (np.abs(X) ** 2) * dt / nfft

    # For real signals, redistribute power symmetrically to negative freqs
    if np.isrealobj(signal):
        PSD_shifted = np.real(np.fft.fftshift(PSD))
    else:
        PSD_shifted = np.fft.fftshift(PSD)

    return absSIGNAL, ff, phaSIGNAL, PSD_shifted

def fx_domain(data,
              Fs,
              padding,
              tprRate,
              flipping
              ):
    """
    Function to transform the input 2-dimensional data to the f-x domain.
    Parameters:
    data (numpy matrix, nx X nt): The input 2D-data matrix to be transformed.
    Fs (float): The sample rate of the data [Hz].
    padding (boolean): A boolean that decides if the signal should be padded with zeros before the fft, default to the next power of two.
    Return:
    signal_FXdom (numpy matrix, nx X nt): The transformed data in the f-x domain.
    """

    # This will flip the ends of the time axis and taper that part and not the signal of interest
    if flipping == True:
        # One example flip to get the zeros below correct
        dataT = data[0,:]
        window = timeDomainWindow('tukey', len(dataT), tprRate)
        # Flip each side according to where the taper is not equal to 1
        nidxNot0 = int(len( np.where( window < 1 )[0] )*5)
        del window
        trace2useA = np.concatenate( (dataT[nidxNot0:1:-1], dataT, dataT[-1:-nidxNot0:-1]) )
        nTimeWindow = len(trace2useA)
        window = timeDomainWindow('tukey', nTimeWindow, tprRate)
        del trace2useA
    else:
        window = timeDomainWindow('tukey', data.shape[1], tprRate)

    if padding == True and flipping == True:
        signal_FXdom = np.zeros( (np.int32(data.shape[0]), np.int32(2**nextpow2(nTimeWindow))), dtype=float )
        for i in range(data.shape[0]):
            print(f'Working on channel {i+1} of {data.shape[0]}')
            dataF = np.concatenate( (data[i,nidxNot0:1:-1], data[i,:], data[i,-1:-nidxNot0:-1]) )
            window = timeDomainWindow('tukey', len(dataF), tprRate)
            data2fx = dataF*window
            signal_FXdom[i,:], ff, phas, psd = my_fft(data2fx, Fs, True)
    elif padding == True and flipping == False:
        signal_FXdom = np.zeros( (np.int32(data.shape[0]), np.int32(2**nextpow2(data.shape[1]))), dtype=float )
        for i in range(data.shape[0]):
            print(f'Working on channel {i+1} of {data.shape[0]}')
            signal_FXdom[i,:], ff, phas, sss = my_fft(data[i,:]*window, Fs, True)
    else:
        signal_FXdom = np.zeros( (np.int32(data.shape[0]), np.int32(data.shape[1])), dtype=float )
        for i in range(data.shape[0]):
            print(f'Working on channel {i+1} of {data.shape[0]}')
            if tprRate != 0:
                signal_FXdom[i,:], ff, phas, psd = my_fft(data[i,:]*window, Fs, False)
            else:
                signal_FXdom[i,:], ff, phas, psd = my_fft(data[i,:], Fs, False)

    return signal_FXdom, ff


def fk_domain(data,
              dx,
              dt,
              padding=True,
              ):
    """
    Forward 2D FFT (x,t) -> (k,f), with optional zero-padding to next power of two.

    Amplitudes are normalized by dx*dt so that the result approximates the
    continuous 2D Fourier transform F(k,f) ≈ dx*dt * Σ f(x,t) * e^(-i2π...).
    This ensures amplitudes are comparable across datasets with different
    spatial or temporal sampling.

    Returns
    -------
    amp : |F(k,f)|  [input units × m × s]
    kk, ff : 1D axes (fftshifted) for wavenumber [1/m] and frequency [Hz]
    F : complex f-k spectrum (fftshifted, normalized by dx*dt)
    orig_shape : (nx, nt)
    nyquist : (kk_Nyq, ff_Nyq)
    """
    nx, nt = data.shape
    if padding:
        npx = 2 ** nextpow2(nx)
        npt = 2 ** nextpow2(nt)
    else:
        npx, npt = nx, nt

    # Axes that align with the shifted spectrum
    kk = np.fft.fftshift(np.fft.fftfreq(npx, d=dx))      # [1/m]
    ff = np.fft.fftshift(np.fft.fftfreq(npt, d=dt))      # [Hz]

    # Forward 2D FFT normalized by dx*dt to approximate the continuous transform
    F = np.fft.fftshift(np.fft.fft2(data, s=(npx, npt))) * dx * dt
    amp = np.abs(F)

    kk_Nyq = 1 / (2 * dx)
    ff_Nyq = 1 / (2 * dt)

    return amp, kk, ff, (nx, nt), (kk_Nyq, ff_Nyq), F


def fk_domain_conversion(data,
                         dx,
                         dt,
                         padding=True,
                         convert2=None,
                         eps=1e-10,
                         H = 1
                         ):
    """
    Forward 2D FFT (x,t) -> (k,f), with optional zero-padding to next power of two.

    Returns
    -------
    amp : |F(k,f)|
    kk, ff : 1D axes (fftshifted) for wavenumber [1/m] and frequency [Hz]
    F_conv : complex f-k spectrum after optional conversion (fftshifted)
    orig_shape : (nx, nt) for cropping on inverse
    """
    nx, nt = data.shape
    if padding:
        npx = 2 ** nextpow2(nx)
        npt = 2 ** nextpow2(nt)
    else:
        npx, npt = nx, nt

    # Axes that align with the shifted spectrum
    kk = np.fft.fftshift(np.fft.fftfreq(npx, d=dx))      # [1/m]
    ff = np.fft.fftshift(np.fft.fftfreq(npt, d=dt))      # [Hz]

    # Forward 2D FFT (pad to npx x npt), then center
    F = np.fft.fftshift(np.fft.fft2(data, s=(npx, npt)))

    # Build 2D grids for broadcasting-safe algebra
    # K has shape (npx, npt), W has shape (npx, npt)
    K, W = np.meshgrid(2*np.pi*kk, 2*np.pi*ff, indexing='ij')    # W = ω = 2πf

    # Optional unit conversions in the f-k domain
    # NOTE: These factors assume you're starting from displacement U(k,f).
    #   strain ε <-> U: ε = i k U         (or U = ε / (i k))
    #   velocity v <-> U: v = i ω U       (or U = v / (i ω))
    # If your input F is NOT displacement, adjust the factor accordingly.
    if convert2 == 'strain':
        # ε = (i k / (i ω)) * v   if input was velocity
        # or ε = (i k) * U        if input was displacement
        # Here we implement the ratio you used: multiply by k/ω with a small eps.
        F_conv =  F * ((K+eps) / (W + eps)) / H
    elif convert2 == 'velocity':
        # v = (i ω / (i k)) * U  -> multiply by ω/k with eps
        F_conv = H * F * ((W+eps) / (K + eps))
    else:
        F_conv = F

    kk_Nyq = 1/(2*dx)
    ff_Nyq = 1/(2*dt)
    print(f'Input: dx={dx}, dt={dt}, shape={data.shape}')
    print(f'Nyquist: k={kk_Nyq:.3g} 1/m, f={ff_Nyq:.3g} Hz')
    print(f'F(k,f) shape: {F.shape}')

    return kk, ff, F_conv, (nx, nt)


# Inverse transform to get back t-x data in strain units
def fk_inverse(Fk,
               orig_shape,
               padded=True
               ):
    """
    Inverse 2D FFT (k,f) -> (x,t).
    Fk must be fftshifted (zero at center), like fk_domain returns.
    """
    F_unshift = np.fft.ifftshift(Fk)
    data_padded = np.fft.ifft2(F_unshift)
    data_padded = np.real_if_close(data_padded, tol=1e5)

    if padded:
        nx, nt = orig_shape
        return np.real(data_padded[:nx, :nt])
    else:
        return np.real(data_padded)


def fk_to_fv(
    data: np.ndarray,
    dt: float,
    dx: float,
    v_min: float,
    v_max: float,
    Nv: int = 200,
    normalize_rows: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform data from the time-space domain to the f-v_phase (dispersion)
    domain by remapping the FK spectrum via v = f/k.

    For each positive frequency f, the FK spectrum is interpolated at
    k = f/v for each target velocity v. Contributions from both positive
    (forward) and negative (backward) wavenumbers are summed to maximise
    the signal-to-noise of the dispersion image.

    Args:
        data: Input data, shape (Nt, Nx).
        dt: Temporal sampling period (s).
        dx: Channel spacing (m).
        v_min: Minimum phase velocity (m/s).
        v_max: Maximum phase velocity (m/s).
        Nv: Number of velocity samples. Default 200.
        normalize_rows: If True, normalise each frequency row to [0, 1]
            so that the image brightness is uniform across frequencies.
            Default True.

    Returns:
        f:   Positive frequency axis, shape (Nf,), in Hz.
        v:   Phase velocity axis,     shape (Nv,), in m/s.
        fv:  Dispersion image (power), shape (Nf, Nv).
    """
    Nt, Nx = data.shape

    # --- FK spectrum ---
    spec = fftshift(fft2(data))                   # (Nt, Nx), complex
    f_full = fftshift(fftfreq(Nt, d=dt))          # (Nt,)  Hz
    k_full = fftshift(fftfreq(Nx, d=dx))          # (Nx,)  cycles/m

    # Keep only positive frequencies (spectrum is conjugate-symmetric for
    # real input, so no information is lost)
    pos = f_full > 0
    f   = f_full[pos]                             # (Nf,)
    amp = np.abs(spec[pos, :])                    # (Nf, Nx)

    # --- Velocity axis and output array ---
    v  = np.linspace(v_min, v_max, Nv)            # (Nv,)
    Nf = len(f)
    fv = np.zeros((Nf, Nv), dtype=float)

    # --- Remap k -> v for each frequency row ---
    # np.interp requires k_full to be monotonically increasing, which it is
    # after fftshift (goes from -kNyq to +kNyq).
    for i, fi in enumerate(f):
        k_pos =  fi / v   # forward-propagating:  k > 0
        k_neg = -fi / v   # backward-propagating: k < 0

        row = amp[i, :]
        fv[i, :] = (
            np.interp(k_pos, k_full, row, left=0.0, right=0.0)
            + np.interp(k_neg, k_full, row, left=0.0, right=0.0)
        ) ** 2            # square -> power

    # --- Optional per-frequency normalisation ---
    if normalize_rows:
        row_max = fv.max(axis=1, keepdims=True)
        row_max[row_max == 0] = 1.0               # avoid division by zero
        fv /= row_max

    return f, v, fv


def taup_transform(
    signal: np.ndarray,
    offset: np.ndarray,
    dt: float,
    vmin: float,
    vmax: float,
    np_: int,
    ntau: int,
    ttau: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Linear tau-p (slant-stack) transform followed by a tau→f FFT,
    producing an f-p (frequency-slowness) dispersion image.

    For each (tau, p) cell the signal is sampled at t = tau + p*offset
    over all receivers (linear interpolation) and summed (slant stack).

    Args:
        signal: Data matrix, shape (nt, nr).
        offset: Receiver offsets in metres, shape (nr,).
        dt:     Temporal sampling interval (s).
        vmin:   Minimum phase velocity (m/s).
        vmax:   Maximum phase velocity (m/s).
        np_:    Number of slowness samples.
        ntau:   Number of intercept-time (tau) samples.
        ttau:   Maximum intercept time (s), e.g. 2.0.

    Returns:
        ff:   Frequency axis,    shape (nft,),   Hz.
        p:    Slowness axis,     shape (np_,),    s/m  (high → low).
        fp:   f-p amplitude,     shape (nft, np_).
        tau:  Tau axis,          shape (ntau,),   s.
        taup: Tau-p slant stack, shape (ntau, np_).
    """
    nt, nr = signal.shape

    # --- Slowness axis: high slowness (slow) → low slowness (fast) ---
    pmax = 1.0 / vmin
    pmin = 1.0 / vmax
    p    = np.linspace(pmax, pmin, np_)   # (np_,)

    # --- Intercept-time axis ---
    tau  = np.linspace(0.0, ttau, ntau)   # (ntau,)
    dtau = tau[1] - tau[0]

    # --- Slant stack with linear interpolation (loop over p only) ---
    taup = np.zeros((ntau, np_))
    rec  = np.arange(nr)                  # receiver indices, shape (nr,)

    for ip, pi in enumerate(p):
        # Continuous sample position for every (tau, receiver): (ntau, nr)
        t_cont = (tau[:, np.newaxis] + pi * offset[np.newaxis, :]) / dt

        # Valid mask: within record bounds
        valid  = (t_cont >= 0) & (t_cont <= nt - 1)

        # Linear interpolation weights
        t_cont_clipped = np.clip(t_cont, 0, nt - 1)
        t_lo  = np.floor(t_cont_clipped).astype(int)
        t_hi  = np.minimum(t_lo + 1, nt - 1)
        frac  = t_cont_clipped - t_lo     # fractional part, (ntau, nr)

        # Interpolated amplitude at each (tau, receiver)
        amp = (1.0 - frac) * signal[t_lo, rec] + frac * signal[t_hi, rec]
        amp[~valid] = 0.0

        taup[:, ip] = amp.sum(axis=1)    # sum over receivers

    # --- f-p transform: FFT along tau axis with 8× zero-padding ---
    nft = 8 * ntau
    df  = 1.0 / (dtau * nft)
    ff  = np.arange(nft) * df            # (nft,)

    fp  = np.abs(fft(taup, n=nft, axis=0))   # (nft, np_)

    return ff, p, fp, tau, taup


def compute_spectrogram(
    x,
    fs,
    nperseg=1.0,
    noverlap=int(0.9),
    nfft=None,
    window="hann",
    detrend="constant",
    scaling="density",   # "density" (V^2/Hz) or "spectrum" (V^2)
    mode="psd",          # "psd", "magnitude", "complex", "phase"
    to_db=True,
    ref=1.0,             # reference for dB; if None, uses max of S for 0 dBFS-style scaling
):
    """
    Compute a spectrogram for a 1-D waveform x sampled at fs (Hz).

    Returns
    -------
    f : ndarray
        Frequencies in Hz (size: n_freq)
    t : ndarray
        Time centers in seconds (size: n_time)
    S : ndarray
        Spectrogram (n_freq x n_time). dB-scaled if to_db=True.
    """
    x = np.asarray(x)
    #nperseg = int(round(win_sec * fs))
    #nstep   = int(round(step_sec * fs))
    #noverlap = max(0, nperseg - nstep)

    if nfft is None:
        # Use power-of-two >= nperseg for efficiency
        nfft = 1 << (int(np.ceil(np.log2(max(1, nperseg)))))

    win = get_window(window, nperseg, fftbins=True)

    f, t, S = spectrogram(
        x,
        fs=fs,
        window=win,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=detrend,
        scaling=scaling,
        mode=mode,
    )

    # Convert to magnitude before dB if needed
    if mode == "psd":
        S_lin = S  # already power (V^2/Hz or V^2)
    elif mode == "magnitude":
        S_lin = S**2 if scaling == "density" else S**2  # make it power-like
    elif mode == "complex":
        S_lin = np.abs(S)**2
    elif mode == "phase":
        S_lin = np.abs(np.exp(1j*S))  # not meaningful for dB; keep as is
    else:
        S_lin = S

    if to_db:
        if ref is None:
            ref_val = np.max(S_lin) if np.max(S_lin) > 0 else 1.0
        else:
            ref_val = float(ref)
        # Avoid log of zero
        S = 10.0 * np.log10(np.maximum(S_lin, np.finfo(float).tiny) / ref_val)
    else:
        S = S_lin

    return f, t, S
