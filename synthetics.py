"""Synthetic signals and travel-time curves for testing and modelling."""
import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "ricker_wavelet",
    "sine_wave_two_frequencies",
    "das_hyperbola",
]


def ricker_wavelet(f, lengthTime, dt, t0):
    t = np.arange(0, (lengthTime / 1) + dt, dt)
    y = (1 - 2 * (np.pi**2) * (f**2) * ((t-t0)**2)) * np.exp(-(np.pi**2) * (f**2) * ((t-t0)**2))
    return t, y


# Function generating one sin wavelet with two frequencies for each period
def sine_wave_two_frequencies(T1, T2, dt):
    #T1 = 1
    #T2 = T1*1.2
    A1, A2 = 1.0, T1 / T2  # equal area condition
    # Time vectors
    t1 = np.arange(0, T1, dt) #np.linspace(0, T1, 1000)
    t2 = np.arange(0, T2, dt) #np.linspace(0, T2, 1000)
    dt = np.mean(np.diff(t1))
    # Signals
    y1 = A1 * np.sin(2 * np.pi * t1 / T1)
    y2 = A2 * np.sin(2 * np.pi * t2 / T2)

    # Combine
    idxPos = np.where(y1 >= 0)[0]
    idxNeg = np.where(y2 <= 0)[0]

    yy = np.concatenate([y1[idxPos], y2[idxNeg[1]+1:]])
    tt = np.arange(0, len(yy)) * dt

    # Plot
    plt.plot(t1, y1,'b')
    plt.plot(t2, y2,'r')
    plt.plot(tt, yy,'k')

    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Sine waves with equal area under each half-cycle')
    plt.grid()
    plt.show()
    return tt, yy


def das_hyperbola(
    velocity: float,
    distance: float,
    x_array: np.ndarray,
    branch: str = "both",
) -> np.ndarray:
    """
    Generate a hyperbola for DAS data (e.g. from a point diffractor or passing source).

    The hyperbola follows:
        t(x) = sqrt(t0^2 + ((x - x0) / velocity)^2)

    where t0 = distance / velocity is the two-way (or one-way) travel time at the apex.

    Parameters
    ----------
    velocity : float
        Propagation velocity [m/s]. Controls the opening angle of the hyperbola.
        Higher velocity means flatter/wider hyperbola.
    distance : float
        Perpendicular distance from the fibre to the source [m].
        Used to compute t0 if t0 is passed as None (see below).
        Also stored for reference.
    x_array : np.ndarray
        1-D array of x positions (channel positions) at which to evaluate t [m].
    branch : str
        Which branch(es) to return:
        - "both"  → shape (2, len(x_array)), rows are [positive_branch, negative_branch]
        - "pos"   → shape (len(x_array),), the positive (later-arriving) branch only
        - "neg"   → shape (len(x_array),), the negative branch (same values, for symmetry)

    Returns
    -------
    np.ndarray
        Travel times [s] at each x position. NaN where the hyperbola is not defined.
    """
    # Allow t0 to be derived from distance and velocity
    t0 = distance / velocity
    t = np.sqrt(t0**2 + (x_array / velocity) ** 2)

    if branch == "pos":
        return t
    elif branch == "neg":
        return -t
    elif branch == "both":
        return np.stack([t, -t], axis=0)
    else:
        raise ValueError(f"branch must be 'pos', 'neg', or 'both', got '{branch}'")
