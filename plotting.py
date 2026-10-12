"""Plotting helpers for DAS sections, spectrograms and wiggle traces."""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

__all__ = [
    "plot_tx",
    "plot_fk",
    "save_figure",
    "plot_das_strain_section",
    "plot_spectrogram",
    "wiggle",
]

def plot_tx(data_xt, x, t_s, ax=None, t0=None, timeUnit = 'sec',
            xlimVal=None, ylimVal=None, climVal=(-1e-9,1e-9),
            title='t-x plot', figSize = (11.69, 8.27),  scale=1.0,
            cmap='bwr', colorbar=True, clrLabel='Strain'):
    """
    Waterfall plot of a (channel, time) gather, time running downwards.

    Draws into ``ax`` if given (e.g. one panel of a subplot grid), otherwise
    creates a new figure.

    Parameters
    ----------
    data_xt : (nx, nt) ndarray
        Data with channels along axis 0 and time along axis 1.
    x : (nx,) ndarray
        Distance along the fibre [m].
    t_s : (nt,) ndarray
        Time [s], relative to the start of the record. Converted to ``timeUnit`` for plotting.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. ``axs[1, 0]`` from ``fig, axs = plt.subplots(2, 2)``.
        Default: a new figure of size ``figSize``.
    t0 : datetime-like or str, optional
        Absolute start time; if given, the y-label reads 'Time (<unit>) after {t0}'.
    timeUnit : str, default 'sec'
        Unit of the time axis: seconds ('s', 'sec', ...), minutes ('min', ...) or
        hours ('h', 'hr', ...). Raises ValueError for anything else.
    xlimVal : (float, float), optional
        Distance limits (left, right) [m]. Default: the full range of ``x``.
    ylimVal : (float, float), optional
        Time limits (bottom, top) in ``timeUnit``; give the later time first to keep
        time running downwards, e.g. ``(30, 0)``. Default: the full range of ``t_s``.
    climVal : (float, float), default (-1e-9, 1e-9)
        Colour limits (min, max), in the units after dividing by ``scale``.
    title : str, default 't-x plot'
        Axes title.
    figSize : (float, float), default (11.69, 8.27)
        Figure size in inches when a new figure is created (ignored if ``ax`` is given);
        the default is A4 landscape.
    scale : float, default 1.0
        The data are plotted as ``data_xt / scale``, e.g. ``scale=1e-9`` to plot in
        nano-units (then also set ``climVal`` in nano-units).
    cmap : str, default 'bwr'
        The colormap to be used in the plotting.
    colorbar : bool, default True
        Add a colour bar next to ``ax``. Set False to add one shared bar for a
        subplot grid yourself.
    clrLabel : str, default 'Strain'
        Colour-bar label, e.g. 'Strain rate (nε/s)'.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The plotted image. Get the axes with ``im.axes`` and the figure with
        ``im.axes.figure``; use ``im`` for a shared colour bar or ``im.set_clim(...)``.
    """
    unit = timeUnit.lower()
    if unit in {'sec','s','second','seconds'}:
        t, unitLabel = t_s, 's'
    elif unit in {'min','minute','minutes'}:
        t, unitLabel = t_s / 60, 'min'
    elif unit in {'h','hr','hrs','hour','hours'}:
        t, unitLabel = t_s / (60*60), 'h'
    else:
        raise ValueError(f"Unknown timeUnit {timeUnit!r}; use seconds, minutes or hours")

    if ax is None:
        _, ax = plt.subplots(figsize=figSize)

    im = ax.imshow(data_xt.T / scale, extent=[x.min(), x.max(), t.max(), t.min()],
                   aspect='auto', 
                   cmap=cmap,
                   vmin=climVal[0], vmax=climVal[1])
    if colorbar: 
        ax.figure.colorbar(im, ax=ax, label=clrLabel)
        
    ax.set_xlim( (x.min(), x.max()) if xlimVal is None else xlimVal )
    ax.set_ylim( (t.max(), t.min()) if ylimVal is None else ylimVal )
    ax.set_xlabel('Distance along fibre (m)')
    ax.set_ylabel(f'Time ({unitLabel})' if t0 is None else f'Time ({unitLabel}) after {t0}')
    ax.set_title(title)
    return im

def plot_fk(amp, kk, ff, climVal, ax=None, xlimVal=None, ylimVal=None,
            xlabel='Wavenumber (1/m)', ylabel='Frequency (Hz)', title='f-k plot', figSize = (11.69, 8.27),
            cmap='bwr', colorbar=True, clrLabel='Amplitude (dB)'):
    """
    Plot an f-k amplitude spectrum in dB.

    Draws into ``ax`` if given (e.g. one panel of a subplot grid), otherwise
    creates a new figure.

    Parameters
    ----------
    amp : (nk, nf) ndarray
        Linear f-k amplitude spectrum, wavenumber along axis 0 and frequency along
        axis 1, e.g. the ``amp`` output of ``fk_domain``. Converted to 20*log10(amp).
    kk : (nk,) ndarray
        Wavenumber axis [1/m], fftshifted (increasing), matching axis 0 of ``amp``.
    ff : (nf,) ndarray
        Frequency axis [Hz], fftshifted (increasing), matching axis 1 of ``amp``.
    climVal : (float, float)
        Colour limits (min, max) in dB.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on, e.g. ``axs[0, 1]`` from ``fig, axs = plt.subplots(2, 2)``.
        Default: a new figure of size ``figSize``.
    xlimVal : (float, float), optional
        Wavenumber limits [1/m]. Default: the full range of ``kk``.
    ylimVal : (float, float), optional
        Frequency limits [Hz], e.g. ``(0, 2)`` to show only low, positive frequencies.
        Default: the full range of ``ff``, negative frequencies included.
    xlabel : str, default 'Wavenumber (1/m)'
        x-axis label.
    ylabel : str, default 'Frequency (Hz)'
        y-axis label.
    title : str, default 'f-k plot'
        Axes title.
    figSize : (float, float), default (11.69, 8.27)
        Figure size in inches when a new figure is created (ignored if ``ax`` is given).
    cmap : str, default 'bwr'
            The colormap to be used in the plotting.
    colorbar : bool, default True
        Add a colour bar next to ``ax``. Set False to add one shared bar for a
        subplot grid yourself.
    clrLabel : str, default 'Amplitude (dB)'
        Colour-bar label.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The plotted image. Get the axes with ``im.axes`` and the figure with
        ``im.axes.figure``; use ``im`` for a shared colour bar or ``im.set_clim(...)``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figSize)
    # Tiny offset so exact zeros give a very low dB value instead of -inf and a warning
    im = ax.imshow(20 * np.log10(amp + 1e-30).T,
                   extent=[kk.min(), kk.max(), ff.max(), ff.min()],
                   aspect='auto', 
                   cmap=cmap,
                   vmin=climVal[0], vmax=climVal[1]
                   )
    if colorbar:
        ax.figure.colorbar(im, ax=ax, label=clrLabel)

    ax.set_xlim((kk.min(), kk.max()) if xlimVal is None else xlimVal)
    ax.set_ylim((ff.min(), ff.max()) if ylimVal is None else ylimVal)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return im


def save_figure(fig, path, dpi=200):
    """
    Save a figure, creating the output folder if needed.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or pathlib.Path
        Output file; the extension sets the format (.png, .pdf, .svg, ...).
    dpi : int, default 200
        Resolution for raster formats such as .png.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')


def plot_das_strain_section(
    signal: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    xlabel: str = 'x-axis',
    ylabel: str = 'y-axis',
    xlimVal: Optional[Tuple[float, float]] = None,
    ylimVal: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
    colorbar: bool = False,
    climVal: Tuple[float, float] = (-1e-9, 1e-9),
    cmap: str = 'bwr',
    cbarlabel: str = 'colorbar label',
    figsize: Tuple[float, float] = (11.69, 8.27),
    subplotSize: Tuple[int, int] = (1,1),
):

    fig, axT= plt.subplots(subplotSize[0], subplotSize[1], figsize=figsize, squeeze=False)
    ax = axT[0,0]
    im = ax.imshow(
        signal.T ,
        extent=[xx.min(),xx.max(),yy.max(),yy.min()],
        aspect='auto',
        cmap=cmap
        )

    if colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbarlabel, fontsize=14, fontweight='bold')
    im.set_clim(climVal)

    ax.set_xlabel(
        xlabel,
        fontdict={'fontsize': 14, 'fontweight': 'bold', 'color':'black'}
        )
    ax.set_ylabel(
        ylabel,
        fontdict={'fontsize': 14, 'fontweight': 'bold', 'color':'black'}
        )
    if title is not None:
        ax.set_title(
            title,
            fontdict={'fontsize': 14, 'fontweight': 'bold', 'color':'black'}
            )
    if xlimVal is not None:
        ax.set_xlim(xlimVal[0],xlimVal[1])
    if ylimVal is not None:
        ax.set_ylim(ylimVal[0],ylimVal[1])

    ax.tick_params(axis='both', labelsize=14)

    return fig, ax


def plot_spectrogram(
    f, t, S,
    figsize=(11.69, 8.27),          # A4 landscape
    cmap="viridis",
    vmin=None, vmax=None,
    title="Spectrogram",
    ylabel="Frequency (Hz)",
    xlabel="Time (s)",
    add_colorbar=True
):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(
        S,
        aspect="auto",
        origin="lower",
        extent=[t[0], t[-1], f[0], f[-1]],
        cmap=cmap,
        vmin=vmin, vmax=vmax,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("dB" if "Spectrogram" in title or np.nanmean(S) < 0 else "Amplitude")
    plt.tight_layout()
    return fig, ax


def wiggle(xx,
           yy,
           offset,
           clrLine,
           clrFill
           ):
    plt.plot(xx,yy,'-',color=clrLine) # Normal wigigle
    plt.fill_betweenx(yy,offset,xx,where=(xx>=offset),color=clrFill) # Fill positive valuesc
