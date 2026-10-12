# DASFunctions

Functions for loading, processing and plotting DAS data.

```python
import sys
sys.path.append('C:/Forskning/Python')
from DASFunctions.filters import filter_butterworth
from DASFunctions.transforms import fk_domain
```

`mathTools.py` re-exports everything below, so old `from DASFunctions.mathTools import ...` lines still work.

## `das_io.py`

Loading and saving DAS data and related files (.mat, .hdf5, .csv, .wav).

- `DASmeta`
- `load_resamples_DAS_data`
- `load_Processed_DAS_data` — This function load various file formats in which DAS data has been loaded, processed and saved
- `load_Processed_DAS_data_mat` — This function load various file formats in which DAS data has been loaded, processed and saved
- `load_mat_files` — This function load mat files
- `loadCSV`
- `saveCSV` — Function to save a DataFrame to a CSV file.
- `makeAudioFile` — Function to save a signal as an audio file.

## `tapers.py`

Time-domain windows and tapers.

- `timeDomainWindow` — Function to generate a time domain window to be used in further, e.g., frequency domain operations.
- `fade_in_taper` — Apply a fade-in taper to the first n_taper samples of x.

## `transforms.py`

Spectral transforms: FFT, f-x, f-k, f-v, tau-p and spectrograms.

- `nextpow2` — Returns the exponent of the next power of 2 greater than or equal to n. Often used to improve the fft function.
- `my_fft` — Compute amplitude, phase, and optionally power spectral density (PSD)
- `my_fft_OLD` — Compute amplitude, phase, and optionally power spectral density (PSD)
- `fx_domain` — Function to transform the input 2-dimensional data to the f-x domain.
- `fk_domain` — Forward 2D FFT (x,t) -> (k,f), with optional zero-padding to next power of two.
- `fk_domain_conversion` — Forward 2D FFT (x,t) -> (k,f), with optional zero-padding to next power of two.
- `fk_inverse` — Inverse 2D FFT (k,f) -> (x,t).
- `fk_to_fv` — Transform data from the time-space domain to the f-v_phase (dispersion)
- `taup_transform` — Linear tau-p (slant-stack) transform followed by a tau→f FFT,
- `compute_spectrogram` — Compute a spectrogram for a 1-D waveform x sampled at fs (Hz).

## `filters.py`

Frequency filters (Butterworth, Ormsby, simple f-k) and gain control.

- `filter_butterworth` — Filter the input data in the time domain using a specified filter type and cutoff frequency.
- `filter_butterworth_vClaude` — Taper then Butterworth-filter `data` along `axis`.
- `filter_ormsby_vClaude` — Taper then Ormsby-filter `data` along `axis`.
- `fk_filter`
- `agc` — Apply automatic gain control to 2D data.
- `filter_time_domain` — Deprecated name for filter_butterworth, kept so older scripts still run.

## `processing.py`

Basic DAS data processing: demeaning, RMS, unit conversion,

- `demean_array` — Subtract the mean value from each element in the data array.
- `compute_RMS` — Compute the root mean square (RMS) of the input data along a specified axis.
- `strain2strainRate` — Convert strain data to strain rate data by temporal differentiation.
- `strainRate2strain` — Convert strain rate data to strain data by temporal integration.
- `numerical_integration` — Convert particle velocity to displacement by time integration.
- `changeGaugeLength` — Synthetically increase the gauge length of DAS strain(-rate) data by
- `changeGaugeLength_OLD` — INPUT:
- `concatenate_DAS_timeaxis` — Function that will concatenate DAS data along the time axis.

## `curve_fitting.py`

Smoothing and robust fitting of 1-D curves.

- `smooth_curve` — Smooth a 1D curve using several possible methods.
- `robust_polyfit` — Robust polynomial fit using statsmodels.RLM.

## `detection.py`

Event detection on DAS channels.

- `channelBychannel_detection` — Function for extracting one detection for one trace.

## `geo.py`

Coordinate conversions (lat/lon <-> UTM, Norway/Svalbard zones) and map tiles.

- `geocoord2utmcoord` — Simple script converting latitude longitude pair to utm pair.
- `utmcorrd2geocoord` — Simple script converting utm pair to latitude longitude pair.
- `computeDistanceFrom_Lat_Lon` — Script computing the distance between point and accumulated distances based on Lat, Lon vectors.
- `latlon_to_utm_svalbard` — Converts latitude/longitude to UTM coordinates in the correct Svalbard zone.
- `latlon_to_utm_no` — Convert latitude/longitude (WGS84) to UTM easting/northing/zone
- `utm_to_latlon` — Convert UTM (easting, northing, zone) to latitude/longitude (WGS84).
- `get_basemap` — Returns a Cartopy tile source (img_tiles) for a given basemap name.

## `plotting.py`

Plotting helpers for DAS sections, spectrograms and wiggle traces.

- `plot_das_strain_section`
- `plot_spectrogram`
- `wiggle`

## `synthetics.py`

Synthetic signals and travel-time curves for testing and modelling.

- `ricker_wavelet`
- `sine_wave_two_frequencies`
- `das_hyperbola` — Generate a hyperbola for DAS data (e.g. from a point diffractor or passing source).

## `fk_filter.py`

Frequency-wavenumber (f-k) band / fan filter for 2-D gathers.

- `tukey` — Tukey (cosine-tapered) window of length ``n``.
- `fk_mask` — Build the (fftshift-ed) f-k mask for a gather of ``shape = (Nt, Nx)``.
- `fk_filter` — Apply an f-k band / fan filter to a 2-D gather.

## `extract_along_velocity.py`

Extract amplitudes from an f-k spectrum along a line of constant apparent

- `velocity_resolution` — Velocity smear from one wavenumber sample: dv = v^2 / (f * L).
- `extract_along_velocity` — Amplitude along the line k = -f/v (or +f/v) through an f-k spectrum.
- `fv_panel` — (n_f, n_v) amplitude panel -- one column per velocity.
- `plot_extraction` — QC figure: f-k panel with the extraction line, plus the spectrum.
