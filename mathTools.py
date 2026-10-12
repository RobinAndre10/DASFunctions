"""
Backwards-compatibility module.

The functions that used to live here are now split by topic:

    das_io.py         loading / saving data
    tapers.py         time-domain windows and tapers
    transforms.py     FFT, f-x, f-k, f-v, tau-p, spectrograms
    filters.py        Butterworth / Ormsby / simple f-k filters, AGC
    processing.py     demean, RMS, strain <-> strain rate, gauge length, ...
    curve_fitting.py  smoothing and robust polynomial fits
    detection.py      channel-by-channel event detection
    geo.py            lat/lon <-> UTM conversions, basemaps
    plotting.py       plotting helpers
    synthetics.py     synthetic wavelets and travel-time curves

Old scripts doing ``from DASFunctions.mathTools import ...`` keep working.
New code should import from the topic modules directly, e.g.
``from DASFunctions.filters import filter_butterworth``.
"""
from .das_io import *
from .tapers import *
from .transforms import *
from .filters import *
from .processing import *
from .curve_fitting import *
from .detection import *
from .geo import *
from .plotting import *
from .synthetics import *
from .geo import _pick_utm_zone_norway_svalbard
