"""Event detection on DAS channels."""
import numpy as np

__all__ = [
    "channelBychannel_detection",
]


def channelBychannel_detection(trace,
                               typeAlgo,
                               nSTA,
                               nLTA
                               ):
    """
    Function for extracting one detection for one trace.

    Parameters:
    srace (1d numpy array): Time series signal.
    typeAlgo: Type of channel by channel detection algorithm
        stalta: sta/lta algorithm by obspy. nSTA and nLTA needs to defined for this one.
                STA must be longer than few periods of the main frequency but shorter than the shortest events and longer than potential spikes
                LT should be longer than a few periods of typically irregular seisic noise fluctuations. Typically an order of magnitude larger than the STA duration
                e.g., LTA = STA*10
        max4trace: Find the index of the maximum energy in trace

    Return:
    idxTriggerTime: Index along the time axis for the detection.
    """

    if typeAlgo == 'stalta':
        cft = recursive_sta_lta( trace, nSTA, nLTA)
        idxTriggerTime = np.argmax(np.abs(cft))
    elif typeAlgo == 'max4trace':
        idxTriggerTime = np.argmax(np.abs(trace))

    return idxTriggerTime
