"""Loading and saving DAS data and related files (.mat, .hdf5, .csv, .wav)."""
import sys
sys.path.append('C:/Forskning/PythonCodes/simpleDAS')
import simpledas
import numpy as np
import scipy as sp
import scipy.io
import pandas as pd
from scipy.io.wavfile import write

__all__ = [
    "DASmeta",
    "load_resamples_DAS_data",
    "load_Processed_DAS_data",
    "load_Processed_DAS_data_mat",
    "load_mat_files",
    "loadCSV",
    "saveCSV",
    "makeAudioFile",
]


class DASmeta:
    def __init__(self, path2data, dx, dt):
        self.path2data = path2data
        self.dx = dx
        self.dt = dt

    def print_summary(self):
        print('The loaded path + data is ' + self.path2data)
        print(f"dx: {round(self.dx,4)} m")
        print(f"dt: {round(self.dt,4)} s")


def load_resamples_DAS_data(path2data):

    dfdas_reloaded = simpledas.load_DAS_files(path2data)
    meta = dfdas_reloaded.meta

    print(meta['dx'])
    data = dfdas_reloaded
    signalChar = data.to_numpy().T
    print(f'Size of time axis: {signalChar.shape[0]}')
    idx_dx = np.round((meta['dimensionRanges']['dimension1']['max']-meta['dimensionRanges']['dimension1']['min'])/meta['dimensionRanges']['dimension1']['size'])
    if meta['dx'] == 0.0:
        dx_dec1 = 1.0213001907746815 # When there is a bug in resampled hdf5 data
    else:
        dx_dec1 = meta['dx']
    xxChar = np.arange(meta['dimensionRanges']['dimension1']['min'], meta['dimensionRanges']['dimension1']['max']+idx_dx, idx_dx)*dx_dec1 # Create the distance axis in meters
    dx = np.mean(np.diff(xxChar)) # Redfine actual dx
    dt = meta['dt']
    startTime = meta['time']
    ttAbs = data.index.to_numpy()
    _arr = ttAbs[0]
    _ts = pd.to_datetime(_arr).round('s')
    hh, mm, ss = _ts.hour, _ts.minute, _ts.second
    ttChar = np.arange(0,len(ttAbs),1)*dt

    return signalChar, meta, dx, dt, startTime, ttAbs, ttChar, xxChar, hh, mm, ss


def load_Processed_DAS_data(path2data):
    """
    This function load various file formats in which DAS data has been loaded, processed and saved

    Parameters:
    path2data (string): The path with filename to the loaded data.

    Returns:
    date (numpy matrix of size (nx, nt)): The loaded data
    meta (Class): metadata of the loaded data
    """
    # Find extension in order to load the data correctly
    fileformat = path2data.split('.')[-1]

    if fileformat == 'mat':
        data = sp.io.loadmat(path2data)
        meta = DASmeta(path2data,np.mean(np.diff(data['xx_abs'])), 10)
        meta.print_summary()
    elif fileformat == 'hdf5':
        dfdas_reloaded = simpledas.load_DAS_files(path2data)
        meta = dfdas_reloaded.meta
        data = dfdas_reloaded

    return data, meta, fileformat


def load_mat_files(path2data):
    """
    This function load mat files

    Parameters:
    path2data (string): The path with filename to the loaded data.

    Returns:
    date (numpy matrix of size (nx, nt)): The loaded data
    meta (Class): metadata of the loaded data
    """
    data = sp.io.loadmat(path2data)
    return data


def loadCSV(path2csv,filename,nHeader,delimiter):
    if nHeader == 0:
        df = pd.read_csv(path2csv + filename, header=None, delimiter=delimiter)
    else:
        df = pd.read_csv(path2csv + filename, header=nHeader, delimiter=delimiter)

    return df


def saveCSV(path2csv,filename,data,delimiter):
    """
    Function to save a DataFrame to a CSV file.

    Parameters:
    path2csv (str): The path where the CSV file should be saved.
    filename (str): The name of the CSV file to be saved.
    data (DataFrame): The DataFrame to be saved.
    delimiter (str): The delimiter to use in the CSV file.

    Returns:
    None
    """
    data.to_csv(path2csv + filename, index=False, sep=delimiter)


def makeAudioFile(signal,
                  Fs,
                  path2save,
                  fileName,
                  nMethod
                  ):
    """
    Function to save a signal as an audio file.

    Parameters:
    signal (numpy array): The input signal to be saved as an audio file.
    Fs (float): The sample rate of the signal [Hz].
    path2save (str): The path where the audio file should be saved.
    fileName (str): The name of the audio file to be saved.
    nMethod (int): Three methods available that should provide the same result:
        1. scipy.io.wavfile.write
        2. soundfile.write
        3. wave module

    Returns:
    None
    """
    output_filename = path2save + fileName
    signal_normalized_4audio = signal / np.max(np.abs(signal))  # Normalize the signal to the range [-1, 1]

    if nMethod == 1:
        # Convert to 16-bit PCM format
        signal_audio = np.int16(signal_normalized_4audio * 32767) # 32767 is the maximum value for 16-bit signed integers, use this
        # Save as WAV file
        write(output_filename, int(Fs), signal_audio)

    elif nMethod == 2:
        # Imported here so the rest of DASFunctions works without soundfile installed
        import soundfile as sf
        sf.write(output_filename, signal_normalized_4audio, int(Fs), subtype='PCM_16')

    elif nMethod == 3:
        import wave
        signal_audio = np.int16(signal_normalized_4audio * 32767) # 32767 is the maximum value for 16-bit signed integers, use this
        with wave.open(output_filename, 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16 bits = 2 bytes
            wf.setframerate(int(Fs))
            wf.writeframes(signal_audio.tobytes())
