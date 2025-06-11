#import sys
#sys.path.append('C:/Forskning/PythonCodes/simpleDAS')
#import simpledas
#import numpy as np
#import scipy as sp
#import matplotlib.pyplot as plt
#import pandas as pd


def testFun(n):
    n2 = 2*n
    return n2

def nextpow2(n):
    """
    Returns the exponent of the next power of 2 greater than or equal to n. Often used to improve the fft function.
    """
    return np.ceil(np.log2(n)).astype(int)

def my_fft(signal, Fs, padding):
    """
    Function to find the index of the nearest value in an given array. 
    ### INPUTS: ###
    signal is a (nt,) vector containing a time seris. 
    Fs is the sample rate of the signal. 
    padding is a boolean that decides if the signal should be padded with zeros before the fft, default to the next power of two.
    ### OUTPUTS: ###
    SIGNAL is the fft of the signal. It has been fft shifted such that it is centered around zero.
    ff is the frequency vector of the signal containing both positive and negative spectral components.
    ffOnlyPos is the frequency vector of the signal containing only positive spectral components.
    """
    if padding == True:
        SIGNAL = np.fft.fft(signal,2**nextpow2(len(signal)))
    else:
        SIGNAL = np.fft.fft(signal,len(signal))
    # FFt shift the signal such that it clearly shows the negative and positive frequencies
    SIGNAL = np.fft.fftshift(SIGNAL)     
    # Define frequency vector
    ff = Fs * np.arange(-len(SIGNAL)/2,len(SIGNAL)/2,1) / len(SIGNAL) # Based on the sample rate Fs and the amount of data points in the fft'ed signal - define the frequency vector
    #ffOnlyPos = Fs * np.arange(len(SIGNAL)/2) / len(SIGNAL)
    return SIGNAL, ff

def timeDomainWindow(type, length, alpha):
    """
    Function to generate a time domain window to be used in further, e.g., frequency domain operations. 
    
    Parameters:
    type is a string that specifies the type of window to be generated. 
    length is the length of the window to be generated. 
    alpha is the taper rate of the window, only applicable for type: 'tukey' .
    
    Returns:
    window is the window of the specified type and length. 
    """
    if type == 'tukey':
        window = sp.signal.windows.tukey(length, alpha)
    elif type == 'hamming':
        window = sp.signal.windows.hamming(length)
    elif type == 'hanning':
        window = sp.signal.windows.hanning(length)
    elif type == 'blackman':
        window = sp.signal.windows.blackman(length)

    return window


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

def filter_time_domain(data, filterOrder, filter_type, cutoff_freq, Fs, tprRate):
    """
    Filter the input data in the time domain using a specified filter type and cutoff frequency.
    
    Parameters:
    data (numpy array): The input data array to be filtered.
    filterOrder (int): The order of the filter to be used.
    filter_type (str): The type of filter to be used. Options: 'lowpass', 'highpass', 'bandpass', 'bandstop
    cutoff_freq (float or list): The cutoff frequency of the filter. If filter_type is 'bandpass' or 'bandstop', this should be a list of two values - otherwise it should be one value.
    Fs (float): The sample rate of the data [Hz].

    Return:
    data_filtered (numpy array): The filtered data array.
    """

    # ### Window data
    window = timeDomainWindow('tukey', len(data), tprRate)

    # Need to flip the signal at the end to be equal to the length of the non-one values in the windo
    # Apply the window to the signal
    data = data * window

    # ### High-pass filter ###
    if filter_type == 'highpass' or filter_type == 'hp':
        # Method one - same as matlab version
        #bHighcut, aHighcut = sp.signal.butter(order, cutFreq_hp / (Fs / 2), btype='highpass')
        #singal_HPFiltered = sp.signal.filtfilt(bHighcut, aHighcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_HP = sp.signal.butter(filterOrder, cutoff_freq,  btype='highpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_HP, data)

    elif filter_type == 'lowpass' or filter_type == 'lp':        
        # Method one - same as matlab version
        #bLowcut, aLowcut = sp.signal.butter(order, cutFreq_Lp / (Fs / 2), btype='lowpass')
        #singal_LPFiltered = sp.signal.filtfilt(bLowcut, aLowcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_LP = sp.signal.butter(filterOrder, cutoff_freq,  btype='lowpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_LP, data)

    elif filter_type == 'bandpass' or filter_type == 'bp':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        # Method one - same as matlab version
        #bBandcut, aBandcut = sp.signal.butter(order, cutFreq_bp / (Fs / 2), btype='bandpass')
        #singal_BPFiltered = sp.signal.filtfilt(bBandcut, aBandcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_BP = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_BP, data)
        
    elif filter_type == 'bandstop' or filter_type == 'bs':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        # Method one - same as matlab version
        #bBandStopcut, aBandStopcut = sp.signal.butter(order, cutFreq_bp / (Fs / 2), btype='bandstop')
        #singal_BSFiltered = sp.signal.filtfilt(bBandStopcut, aBandStopcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_BS = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandstop', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_BS, data)
    
    # Function done, return wanted value(s)
    return data_filtered

class DASmeta:
    def __init__(self, path2data, dx, dt):
        self.path2data = path2data
        self.dx = dx
        self.dt = dt

    def print_summary(self):
        print('The loaded path + data is ' + path2data)
        print(f"dx: {round(self.dx,4)} m")
        print(f"dt: {round(self.dt,4)} s")

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
    #print(fileformat)  # Output: 'mat'

    if fileformat == 'mat':
       # print(path2data)
        data = sp.io.loadmat(path2data)
        meta = DASmeta(path2data,np.mean(np.diff(data['xx_abs'])), 10)
        meta.print_summary()
    elif fileformat == 'hdf5':
        dfdas_reloaded = simpledas.load_DAS_files(path2data)
        meta = dfdas_reloaded.meta                                      
        data = dfdas_reloaded

    return data, meta, fileformat

def wiggle(xx,yy,offset,clrLine,clrFill):
    plt.plot(xx,yy,'-',color=clrLine) # Normal wigigle    
    plt.fill_betweenx(yy,offset,xx,where=(x>=offset),color=clrFill) # Fill positive valuesc
import sys
sys.path.append('C:/Forskning/PythonCodes/simpleDAS')
import simpledas
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd


def nextpow2(n):
    """
    Returns the exponent of the next power of 2 greater than or equal to n. Often used to improve the fft function.
    """
    return np.ceil(np.log2(n)).astype(int)

def my_fft(signal, Fs, padding):
    """
    Function to find the index of the nearest value in an given array. 
    ### INPUTS: ###
    signal is a (nt,) vector containing a time seris. 
    Fs is the sample rate of the signal. 
    padding is a boolean that decides if the signal should be padded with zeros before the fft, default to the next power of two.
    ### OUTPUTS: ###
    SIGNAL is the fft of the signal. It has been fft shifted such that it is centered around zero.
    ff is the frequency vector of the signal containing both positive and negative spectral components.
    ffOnlyPos is the frequency vector of the signal containing only positive spectral components.
    """
    if padding == True:
        SIGNAL = np.fft.fft(signal,2**nextpow2(len(signal)))
    else:
        SIGNAL = np.fft.fft(signal,len(signal))
    # FFt shift the signal such that it clearly shows the negative and positive frequencies
    SIGNAL = np.fft.fftshift(SIGNAL)     
    # Define frequency vector
    ff = Fs * np.arange(-len(SIGNAL)/2,len(SIGNAL)/2,1) / len(SIGNAL) # Based on the sample rate Fs and the amount of data points in the fft'ed signal - define the frequency vector
    #ffOnlyPos = Fs * np.arange(len(SIGNAL)/2) / len(SIGNAL)
    return SIGNAL, ff

def timeDomainWindow(type, length, alpha):
    """
    Function to generate a time domain window to be used in further, e.g., frequency domain operations. 
    
    Parameters:
    type is a string that specifies the type of window to be generated. 
    length is the length of the window to be generated. 
    alpha is the taper rate of the window, only applicable for type: 'tukey' .
    
    Returns:
    window is the window of the specified type and length. 
    """
    if type == 'tukey':
        window = sp.signal.windows.tukey(length, alpha)
    elif type == 'hamming':
        window = sp.signal.windows.hamming(length)
    elif type == 'hanning':
        window = sp.signal.windows.hanning(length)
    elif type == 'blackman':
        window = sp.signal.windows.blackman(length)

    return window


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

def filter_time_domain(data, filterOrder, filter_type, cutoff_freq, Fs, tprRate):
    """
    Filter the input data in the time domain using a specified filter type and cutoff frequency.
    
    Parameters:
    data (numpy array): The input data array to be filtered.
    filterOrder (int): The order of the filter to be used.
    filter_type (str): The type of filter to be used. Options: 'lowpass', 'highpass', 'bandpass', 'bandstop
    cutoff_freq (float or list): The cutoff frequency of the filter. If filter_type is 'bandpass' or 'bandstop', this should be a list of two values - otherwise it should be one value.
    Fs (float): The sample rate of the data [Hz].

    Return:
    data_filtered (numpy array): The filtered data array.
    """

    # ### Window data
    window = timeDomainWindow('tukey', len(data), tprRate)

    # Need to flip the signal at the end to be equal to the length of the non-one values in the windo
    # Apply the window to the signal
    data = data * window

    # ### High-pass filter ###
    if filter_type == 'highpass' or filter_type == 'hp':
        # Method one - same as matlab version
        #bHighcut, aHighcut = sp.signal.butter(order, cutFreq_hp / (Fs / 2), btype='highpass')
        #singal_HPFiltered = sp.signal.filtfilt(bHighcut, aHighcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_HP = sp.signal.butter(filterOrder, cutoff_freq,  btype='highpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_HP, data)

    elif filter_type == 'lowpass' or filter_type == 'lp':        
        # Method one - same as matlab version
        #bLowcut, aLowcut = sp.signal.butter(order, cutFreq_Lp / (Fs / 2), btype='lowpass')
        #singal_LPFiltered = sp.signal.filtfilt(bLowcut, aLowcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_LP = sp.signal.butter(filterOrder, cutoff_freq,  btype='lowpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_LP, data)

    elif filter_type == 'bandpass' or filter_type == 'bp':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        # Method one - same as matlab version
        #bBandcut, aBandcut = sp.signal.butter(order, cutFreq_bp / (Fs / 2), btype='bandpass')
        #singal_BPFiltered = sp.signal.filtfilt(bBandcut, aBandcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_BP = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandpass', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_BP, data)
        
    elif filter_type == 'bandstop' or filter_type == 'bs':
        if len(cutoff_freq) != 2:
            raise ValueError("The cutoff frequency must be a list of two values.")
        # Method one - same as matlab version
        #bBandStopcut, aBandStopcut = sp.signal.butter(order, cutFreq_bp / (Fs / 2), btype='bandstop')
        #singal_BSFiltered = sp.signal.filtfilt(bBandStopcut, aBandStopcut, singalWindowed)

        # Method two, supposed to be for general use
        sos_BS = sp.signal.butter(filterOrder, cutoff_freq,  btype='bandstop', analog=False, output = 'sos', fs = Fs)
        data_filtered = sp.signal.sosfilt(sos_BS, data)
    
    # Function done, return wanted value(s)
    return data_filtered


def fx_domain(data, Fs, padding, tprRate):
    """
    Function to transform the input 2-dimensional data to the f-x domain.
    Parameters:
    data (numpy matrix, nx X nt): The input 2D-data matrix to be transformed.
    Fs (float): The sample rate of the data [Hz].
    padding (boolean): A boolean that decides if the signal should be padded with zeros before the fft, default to the next power of two.
    Return:
    signal_FXdom (numpy matrix, nx X nt): The transformed data in the f-x domain.
    """
    window = timeDomainWindow('tukey', data.shape[1], tprRate)

    if padding == True:
        signal_FXdom = np.zeros( (np.int32(data.shape[0]), np.int32(2**nextpow2(data.shape[1]))), dtype=float )

        for i in range(data.shape[0]):    
            print(f'Working on channel {i+1} of {data.shape[0]}')
            signal_FXdom[i,:], ff = my_fft(data[i,:]*window, Fs, True)
    else:
        signal_FXdom = np.zeros( (np.int32(data.shape[0]), np.int32(data.shape[1])), dtype=float )

        for i in range(data.shape[0]):    
            print(f'Working on channel {i+1} of {data.shape[0]}')
            signal_FXdom[i,:], ff = my_fft(data[i,:]*window, Fs, False)

    # Compute amplitude spectrum
    #amp_signal_FXdom = np.abs(signal_FXdom.T)

    return signal_FXdom, ff

def fk_domain(data, dx, dt, padding):
    """
    Function that transform input data of size nx X nt (distance X time) to the f-k domain using a 2-dimensional fft.
    Parameters:
    data (numpy matrix, nx X nt): The input 2D-data matrix to be transformed.
    dt (float): The sample rate of the time axis [s].
    dx (float): The sample rate of the distance axis [m].
    padding (boolean): A boolean that decides if the signal should be padded with zeros before the fft, default to the next power of two.
    """

    # Compute sample rate and Nyquist rate for wavenumber and frequency
    print(f'Inpt: dx = {dx}, dt = {dt}')
    kk_Nyq = 1 / (2*dx) 
    ff_Nyq = 1 / (2*dt) 
    print(f'Nyquist limits: kk_Nyq = {kk_Nyq}, ff_Nyq = {ff_Nyq}')
    
    if padding == True:
        # Genereate wavenumber and frequency vectors
        kk = np.arange(-kk_Nyq, kk_Nyq, 1/(2**nextpow2(np.shape(data)[0])*dx)) # With wavenunmber shift
        ff = np.arange(-ff_Nyq, ff_Nyq, 1/(2**nextpow2(np.shape(data)[1])*dt)) # With frequency shift
        # Transform the signal to the f-k domain
        fft2_signal = np.fft.fftshift( np.fft.fft2(data,[2**nextpow2(np.shape(data)[0]), 2**nextpow2(np.shape(data)[1])]) ) # Transform the signal to the f-k domain with size (nx X nt), then fft shift to center the zero frequency and wavenumber

    else: 
        kk = np.arange(-kk_Nyq, kk_Nyq, 1/(np.shape(data,0)*dx)) # With wavenunmber shift
        ff = np.arange(-ff_Nyq, ff_Nyq, 1/(np.shape(data,1)*dt)) # With frequency shift
        fft2_signal = np.fft.fftshift( np.fft.fft2(data,[np.shape(data)[0], np.shape(data)[1]] ) ) # Transform the signal to the f-k domain with size (nx X nt), then fft shift to center the zero frequency and wavenumber
    
    amp_fft2_signal = np.abs(fft2_signal) # Compute the amplitude spectrum of the f-k domain signal
    del fft2_signal

    print(f'Size input matrix = {np.shape(data)}, shape 2D fft return = {np.shape(amp_fft2_signal)}')

    return amp_fft2_signal, kk, ff

class DASmeta:
    def __init__(self, path2data, dx, dt):
        self.path2data = path2data
        self.dx = dx
        self.dt = dt

    def print_summary(self):
        print('The loaded path + data is ' + path2data)
        print(f"dx: {round(self.dx,4)} m")
        print(f"dt: {round(self.dt,4)} s")

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
    #print(fileformat)  # Output: 'mat'

    if fileformat == 'mat':
       # print(path2data)
        data = sp.io.loadmat(path2data)
        meta = DASmeta(path2data,np.mean(np.diff(data['xx_abs'])), 10)
        meta.print_summary()
    elif fileformat == 'hdf5':
        dfdas_reloaded = simpledas.load_DAS_files(path2data)
        meta = dfdas_reloaded.meta                                      
        data = dfdas_reloaded

    return data, meta, fileformat

def wiggle(xx,yy,offset,clrLine,clrFill):
    plt.plot(xx,yy,'-',color=clrLine) # Normal wigigle    
    plt.fill_betweenx(yy,offset,xx,where=(x>=offset),color=clrFill) # Fill positive valuesc

def compute_RMS(data, axisNum):
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

def strain2strainRate(data, dt):
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

def strainRate2strain(data, dt):
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

def changeGaugeLength(data,oldROIDec,newGL,newROIDec,windowType):
    """
    INPUT:
    data = Original data with gauge length eqaul to 'oldGL' (size [time,space])
    olddx = Spatial sampling of original data
    newGL = The new gauge length that is wanted
    filterType = The wanted filter type (normally from -newGL/2 tp +newGL/2).
    Options: 'triangle'
    -------------------------------------------------------------------------
    Return:
    newData = Data with new gauge length
    newGL = The new gauge length that is wanted
    """
    
    if windowType.lower() in ['rectwin', 'rectangularwindow', 'squarewindow', 'squarwin']:
        sigma = np.ones(len(np.arange(-round(newGL/2), round(newGL/2)+1, oldROIDec)))
    elif windowType.lower() in ['triang', 'triangle', 'triangle']:
        sigma = sp.signal.triang(len(np.arange(-round(newGL/2), round(newGL/2)+1, oldROIDec)))
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

def makeAudioFile(signal, Fs, path2save, fileName, nMethod):
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
        from scipy.io.wavfile import write       
        # Convert to 16-bit PCM format
        signal_audio = np.int16(signal_normalized_4audio * 32760) # 32767 is the maximum value for 16-bit signed integers, use this
        # Save as WAV file
        write(output_filename, int(Fs), signal_audio)

    elif nMethod == 2:
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