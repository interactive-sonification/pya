from . import Astft

class Amfcc(Astft):
    """Mel filtered Fourier spectrum (MFCC) class
    
    Steps of mfcc:
        Frame the signal into short frames.
        For each frame calculate the periodogram estimate of the power spectrum.
        Apply the mel filterbank to the power spectra, sum the energy in each filter.
        Take the logarithm of all filterbank energies.
        Take the DCT of the log filterbank energies.
        Keep DCT coefficients 2-13, discard the rest.
        
    Attributes
    ----------
    x : Asig or numpy.ndarray
        x can be two forms, the most commonly used is an Asig object. 
        Such as directly acquired from an Asig object via Asig.to_stft().
    sr : int
        sampling rate, this is only necessary if x is not Asig. (Default value = None)
    label : str
        name of the Asig. (Default value = None)
    window : str
        type of the window function (Default value = 'hann')
    nperseg : int
        number of samples per stft segment (Default value = '256')
    noverlap : int
        number of samples to overlap between segments (Default value = None)
    detrend : str or function or bool
        Specifies how to detrend each segment. If detrend is a string, 
        it is passed as the type argument to the detrend function. If it is a function, 
        it takes a segment and returns a detrended segment. If detrend is False, 
        no detrending is done. (Default value = False).
    return_onesided : bool
        If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. 
        Defaults to True, but for complex data, a two-sided spectrum is always returned. (Default value = True)
    boundary : str or None
        Specifies whether the input signal is extended at both ends, and how to generate the new values, 
        in order to center the first windowed segment on the first input point. 
        This has the benefit of enabling reconstruction of the first input point 
        when the employed window function starts at zero. 
        Valid options are ['even', 'odd', 'constant', 'zeros', None]. Defaults to ‘zeros’, 
        for zero padding extension. I.e. [1, 2, 3, 4] is extended to [0, 1, 2, 3, 4, 0] for nperseg=3. (Default value = 'zeros')
    padded : bool
        Specifies whether the input signal is zero-padded at the end to make the signal fit exactly into 
        an integer number of window segments, so that all of the signal is included in the output. 
        Defaults to True. Padding occurs after boundary extension, if boundary is not None, and padded is True, 
        as is the default. (Default value = True)
    axis : int
        Axis along which the STFT is computed; the default is over the last axis. (Default value = -1)
    cn : list or None
        Channel names of the Asig, this will be used for the Astft for consistency. (Default value = None)
    """

    def __init__(self, x, sr=None, label=None, nmfcc=20, window='hann', nperseg=256,
                 noverlap=None, nfft=None, detrend=False, return_onesided=True,
                 boundary='zeros', padded=True, axis=-1, cn=None):
        super().__init__()
        # Find the nearest nfft based on sr

        