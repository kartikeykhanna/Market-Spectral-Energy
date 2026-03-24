import numpy as np
import pywt

def compute_wavelet_surface(price_series, wavelet='cmor', scales=np.arange(1, 64)):
    coef, freqs = pywt.cwt(price_series, scales, wavelet)
    power = np.abs(coef)**2
    times = np.arange(len(price_series))
    return power.T, freqs, times
