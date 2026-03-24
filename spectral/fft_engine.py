import numpy as np
import pandas as pd
from scipy.signal import get_window

def compute_fft_surface(price_series, window_size=64, freq_resolution=1.0):
    n = len(price_series)
    times = np.arange(window_size, n)
    freqs = np.fft.rfftfreq(window_size, d=freq_resolution)
    surface = np.zeros((len(times), len(freqs)))
    for i, t in enumerate(times):
        windowed = price_series[t-window_size:t]
        fft_vals = np.fft.rfft(windowed * get_window('hann', window_size))
        power = np.abs(fft_vals)**2
        surface[i] = power
    return surface, freqs, times
