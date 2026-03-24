import sys
import os
from utils.dependency_installer import install_dependencies

# Ensure dependencies are installed
install_dependencies()

from spectral.fft_engine import compute_fft_surface
from spectral.wavelet_engine import compute_wavelet_surface
from visualization.spectral_surface import SpectralSurfaceDashboard
from ai.regime_classifier import RegimeClassifier
import pandas as pd
import numpy as np

# 1) Generate or load financial price data
def load_or_generate_data(n_steps=1000):
    # Placeholder: generate synthetic price series
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(n_steps)) + 100
    dates = pd.date_range('2020-01-01', periods=n_steps)
    df = pd.DataFrame({'date': dates, 'price': prices})
    return df

def main():
    df = load_or_generate_data()
    fft_surface, freqs, times = compute_fft_surface(df['price'])
    wavelet_surface, wave_freqs, wave_times = compute_wavelet_surface(df['price'])
    classifier = RegimeClassifier()
    regimes = classifier.predict_regimes(fft_surface, wavelet_surface)
    dashboard = SpectralSurfaceDashboard(
        fft_surface, freqs, times, regimes,
        window_size=64, freq_resolution=1.0
    )
    dashboard.run()

if __name__ == "__main__":
    main()
