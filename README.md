# Market Spectral Energy Engine

Analyze financial price series using spectral analysis (FFT and wavelets) and visualize the market’s hidden frequency structure as a 3D animated energy landscape.

## Creator / Developer
**tubakhxn**

---

## What is this project?
This project is a Python-based engine for analyzing financial market price series using advanced spectral techniques (FFT and wavelets). It visualizes the hidden frequency structure of markets as a 3D animated energy landscape and uses AI to classify market regimes (bull, bear, mean reversion, volatility spike, crash risk). The dashboard provides interactive controls for analysis and visualization.

---

## How to Fork
1. Click the "Fork" button at the top right of the GitHub repository page.
2. Clone your forked repository:
    ```
    git clone https://github.com/your-username/market-spectral-engine.git
    ```
3. Install Python 3.8+ and run:
    ```
    python main.py
    ```
    All dependencies will be auto-installed.

Feel free to contribute or customize for your own financial analysis needs!

## Features
- Load or generate financial price data
- Compute rolling FFT and wavelet transforms
- Calculate spectral energy density
- 3D animated visualization (frequency x time x spectral power)
- AI regime classification (bull, bear, mean reversion, volatility spike, crash risk)
- Dashboard controls: FFT window size, frequency resolution, play/pause, regime overlay

## Usage

1. Install Python 3.8+
2. Run:
    ```
    python main.py
    ```
   All dependencies will be auto-installed.

## Project Structure
- `main.py`: Entry point
- `requirements.txt`: Dependencies
- `spectral/fft_engine.py`: FFT computation
- `spectral/wavelet_engine.py`: Wavelet computation
- `ai/regime_classifier.py`: Regime classification
- `visualization/spectral_surface.py`: 3D visualization
- `utils/dependency_installer.py`: Auto-installs dependencies

## Notes
- Visualization uses Plotly for interactive 3D animation
- AI regime classifier uses scikit-learn and PyTorch
- Animation handles 1000+ time steps smoothly
