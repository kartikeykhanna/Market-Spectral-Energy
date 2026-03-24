import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import time
import threading

REGIME_COLORS = {
    'bull': 'lime',
    'bear': 'red',
    'mean': 'blue',
    'volatility': 'orange',
    'crash': 'magenta'
}

class SpectralSurfaceDashboard:
    def __init__(self, surface, freqs, times, regimes, window_size=64, freq_resolution=1.0):
        self.surface = surface
        self.freqs = freqs
        self.times = times
        self.regimes = regimes
        self.window_size = window_size
        self.freq_resolution = freq_resolution
        self.playing = True
        self.overlay = True
        self.current_frame = 0
        self.max_frame = surface.shape[0]

    def _get_color_map(self):
        # Map regimes to colors
        return [REGIME_COLORS.get(r, 'gray') for r in self.regimes]

    def _update_surface(self, ax, frame):
        try:
            ax.clear()
            z = self.surface[max(0, frame-30):frame+1]
            x = self.freqs
            y = self.times[max(0, frame-30):frame+1]
            X, Y = np.meshgrid(x, y)
            surf = ax.plot_surface(X, Y, z, cmap='viridis', alpha=0.95)
            ax.set_title('Market Spectral Energy Landscape')
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Time')
            ax.set_zlabel('Spectral Power')
            if self.overlay:
                regime_colors = self._get_color_map()[max(0, frame-30):frame+1]
                ax.scatter(X.flatten(), Y.flatten(), z.flatten(), c=regime_colors * len(x), s=10, alpha=0.7)
        except Exception as e:
            print(f"Error updating surface: {e}")

    def _animate(self):
        plt.ion()
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        while self.playing and self.current_frame < self.max_frame:
            self._update_surface(ax, self.current_frame)
            plt.draw()
            plt.pause(0.05)
            self.current_frame += 1

    def run(self):
        print("Launching dashboard...")
        self._animate()
        plt.ioff()
        plt.show()
