import numpy as np

def hann(M):
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(M) / (M - 1))