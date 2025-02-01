import numpy as np

def hamming(M):
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(M) / (M - 1))