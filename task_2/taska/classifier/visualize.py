
import os
import random
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataloader import hann_window, hamming_window, rectangular_window

# Configuration
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
NUM_EXAMPLES = 5
OUTPUT_DIR = "spectrogram_visualizations"

def save_spectrogram(y, sr, window_func, filename, title_suffix):
    """Generate and save spectrogram for a specific window"""
    window = window_func(N_FFT)
    

    stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, window=window)
    spectrogram = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, sr=sr, hop_length=HOP_LENGTH, 
                            x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"Spectrogram ({title_suffix})")
    plt.tight_layout()
    
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def visualize_samples():
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metadata = pd.read_csv('/data/b22cs089/speechPA1/dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
    selected_files = metadata.groupby('fold').apply(
        lambda x: x.sample(1)).reset_index(drop=True).head(NUM_EXAMPLES)

    for idx, row in selected_files.iterrows():
        file_path = f"/data/b22cs089/speechPA1/dataset/UrbanSound8K/audio/fold{row['fold']}/{row['slice_file_name']}"
        audio, _ = librosa.load(file_path, sr=SR)
        audio = librosa.util.fix_length(audio, size=SR*4)
        
        base_name = f"fold{row['fold']}_{row['slice_file_name'].replace('.wav', '')}"
        for window_func, wname in [(hann_window, 'hann'),
                                 (hamming_window, 'hamming'),
                                 (rectangular_window, 'rectangular')]:
            output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{wname}.png")
            save_spectrogram(audio, SR, window_func, output_path, 
                           f"{wname} window - {row['class']}")

if __name__ == "__main__":
    visualize_samples()
    print(f"Generated {NUM_EXAMPLES*3} spectrograms in '{OUTPUT_DIR}' directory")
