import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from hamming import hamming
from hann import hann
from rectangular import rectangular

def generate_spectrograms(song_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    audio_files = [f for f in os.listdir(song_folder) if f.endswith(('.wav', '.mp3', '.flac'))]
    
    windows = {
        'hamming': hamming,
        'hann': hann,
        'rectangular': rectangular
    }
    
    n_fft = 2048
    hop_length = 512
    
    for audio_file in audio_files:
        file_path = os.path.join(song_folder, audio_file)
        y, sr = librosa.load(file_path, sr=None)
        base_name = os.path.splitext(audio_file)[0]
        
        for win_name, win_func in windows.items():
            window = win_func(n_fft)
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
            S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
            
            plt.figure(figsize=(12, 6))
            librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Spectrogram ({win_name} window) - {base_name}')
            plt.tight_layout()
            
            output_filename = os.path.join(output_folder, f'{base_name}_{win_name}_spec.png')
            plt.savefig(output_filename)
            plt.close()

if __name__ == "__main__":
    generate_spectrograms('/data/b22cs089/speechPA1/taskb/songs', '/data/b22cs089/speechPA1/taskb/spectograms')