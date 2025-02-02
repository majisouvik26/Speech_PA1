import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def hann_window(N):
    n = np.arange(N)
    return 0.5 * (1 - np.cos(2 * np.pi * n / (N - 1)))

def hamming_window(N):
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2 * np.pi * n / (N - 1))

def rectangular_window(N):
    return np.ones(N, dtype=np.float64)

def load_metadata(csv_path):
    return pd.read_csv(csv_path)[['slice_file_name', 'fold', 'class']]

def robust_stats(x, axis):
        mean = np.mean(x, axis=axis, keepdims=True, dtype=np.float64)
        std = np.sqrt(
            np.mean((x - mean)**2, axis=axis, keepdims=True, dtype=np.float64)
            + 1e-8  
        )
        return mean.squeeze(), std.squeeze()

def extract_features(file_path, window_func, sr=22050, n_fft=2048, hop_length=512, n_mfcc=13):
    y, _ = librosa.load(file_path, sr=sr)
    y = librosa.util.fix_length(y, size=sr*4)
    window = window_func(n_fft).astype(np.float32)
   
    stft = librosa.stft(
        y, 
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
        center=True,
        pad_mode='reflect'
    )
    spectrogram = np.abs(stft)
    
    S = librosa.feature.melspectrogram(
        S=spectrogram,
        sr=sr,
        n_mels=128,
        fmax=sr/2
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mfccs = librosa.feature.mfcc(S=S_db, n_mfcc=n_mfcc)
    
    m_mean, m_std = robust_stats(mfccs, axis=1)
    return np.hstack([m_mean, m_std])

def get_features_labels(metadata, window_func):
    features, labels = [], []
    for idx, row in metadata.iterrows():
        file_path = f"/data/b22cs089/speechPA1/dataset/UrbanSound8K/audio/fold{row['fold']}/{row['slice_file_name']}"
        try:
            feat = extract_features(file_path, window_func)
            features.append(feat)
            labels.append(row['class'])
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            continue 
    # le = LabelEncoder()
    # encoded_labels = le.fit_transform(labels)
    return np.array(features), np.array(labels)
