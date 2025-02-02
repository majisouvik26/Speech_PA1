import argparse
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from dataloader import load_metadata, get_features_labels, hann_window, hamming_window, rectangular_window
from getmodel import get_model
from tqdm import tqdm

def train_and_save(window_func, model_type):
    metadata = load_metadata('/data/b22cs089/speechPA1/dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
    train_metadata = metadata[metadata['fold'] <= 9]  
    
    X_train, y_train = get_features_labels(train_metadata, window_func)
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    model = get_model(model_type)
    print(f"Training {model_type.upper()} with {window_func.__name__}...")
    for _ in tqdm(range(1)):
        model.fit(X_scaled, y_train)
    joblib.dump(model, f'model_{model_type}_{window_func.__name__}.pkl')
    joblib.dump(scaler, f'scaler_{model_type}_{window_func.__name__}.pkl')
    print(f"Trained {model_type.upper()} with {window_func.__name__}!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--window', required=True, choices=['hann', 'hamming', 'rectangular'])
    parser.add_argument('--model', required=True, choices=['svm', 'mlp'])
    args = parser.parse_args()
    
    window_map = {
        'hann': hann_window,
        'hamming': hamming_window,
        'rectangular': rectangular_window
    }
    
    train_and_save(window_map[args.window], args.model)
