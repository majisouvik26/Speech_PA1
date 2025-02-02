import argparse
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from dataloader import load_metadata, get_features_labels, hann_window, hamming_window, rectangular_window

def test_model(window_func, model_type):
    metadata = load_metadata('/data/b22cs089/speechPA1/dataset/UrbanSound8K/metadata/UrbanSound8K.csv')
    test_metadata = metadata[metadata['fold'] == 10] 
    
    X_test, y_test = get_features_labels(test_metadata, window_func)

    model = joblib.load(f'model_{model_type}_{window_func.__name__}.pkl')
    scaler = joblib.load(f'scaler_{model_type}_{window_func.__name__}.pkl')
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\nTest Results ({model_type.upper()}, {window_func.__name__}):")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

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
    
    test_model(window_map[args.window], args.model)
