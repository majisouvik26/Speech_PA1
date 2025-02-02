import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
import librosa
from sklearn.preprocessing import LabelEncoder

# Import the model architecture
from model import AudioCNN
# Import helper functions from the dataloader module
from dataloader import load_metadata, hann_window, hamming_window, rectangular_window, get_features_labels

def resize_feature(feature, target_shape=(128, 128)):
    """
    Resize a 1D feature vector to a 2D array.
    (Using np.resize for simplicity; for more refined behavior, consider interpolation.)
    """
    return np.resize(feature, target_shape)

class UrbanSoundDataset(Dataset):
    """
    A dataset class that extracts features from the UrbanSound8K audio files.
    This version is used for testing and expects metadata filtered for fold 10.
    """
    def __init__(self, metadata, window_func):
        features, labels = get_features_labels(metadata, window_func)
        # Resize each feature to shape (128, 128)
        self.features = np.array([resize_feature(feat) for feat in features])
        self.labels = np.array(labels)
        # Encode string labels to integer indices
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
       
        self.features = np.expand_dims(self.features, axis=1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def main(args):
    window_funcs = {
         'hann': hann_window,
         'hamming': hamming_window,
         'rectangular': rectangular_window,
    }
    window_func = window_funcs[args.window]

    metadata = load_metadata(args.metadata)
    test_metadata = metadata[metadata['fold'] == 10]

    test_dataset = UrbanSoundDataset(test_metadata, window_func)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(input_channels=1)
    model.load_state_dict(torch.load(args.model_in, map_location=device))
    model.to(device)
    model.eval() 
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(test_loader)
    accuracy = 100.0 * total_correct / total_samples
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test AudioCNN on UrbanSound8K (using fold 10)")
    parser.add_argument('--metadata', type=str, default="path/to/metadata.csv",
                        help="Path to the metadata CSV file")
    parser.add_argument('--window', type=str, choices=['hann', 'hamming', 'rectangular'], default='hann',
                        help="Window function to use for feature extraction")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for testing")
    parser.add_argument('--model_in', type=str, default="/data/b22cs089/speechPA1/classifier/audio_cnn_hann.pth",
                        help="Path to the trained model checkpoint")
    args = parser.parse_args()
    main(args)
