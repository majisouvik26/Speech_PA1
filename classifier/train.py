import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataloader import load_metadata, get_features_labels, hann_window, hamming_window, rectangular_window
from model import AudioCNN

def resize_feature(feature, target_shape=(128, 128)):
    """
    Resize a 1D feature vector to a 2D array.
    (Using np.resize for simplicity; for more refined behavior, consider interpolation.)
    """
    return np.resize(feature, target_shape)

class UrbanSoundDataset(Dataset):
    def __init__(self, metadata, window_func):
        features, labels = get_features_labels(metadata, window_func)
        # Resize each feature to shape (128, 128)
        self.features = np.array([resize_feature(feat) for feat in features])
        self.labels = np.array(labels)
        # Encode labels to integer indices
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)
        # Add a channel dimension: each sample becomes (1, 128, 128)
        self.features = np.expand_dims(self.features, axis=1)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def init_weights(m):
    """
    Initialize weights using Xavier (Glorot) uniform initialization,
    which is well suited for layers with tanh activation.
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
         nn.init.xavier_uniform_(m.weight)
         if m.bias is not None:
             nn.init.constant_(m.bias, 0)

def main(args):
    window_funcs = {
         'hann': hann_window,
         'hamming': hamming_window,
         'rectangular': rectangular_window,
    }
    window_func = window_funcs[args.window]

    num_epochs = args.epochs
    batch_size = args.batch_size
    learning_rate = args.lr

    metadata = load_metadata(args.metadata)
    train_metadata = metadata[metadata['fold'] != 10]

    train_dataset = UrbanSoundDataset(train_metadata, window_func)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioCNN(input_channels=1)
    model.apply(init_weights)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience, verbose=True
    )

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * running_correct / total_samples
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")
        scheduler.step(epoch_loss)
    
    torch.save(model.state_dict(), args.model_out)
    print("Model saved to", args.model_out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AudioCNN on UrbanSound8K")
    parser.add_argument('--metadata', type=str, default="path/to/metadata.csv",
                        help="Path to the metadata CSV file")
    parser.add_argument('--window', type=str, choices=['hann', 'hamming', 'rectangular'], default='hann',
                        help="Window function to use for feature extraction")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0, help="Weight decay for the optimizer")
    parser.add_argument('--lr_factor', type=float, default=0.1, help="Factor by which the learning rate will be reduced")
    parser.add_argument('--lr_patience', type=int, default=10, help="Number of epochs with no improvement before reducing LR")
    parser.add_argument('--model_out', type=str, default="audio_cnn.pth", help="Path to save the trained model")
    args = parser.parse_args()
    main(args)
