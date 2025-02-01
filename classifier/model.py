import torch
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*32*32, 1024), 
            nn.Tanh(),
            nn.Linear(1024, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
