import torch
import torch.nn as nn
import os

# Define the architecture exactly as in app.py
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)

def verify():
    print("Initializing model...")
    model = CNN(num_classes=10)
    
    path = 'animals10_cnn.pth'
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return

    print("Loading weights...")
    try:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        print("Weights loaded successfully!")
    except Exception as e:
        print(f"Failed to load weights: {e}")
        return

    model.eval()
    print("Running dummy inference...")
    dummy_input = torch.randn(1, 3, 48, 48)
    try:
        output = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print("Inference successful!")
    except Exception as e:
        print(f"Inference failed: {e}")

if __name__ == "__main__":
    verify()
