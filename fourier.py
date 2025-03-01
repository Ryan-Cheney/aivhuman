
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

df = pd.read_csv("train.csv", index_col=0, nrows=8000)

def fourier_transform(file_name):
    # Load grayscale image
    image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (256, 256))

    # Apply Fourier Transform
    fft_image = fft2(image)
    fft_image = fftshift(np.abs(fft_image))  # Shift frequencies to center

    # Display the frequency domain representation
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(np.log(1 + fft_image), cmap="gray")  # Log transform for visibility
    plt.title("Fourier Transform")
    plt.show()



class FourierTransformDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.df.iloc[idx]['file_name']
        label = self.df.iloc[idx]['label']

        # Load image in grayscale
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Error loading image: {file_path}")

        image = cv2.resize(image, (256, 256))

        # Compute Fourier Transform
        fft_image = np.abs(fftshift(fft2(image)))
        fft_image = np.log(1 + fft_image)  # Log transform to enhance visibility

        # Normalize and convert to tensor
        fft_image = (fft_image - np.min(fft_image)) / (np.max(fft_image) - np.min(fft_image))
        fft_image = torch.tensor(fft_image, dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        label = torch.tensor(label, dtype=torch.long)
        return fft_image, label

# Create dataset instance
dataset = FourierTransformDataset(df)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)  # Adjust batch size as needed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FourierCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FourierCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create an empty list to store features
all_features = []

# Extract features
for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)

    # Forward pass (get features before the final layer)
    x = model.conv1(images)
    x = model.pool(x)
    x = model.conv2(x)
    x = model.pool(x)
    x = x.view(x.size(0), -1)  # Flatten
    features = model.fc(x)  # Features from fully connected layer
    all_features.append(features.cpu().detach().numpy())

# Save features to CSV
features_df = pd.DataFrame(all_features)
features_df.to_csv('features.csv', index=False)

print("Features saved to features.csv")
