#%% Imports
import pandas as pd
import os
import torch
from torchvision import transforms
from PIL import Image

#%% Get Sample Files (5)
df = pd.read_csv('data/train.csv', index_col='Unnamed: 0').iloc[:5]
paths = df['file_name'].tolist()
labels = df['label'].tolist()

#%%
# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to 224x224
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Load and transform images
image_tensors = [transform(Image.open(f"data/images/{path}").convert("RGB")) for path in paths]

# Stack into a batch
image_tensors = torch.stack(image_tensors)

# Convert labels to tensor
label_tensors = torch.tensor(labels)