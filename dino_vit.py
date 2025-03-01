import torch
import timm
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

# Load DINO-ViT model (base version)
model = timm.create_model("vit_base_patch16_224.dino", pretrained=True)
model.eval()  # Set to evaluation mode

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
])

def extract_features(img_path, model):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        features = model(img_tensor)  # Extract features

    return features.flatten().cpu().numpy()

# Load the dataset
train_csv_path = "/path/to/train.csv"  # Update with the correct path to train.csv
df = pd.read_csv(train_csv_path)

# Randomly sample 5,000 rows
df_sampled = df.sample(n=5000, random_state=42)

all_features = []

# Go through the sampled dataset and extract features
for index, row in df_sampled.iterrows():
    img_path = os.path.join('/path/to/images', row['file_name'])  # Update the image directory

    if os.path.exists(img_path):
        features = extract_features(img_path, model)
        all_features.append(features)
    else:
        print(f"Missing image: {img_path}")

    if (index + 1) % 100 == 0:  # Update progress every 100 images
            print(f"Processed {index + 1}/{len(df_sampled)} images")

# Convert the extracted features to a numpy array
all_features_array = np.array(all_features)

# Convert features to a DataFrame and save as CSV
features_df = pd.DataFrame(all_features_array)
features_df.to_csv("all_features.csv", index=False)
print("Features saved to 'all_features.csv'")
