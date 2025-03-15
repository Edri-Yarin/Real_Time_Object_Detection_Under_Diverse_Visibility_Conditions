import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset class for loading
class VisibilityDataset(Dataset):
    def __init__(self, data_dir, annotations_file, transform=None):
        self.data_dir = data_dir
        self.annotations = self.load_annotations(annotations_file)
        self.transform = transform

    def load_annotations(self, annotations_file):
        annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split(",")
                annotations.append((img_path, int(label)))
        return annotations

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, label = self.annotations[idx]
        img = Image.open(os.path.join(self.data_dir, img_path)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, label


# Define image transforms
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load datasets (adjust file paths as necessary)
train_data_dir = 'path/to/train/images'
val_data_dir = 'path/to/val/images'
train_annotations_file = 'path/to/train/annotations.csv'
val_annotations_file = 'path/to/val/annotations.csv'

# Load datasets
train_dataset = VisibilityDataset(data_dir=train_data_dir, annotations_file=train_annotations_file, transform=train_transform)
val_dataset = VisibilityDataset(data_dir=val_data_dir, annotations_file=val_annotations_file, transform=val_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define MobileNetV3-based visibility classification model
class MobileNetV3VisibilityModel(nn.Module):
    def __init__(self, num_classes=18):
        super(MobileNetV3VisibilityModel, self).__init__()
        # Load the pre-trained MobileNetV3 model
        self.model = models.mobilenet_v3_small(pretrained=True)  # Use small model for efficiency

        # Replace the classifier with a new fully connected layer to match the number of visibility classes
        self.model.classifier[3] = nn.Linear(self.model.classifier[3].in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Initialize the model
model = MobileNetV3VisibilityModel(num_classes=18).to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize lists to store loss for plotting
train_losses = []
val_losses = []

# Function to train the model
def train_model(model, train_loader, val_loader, num_epochs=10):
    for epoch in range(num_epochs):
        print(f"\n{'='*30} Epoch {epoch+1}/{num_epochs} {'='*30}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

        # Calculate average loss and accuracy for training
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        train_losses.append(epoch_loss)

        print(f"\n[TRAIN] Epoch {epoch+1} Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)

        print(f"\n[VALID] Epoch {epoch+1} Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

# Train the model
train_model(model, train_loader, val_loader, num_epochs=10)

# Step 7: Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', color='b')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='r')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Ensure the target directory exists
weights_dir = '/content/weights'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)  # Create the directory if it doesn't exist

# Save the trained model
model_save_path = os.path.join(weights_dir, 'visibility_classification_model.pth')
torch.save(model.state_dict(), model_save_path)

print(f"Model weights saved to {model_save_path}")
