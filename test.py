import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load the dataset
dataset = datasets.ImageFolder(root="images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Print out some samples
for images, labels in dataloader:
    for i in range(5):
        print(f"Image {i} - Label: {labels[i]}")
    break