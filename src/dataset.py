import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import ssl

# Fix for macOS SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context


# Step 2: The Data Pipeline
# We need to feed images to the model. 
# Crucially, we need to be able to SWITCH augmentation on/off for Experiment 2.

def get_dataloaders(batch_size=64, use_augmentation=False, subset_size=10000):
    """
    Returns train_loader and val_loader.
    """
    
    # 1. Normalization: consistent for all experiments
    # (R, G, B) means/stds are 0.5 to keep pixels between -1 and 1.
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    # 2. Define Transforms (The "Augmentation" logic)
    if use_augmentation:
        # EXPERIMENT 2 STRATEGY: 
        # Make the model work harder by randomly flipping/cropping images.
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
    else:
        # EXPERIMENT 1 (Baseline): Just load the image as-is.
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        
    # Validation data should NEVER be augmented (we want to test on real images)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    # 3. Load CIFAR-10
    # download=True ensures you have the data
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                               download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=test_transform)
    
    # 4. Create the 10,000 Image Subset
    # Your prompt specificially asked for a "10,000-image CIFAR-10 subset".
    # We select the first 10,000 indices. 
    indices = list(range(subset_size))
    train_subset = Subset(train_dataset, indices)
    
    # 5. Create Loaders
    # num_workers=2 helps load data in parallel so the GPU/MPS doesn't wait.
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"--> Loaded {len(train_subset)} Training Images (Augmentation={use_augmentation})")
    print(f"--> Loaded {len(val_dataset)} Validation Images")
    
    return train_loader, val_loader
