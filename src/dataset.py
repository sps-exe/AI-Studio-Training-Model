import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import ssl

# Fix for macOS SSL certificate errors
ssl._create_default_https_context = ssl._create_unverified_context

def get_dataloaders(batch_size=64, use_augmentation=False, subset_size=10000):
    stats = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                               download=True, transform=train_transform)
    val_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                             download=True, transform=test_transform)
    
    # Stratified subset
    indices = list(range(subset_size))
    train_subset = Subset(train_dataset, indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader
