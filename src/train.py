import torch
import time
import copy

# Step 3: The Training Engine
# This file teaches the model. It contains the loops that:
# 1. Feed data to the model
# 2. Calculate the error (Loss)
# 3. Adjust the model's brain (Backpropagation)

def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Runs one full pass over the training data.
    """
    model.train() # Switch to "Training Mode" (Enable Dropout/BatchNorm)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Loop over batches
    for i, (images, labels) in enumerate(loader):
        # Move data to GPU/MPS
        images, labels = images.to(device), labels.to(device)
        
        # 1. Clear old gradients
        optimizer.zero_grad()
        
        # 2. Forward Pass: Ask the model to guess
        outputs = model(images)
        
        # 3. Calculate Loss: How wrong was the guess?
        loss = criterion(outputs, labels)
        
        # 4. Backward Pass: Calculate gradients (how to fix the error)
        loss.backward()
        
        # 5. Optimizer Step: Update the weights
        optimizer.step()
        
        # Track statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, loader, criterion, device):
    """
    Evaluates the model on unseen data.
    CRITICAL: This tells us if the model is memorizing (Overfitting) or learning.
    """
    model.eval() # Switch to "Eval Mode" (Disable Dropout)
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # No gradients needed for validation (saves memory/speed)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=15, device='cpu'):
    """
    The Main Loop. Orchestrates training over multiple epochs.
    """
    since = time.time()
    
    # HISTORY: We must save these numbers to plot the curves later!
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    print(f"Training on {device} for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate immediately
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
              
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    
    return history
