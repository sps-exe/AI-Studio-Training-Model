import matplotlib.pyplot as plt
import json
import os

# Step 4: The Analyst (Visualization)
# This file helps you SEE what happened. 
# Evaluators need "Loss Curves". This code draws them.

def save_experiment(experiment_name, history, config, base_dir='experiments'):
    """
    Saves the numbers (history) and settings (config) to a file.
    So you can look at them later without re-running the training.
    """
    path = os.path.join(base_dir, experiment_name)
    os.makedirs(path, exist_ok=True)
    
    # Save metrics (The numbers)
    with open(os.path.join(path, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    # Save config (The settings)
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Experiment saved to {path}")

def plot_history(history, save_path=None):
    """
    Draws the Loss and Accuracy curves.
    This is the most important output for your report.
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss (The Error)
    # We want this to go DOWN.
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve (Lower = Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Accuracy (The Score)
    # We want this to go UP.
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve (Higher = Better)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
