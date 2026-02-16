import matplotlib.pyplot as plt
import json
import os

def save_experiment(experiment_name, history, config, base_dir='experiments'):
    path = os.path.join(base_dir, experiment_name)
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, 'metrics.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    with open(os.path.join(path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"Experiment saved to {path}")

def plot_history(history, save_path=None):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Loss Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r--', label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    plt.plot(epochs, history['val_acc'], 'r--', label='Val Acc')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
