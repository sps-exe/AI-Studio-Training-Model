import torch
import torch.nn as nn
import torch.optim as optim
import argparse # This lets us run commands like `python experiment_runner.py --exp baseline`
import sys
import os

# Ensure Python knows where to find our 'src' folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import our custom modules
from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.train import train_model
from src.utils import save_experiment, plot_history

# Step 5: The "Captain" (Experiment Runner)
# This file ties everything together. 
# Instead of changing code to run a new experiment, you just change the command.

def run_experiment(exp_name, epochs=15, batch_size=64, learning_rate=0.001):
    print(f"\n{'='*40}")
    print(f"=== Experiment: {exp_name.upper()} ===")
    print(f"{'='*40}\n")
    
    # 1. Device Setup (MPS for Mac, CUDA for Nvidia, CPU fallback)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("--> Using MPS (Apple Silicon) acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("--> Using CUDA.")
    else:
        device = torch.device("cpu")
        print("--> Using CPU (Warning: Slow).")
        
    # 2. Configure Experiment Variables
    # Default settings
    config = {
        'experiment': exp_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device)
    }
    
    use_augmentation = False
    dropout_rate = 0.0
    
    # Logic for switching experiments
    if exp_name == 'baseline':
        print("--> Strategy: Standard training. Expect Overfitting.")
        
    elif exp_name == 'augmentation':
        use_augmentation = True
        print("--> Strategy: Data Augmentation ENABLED (Flips/Crops).")
        print("--> Hypothesis: Generalization should improve.")
        
    elif exp_name == 'capacity':
        # For this experiment, we add Dropout to regularize the model further.
        use_augmentation = True
        dropout_rate = 0.5
        print("--> Strategy: Data Augmentation + Dropout (0.5).")
        print("--> Hypothesis: Even better generalization, but training might be slower.")
        
    else:
        print(f"Error: Unknown experiment '{exp_name}'")
        return

    # Store these in config so we remember what we did
    config['use_augmentation'] = use_augmentation
    config['dropout_rate'] = dropout_rate

    # 3. Load Data
    print("\n[1/3] Loading Data...")
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, use_augmentation=use_augmentation)
    
    # 4. Build Model
    print("\n[2/3] Building Model...")
    model = SimpleCNN(dropout_rate=dropout_rate).to(device)
    
    # Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 5. Train!
    print("\n[3/3] Starting Training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
    
    # 6. Save Results
    print(f"\n{'-'*40}")
    save_dir = os.path.join('experiments', exp_name)
    save_experiment(exp_name, history, config)
    
    # Automatically save the plot
    plot_path = os.path.join(save_dir, 'loss_curve.png')
    plot_history(history, save_path=plot_path)
    
    print(f"___ Experiment {exp_name} Finished ___")
    print(f"Results saved to: {save_dir}")
    print(f"Check {plot_path} to see your Loss Curve!")

if __name__ == "__main__":
    # This block handles the command line arguments
    parser = argparse.ArgumentParser(description='CIFAR-10 Stage 2 Experiments')
    parser.add_argument('--exp', type=str, required=True, choices=['baseline', 'augmentation', 'capacity'],
                        help='Which experiment to run: baseline, augmentation, or capacity')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs (default: 15)')
    
    args = parser.parse_args()
    
    run_experiment(args.exp, epochs=args.epochs)
