import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import SimpleCNN
from src.dataset import get_dataloaders
from src.train import train_model
from src.utils import save_experiment, plot_history

def run_experiment(exp_name, epochs=15, batch_size=64, learning_rate=0.001):
    print(f"\n{'='*40}")
    print(f"=== Experiment: {exp_name.upper()} ===")
    print(f"{'='*40}\n")
    
    # Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("--> Using MPS (Apple Silicon).")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("--> Using CUDA.")
    else:
        device = torch.device("cpu")
        print("--> Using CPU.")
        
    config = {
        'experiment': exp_name,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'device': str(device)
    }
    
    use_augmentation = False
    dropout_rate = 0.0
    
    if exp_name == 'baseline':
        print("--> Mode: Baseline")
        
    elif exp_name == 'augmentation':
        use_augmentation = True
        print("--> Mode: Augmentation Enabled")
        
    elif exp_name == 'capacity':
        use_augmentation = True
        dropout_rate = 0.5
        print("--> Mode: Augmentation + Dropout (0.5)")
        
    else:
        print(f"Error: Unknown experiment '{exp_name}'")
        return

    config['use_augmentation'] = use_augmentation
    config['dropout_rate'] = dropout_rate

    print("\n[1/3] Loading Data...")
    train_loader, val_loader = get_dataloaders(batch_size=batch_size, use_augmentation=use_augmentation)
    
    print("\n[2/3] Building Model...")
    model = SimpleCNN(dropout_rate=dropout_rate).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("\n[3/3] Starting Training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=epochs, device=device)
    
    print(f"\n{'-'*40}")
    save_dir = os.path.join('experiments', exp_name)
    save_experiment(exp_name, history, config)
    
    plot_path = os.path.join(save_dir, 'loss_curve.png')
    plot_history(history, save_path=plot_path)
    
    print(f"___ Experiment {exp_name} Finished ___")
    print(f"Results saved to: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CIFAR-10 Experiments')
    parser.add_argument('--exp', type=str, required=True, choices=['baseline', 'augmentation', 'capacity'],
                        help='Experiment to run')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    
    args = parser.parse_args()
    
    run_experiment(args.exp, epochs=args.epochs)
