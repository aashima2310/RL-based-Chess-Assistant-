import torch
import os
from RL.config import Config

def save_checkpoint(network, filename="checkpoint.pt"):
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    
    path = os.path.join(Config.checkpoint_dir, filename)
    torch.save(network.state_dict(), path)
    print(f"Saved checkpoint: {path}")

def load_checkpoint(network, filename="checkpoint.pt"):
    path = os.path.join(Config.checkpoint_dir, filename)
    if os.path.exists(path):
        network.load_state_dict(torch.load(path))
        
        print(f"Loaded checkpoint: {path}")
    else:
        print("No checkpoint found, starting new")
    return network
