import time
import pickle
import os
import torch
import chess
import sys
import fcntl

sys.path.append('/content/RL-based-Chess-Assistant-')

from google.colab import drive
drive.mount('/content/drive')

DRIVE_PATH  = '/content/drive/MyDrive/chess_rl'
import uuid
worker_id = str(uuid.uuid4())[:8]
BUFFER_PATH = f'{DRIVE_PATH}/buffer_{worker_id}.pkl'
CKPT_PATH   = f'{DRIVE_PATH}/checkpoint.pt'
NNUE_PATH   = f'{DRIVE_PATH}/nnue.pth'

from RL.RL_training.self_play import run_self_play
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.config import Config
from pretraining_nnue_code import NNUE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_buffer():
    if os.path.exists(BUFFER_PATH):
        with open(BUFFER_PATH, 'rb') as f:
            return pickle.load(f)
    return []


def save_buffer(data):
    existing = load_buffer()
    existing.extend(data)
    if len(existing) > 50000:
        existing = existing[-50000:]
    tmp_path = BUFFER_PATH + ".tmp"
    with open(tmp_path, 'wb') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        pickle.dump(existing, f)
        fcntl.flock(f, fcntl.LOCK_UN)
    os.replace(tmp_path, BUFFER_PATH)
    print(f"Buffer saved: {len(existing)} tuples")


def load_latest_checkpoint(network):
    if os.path.exists(CKPT_PATH):
        network.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print("Loaded latest checkpoint from Drive")
    else:
        print("No checkpoint yet — using pretrained NNUE weights")
    return network


def build_model():
    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load(NNUE_PATH, map_location=device))
    nnue.eval()
    print("NNUE loaded")
    network = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=True)
    network = network.to(device)
    network.eval()
    return network


def main():
    network = build_model()

    args = {
        'C': Config.c_puct,
        'num_searches': Config.num_simulations,
        'device': device,             
        'add_noise': True,
        'dirichlet_alpha': 0.3,
        'dirichlet_epsilon': 0.25
    }

    iteration = 0

    while True:
        print(f"\n--- Worker iteration {iteration} ---")

        network = load_latest_checkpoint(network)
        network.eval()

        game_data = run_self_play(network, iteration)
        print(f"Generated {len(game_data)} tuples")

        if len(game_data) > 0:
                        
            try:
                save_buffer(game_data)
            except Exception as e:
                print(f"Buffer write failed: {e}, retrying in 10s")
                time.sleep(10)
                save_buffer(game_data)
        else:
            print("WARNING: no game data generated this iteration")

        iteration += 1
        time.sleep(120)


if __name__ == "__main__":
    main()
