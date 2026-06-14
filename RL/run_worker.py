import time
import pickle
import os
import torch
from RL.RL_training.self_play import run_self_play
from combined_network import CombinedNetwork   
from checkpoints import load_checkpoint
from config import Config

checkpoint_poll_seconds = 120
buffer = "buffer.pkl"

def load_buffer():
    if os.path.exists(buffer):
        with open(buffer, 'rb') as f:
            return pickle.load(f)
    return []


def save_buffer(data):
    existing = load_buffer()
    existing.extend(data)

    if len(existing) > 50000:
        existing = existing[-50000:]

    with open(buffer, 'wb') as f:
        pickle.dump(existing, f)


def main():
    
    network = CombinedNetwork()
    network.eval()
   
    iteration = 0

    while True:

        print(f"\n--- Worker: iteration {iteration} ---")

        network = load_checkpoint(network, "checkpoint.pt")
        game_data = run_self_play(network, iteration)
        save_buffer(game_data)

        print(f"Worker wrote {len(game_data)} tuples to buffer")

        iteration += 1

        time.sleep(checkpoint_poll_seconds)


if __name__ == "__main__":
    main()
