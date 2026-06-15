import time
import pickle
import os
import torch
import chess
from RL.RL_training.self_play import run_self_play
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.checkpoints import load_checkpoint
from RL.config import Config
from pretraining_nnue_code import NNUE
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
import traceback

# monkey patch chess.Board.push to catch the crash
original_push = chess.Board.push
def safe_push(self, move):
    try:
        return original_push(self, move)
    except AssertionError as e:
        print(f"CRASH in push!")
        print(f"Move: {move}")
        print(f"Board: {self.fen()}")
        traceback.print_stack()
        raise
chess.Board.push = safe_push
checkpoint_poll_seconds = 120
buffer = "/content/RL-based-Chess-Assistant-/buffer.pkl"
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
    from google.colab import drive
    drive.mount('/content/drive')

    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load('/content/drive/MyDrive/chess_rl/nnue.pth', map_location='cpu'))
    nnue.eval()
    print("Pretrained NNUE loaded!")
    network = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=True) 
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
