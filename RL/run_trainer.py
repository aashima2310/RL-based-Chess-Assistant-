import pickle
import os
import time
import sys
import glob
import torch

sys.path.append('/teamspace/studios/this_studio/RL-based-Chess-Assistant-')

DRIVE_PATH = '/teamspace/studios/this_studio/drive/MyDrive/chess_rl'
NNUE_PATH  = '/teamspace/studios/this_studio/RL-based-Chess-Assistant-/nnue.pth'
CKPT_PATH  = f'{DRIVE_PATH}/checkpoint.pt'

from RL.RL_training.trainer import Trainer
from RL.RL_training.replay_buffer import ReplayBuffer
from RL.RL_training.champion import evaluate
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.checkpoints import save_checkpoint, load_checkpoint
from RL.training_stats import log
from RL.config import Config
from pretraining_nnue_code import NNUE


def load_buffer_into_ram(replay_buffer):
    buffer_files = glob.glob(f'{DRIVE_PATH}/buffer_*.pkl')

    if not buffer_files:
        print("No buffer files found, waiting for workers...")
        return 0

    all_data = []
    for path in buffer_files:
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            all_data.extend(data)
            print(f"Loaded {len(data)} tuples from {os.path.basename(path)}")
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

    if len(all_data) > 50000:
        all_data = all_data[-50000:]

    print(f"TOTAL: {len(all_data)} tuples from {len(buffer_files)} workers")
    replay_buffer.buffer.clear()
    replay_buffer.push(all_data)
    return len(all_data)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load(NNUE_PATH, map_location=device))
    nnue.eval()
    print("Pretrained NNUE loaded!")

    champion = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=True)
    champion = champion.to(device)

    if os.path.exists(CKPT_PATH):
        champion.load_state_dict(torch.load(CKPT_PATH, map_location=device))
        print("Resumed from existing checkpoint")
    else:
        print("No checkpoint found, starting fresh")

    trainer = Trainer(champion)
    replay_buffer = ReplayBuffer(max_size=50000)
    iteration = 0

    while True:
        print(f"\n=== TRAINER: Iteration {iteration} ===")

        num_tuples = load_buffer_into_ram(replay_buffer)

        if num_tuples < Config.batch_size:
            print(f"Not enough data yet ({num_tuples} tuples). Waiting 60s...")
            time.sleep(60)
            continue

        loss = trainer.train(replay_buffer)
        print(f"Training loss: {loss:.4f}")

        challenger = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=False)
        challenger.load_state_dict(champion.state_dict())
        challenger = challenger.to(device)

        is_better = evaluate(champion, challenger)

        if is_better:
            torch.save(champion.state_dict(), CKPT_PATH)
            print("Champion updated and saved to Drive")
        else:
            print("Champion unchanged")

        estimated_elo = 800 + iteration * 60
        log(iteration, loss, estimated_elo, num_tuples)

        if iteration == 100:
            print("Phase 2: unfreezing backbone")
            param_groups = champion.unfreeze_backbone(lr_scale=0.1)
            trainer.optimizer = torch.optim.Adam(param_groups)

        iteration += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
