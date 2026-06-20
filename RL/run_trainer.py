import pickle
import os
import time
import sys
import glob
import torch
import numpy as np
import csv
import random

sys.path.append('/teamspace/studios/this_studio/RL-based-Chess-Assistant-')

DRIVE_PATH  = '/teamspace/studios/this_studio/drive/MyDrive/chess_rl'
NNUE_PATH   = f'{DRIVE_PATH}/nnue.pth'
CKPT_DRIVE  = f'{DRIVE_PATH}/checkpoint.pt'
CKPT_LOCAL  = '/teamspace/studios/this_studio/RL-based-Chess-Assistant-/RL/checkpoint.pt'
LOG_PATH    = f'{DRIVE_PATH}/training_log.csv'

from RL.RL_training.trainer import Trainer
from RL.RL_training.replay_buffer import ReplayBuffer
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.config import Config
from pretraining_nnue_code import NNUE


def load_all_buffers(replay_buffer):
    buffer_files = glob.glob(f'{DRIVE_PATH}/buffer*.pkl')
    if not buffer_files:
        print("No buffer files found. Waiting...")
        return 0
    print(f"Found {len(buffer_files)} buffer files")
    all_data = []
    for path in sorted(buffer_files):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, list) and len(data) > 0:
                all_data.extend(data)
                print(f"  {os.path.basename(path):35s} {len(data):>7} tuples")
        except Exception as e:
            print(f"  SKIP {os.path.basename(path)}: {e}")
    if len(all_data) == 0:
        return 0
    if len(all_data) > 50000:
        all_data = random.sample(all_data, 50000)
    replay_buffer.buffer.clear()
    replay_buffer.push(all_data)
    print(f"TOTAL: {len(all_data)} tuples loaded")
    return len(all_data)


def load_checkpoint(champion, device):
    if os.path.exists(CKPT_DRIVE):
        champion.load_state_dict(torch.load(CKPT_DRIVE, map_location=device))
        print("Resumed from DRIVE checkpoint")
        return champion
    if os.path.exists(CKPT_LOCAL):
        champion.load_state_dict(torch.load(CKPT_LOCAL, map_location=device))
        print("Resumed from LOCAL checkpoint")
        return champion
    print("No checkpoint found — starting fresh with NNUE weights")
    return champion


def save_checkpoint(champion):
    for path in (CKPT_DRIVE, CKPT_LOCAL):
        tmp_path = path + ".tmp"
        torch.save(champion.state_dict(), tmp_path)
        os.replace(tmp_path, path)
    print("Checkpoint saved to Drive + local")
    


def log_progress(iteration, loss, policy_loss, value_loss, num_tuples):
    file_exists = os.path.exists(LOG_PATH)
    with open(LOG_PATH, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['iteration', 'loss', 'policy_loss',
                              'value_loss', 'tuples'])
        writer.writerow([iteration,
                          round(float(loss), 5),
                          round(float(policy_loss), 5),
                          round(float(value_loss), 5),
                          num_tuples])


def get_resume_iteration():
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, 'r') as f:
            rows = list(csv.reader(f))
        if len(rows) > 1:
            last_iter = int(rows[-1][0]) + 1
            print(f"Resuming from iteration {last_iter}")
            return last_iter
    return 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load(NNUE_PATH, map_location=device))
    nnue.eval()
    print("Pretrained NNUE loaded!")

    champion = CombinedNetwork(
        pretrained_nnue=nnue,
        num_moves=4672,
        freeze_backbone=True
    ).to(device)

    champion = load_checkpoint(champion, device)

    trainer       = Trainer(champion)
    replay_buffer = ReplayBuffer(max_size=50000)
    iteration     = get_resume_iteration()

    if iteration >= Config.unfreeze_at_iteration:
        print("Resumed past unfreeze point — unfreezing backbone now")
        param_groups = champion.unfreeze_backbone(lr_scale=0.1)
        trainer.optimizer = torch.optim.Adam(param_groups)
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer,
            T_max=Config.num_iterations,
            eta_min=1e-5
        )        

    while True:
        print(f"\n{'='*55}")
        print(f"  TRAINER Iteration {iteration}")
        print(f"{'='*55}")

        num_tuples = load_all_buffers(replay_buffer)

        if num_tuples < Config.batch_size:
            print(f"Need {Config.batch_size}, have {num_tuples}. Waiting 60s...")
            time.sleep(60)
            continue

        result = trainer.train(replay_buffer)

        if result is None:
            print("Training returned None")
            time.sleep(30)
            continue

        if isinstance(result, tuple):
            loss, policy_loss, value_loss = result
        else:
            loss        = result
            policy_loss = 0.0
            value_loss  = 0.0

        print(f"Total loss: {loss:.5f}")

        save_checkpoint(champion)
        log_progress(iteration, loss, policy_loss, value_loss, num_tuples)

        if iteration == Config.unfreeze_at_iteration:
            print("Phase 2: unfreezing backbone")
            param_groups = champion.unfreeze_backbone(lr_scale=0.1)
            trainer.optimizer = torch.optim.Adam(param_groups)
            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optimizer,
                T_max=Config.num_iterations,
                eta_min=1e-5
            )

        iteration += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
