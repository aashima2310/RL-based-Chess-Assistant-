import pickle
import os
import time
import sys
import torch
sys.path.append('/teamspace/studios/this_studio/RL-based-Chess-Assistant-')
from RL.RL_training.trainer import Trainer
from RL.RL_training.replay_buffer import ReplayBuffer
from RL.RL_training.champion import evaluate
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.checkpoints import save_checkpoint, load_checkpoint
from RL.training_stats import log
from RL.config import Config
from pretraining_nnue_code import NNUE

buffer_path = "/teamspace/studios/this_studio/drive/MyDrive/chess_rl/buffer.pkl"

def load_buffer_into_ram(replay_buffer):
    if not os.path.exists(buffer_path):
        print("Buffer file not found yet, waiting for workers...")
        return 0
    with open(buffer_path, 'rb') as f:
        data = pickle.load(f)
    replay_buffer.buffer.clear()
    replay_buffer.push(data)
    print(f"Loaded {len(data)} tuples into replay buffer")
    return len(data)

def main():
    # load pretrained NNUE backbone
    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load('/teamspace/studios/this_studio/RL-based-Chess-Assistant-/nnue.pth', map_location='cpu'))
    nnue.eval()
    print("Pretrained NNUE loaded!")

    # wrap in AlphaZero model
    champion = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=True)
    champion = load_checkpoint(champion, "checkpoint.pt")

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
        is_better = evaluate(champion, challenger)

        if is_better:
            ckpt_name = f"checkpoint_iter_{iteration}.pt"
            save_checkpoint(champion, "checkpoint.pt")
            print("Champion updated and saved")
            os.system(f'git add {ckpt_name}')
            os.system(f'git commit -m "checkpoint iteration {iteration}"')
            os.system('git push')
        else:
            print("Champion unchanged")

        estimated_elo = 800 + iteration * 60
        log(iteration, loss, estimated_elo, num_tuples)
        iteration += 1
        time.sleep(30)

if __name__ == "__main__":
    main()
