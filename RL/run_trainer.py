import pickle
import os
import time
import torch
from RL_training.trainer import Trainer
from RL_training.replay_buffer import ReplayBuffer
from RL_training.champion import evaluate
from RL.combined_network import CombinedNetwork
from RL.checkpoints import save_checkpoint, load_checkpoint
from RL.training_stats import log
from RL.config import Config

buffer_path = "buffer.pkl"

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
    champion = CombinedNetwork()
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

        challenger = CombinedNetwork()
        challenger.load_state_dict(champion.state_dict())

        is_better = evaluate(champion, challenger)

        if is_better:
            save_checkpoint(champion, "checkpoint.pt")
            print("Champion updated and saved")
            
        else:
            print("Champion unchanged")

        estimated_elo = 800 + iteration * 60
        log(iteration, loss, estimated_elo, num_tuples)

        iteration += 1
        time.sleep(30)


if __name__ == "__main__":
    main()
