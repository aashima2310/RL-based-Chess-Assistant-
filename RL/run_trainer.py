import pickle
import os
import time
import sys
import glob
import torch
import shutil
import gdown

sys.path.append('/teamspace/studios/this_studio/RL-based-Chess-Assistant-')
FOLDER_ID = "1q6OC-jQTiWvTaudDCZzpWeBCErWZmkkO" 
LOCAL_DOWNLOAD_DIR = './downloaded_worker_buffers'
NNUE_PATH  = '/teamspace/studios/this_studio/RL-based-Chess-Assistant-/nnue.pth'
VERSION_FILE = './latest_checkpoint_version.txt'

from RL.RL_training.trainer import Trainer
from RL.RL_training.replay_buffer import ReplayBuffer
from RL.RL_training.champion import evaluate
from RL.combined_network import NNUE_AlphaZero as CombinedNetwork
from RL.checkpoints import save_checkpoint, load_checkpoint
from RL.training_stats import log
from RL.config import Config
from pretraining_nnue_code import NNUE
processed_files = set()

def clean_old_checkpoints(workspace_path, keep_latest=3):
    checkpoint_files = glob.glob(os.path.join(workspace_path, "checkpoint_iter_*.pt"))
    checkpoint_files.sort(key=os.path.getmtime)
    if len(checkpoint_files) > keep_latest:
        files_to_delete = checkpoint_files[:-keep_latest]
        for file_path in files_to_delete:
            try:
                os.remove(file_path)
                print(f"🧹 Storage Cleanup: Deleted old local checkpoint {os.path.basename(file_path)}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")

def load_buffer_into_ram(replay_buffer):
    if os.path.exists(LOCAL_DOWNLOAD_DIR):
        shutil.rmtree(LOCAL_DOWNLOAD_DIR)
    os.makedirs(LOCAL_DOWNLOAD_DIR, exist_ok=True)

    print("🔄 Downloading latest worker buffers from Google Drive Folder via gdown...")
    try:
        # Pull down folder contents (Compatibility: argument 'remaining_ok' removed)
        gdown.download_folder(f'https://drive.google.com/drive/folders/{FOLDER_ID}', 
                              output=LOCAL_DOWNLOAD_DIR, quiet=True)
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return len(replay_buffer.buffer)

    # Scan the freshly downloaded directory for worker files
    buffer_files = glob.glob(os.path.join(LOCAL_DOWNLOAD_DIR, 'buffer_*.pkl'))

    if not buffer_files:
        print("No buffer files found in the Drive folder yet, waiting for workers...")
        return len(replay_buffer.buffer)

    new_tuples_added = 0
    all_data=[]
    
    for path in buffer_files:
        file_name = os.path.basename(path)
        
        # FIX: Avoid training on the exact same data multiple times
        if file_name in processed_files:
            continue
            
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            
            # Unpack and push valid data tuples into memory
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, (tuple, list)) and len(item) == 3:
                        state, policy, value = item
                        if hasattr(state, 'fen'):
                            state = board_to_tensor(state)
                        all_data.append(item)
                        new_tuples_added += 1
            else:
                if isinstance(data, (tuple, list)) and len(data) == 3:
                    all_data.append(item)
                    new_tuples_added += 1
                    
            processed_files.add(file_name)
            print(f" Processed unique data from new file: {file_name}")
            
        except Exception as e:
            print(f"Skipping corrupt file {file_name}: {e}")
            continue

    print(f"TOTAL ACTIVE BUFFER SIZE IN RAM: {len(replay_buffer.buffer)} (Added {new_tuples_added} fresh tuples)")
    for tuple_item in all_data:
        try:
            replay_buffer.push(tuple_item)
        except TypeError:
            if len(tuple_item) == 3:
                replay_buffer.push(tuple_item[0], tuple_item[1], tuple_item[2])
    print(f"TOTAL ACTIVE BUFFER SIZE IN RAM: {len(replay_buffer.buffer)}")
    shutil.rmtree(LOCAL_DOWNLOAD_DIR)
    return len(replay_buffer.buffer)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nnue = NNUE(input_size=40960)
    nnue.load_state_dict(torch.load(NNUE_PATH, map_location=device))
    nnue.eval()
    print("Pretrained NNUE loaded!")

    champion = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=True)
    champion = champion.to(device)
    resumed = False
    if os.path.exists(VERSION_FILE):
        try:
            with open(VERSION_FILE, 'r') as f:
                latest_ckpt_name = f.read().strip()
            actual_ckpt_path = f"./{latest_ckpt_name}"
            if os.path.exists(actual_ckpt_path):
                champion.load_state_dict(torch.load(actual_ckpt_path, map_location=device))
                print(f"Resumed from existing local checkpoint: {latest_ckpt_name}")
                resumed = True
        except Exception as e:
            print(f"Could not read local pointer file: {e}")

    if not resumed:
        print("No dynamic checkpoint found, starting fresh with base NNUE weights")

    trainer = Trainer(champion)
    replay_buffer = ReplayBuffer(max_size=50000)
    iteration = 0

    while True:
        print(f"\n=== TRAINER: Iteration {iteration} ===")

        num_tuples = load_buffer_into_ram(replay_buffer)

        if num_tuples < Config.batch_size:
            print(f"Not enough data yet ({num_tuples} tuples). Need at least {Config.batch_size}. Waiting 60s...")
            time.sleep(60)
            continue

        loss = trainer.train(replay_buffer)
        print(f"Training loss: {loss:.4f}")

        challenger = CombinedNetwork(pretrained_nnue=nnue, num_moves=4672, freeze_backbone=False)
        challenger.load_state_dict(champion.state_dict())
        challenger = challenger.to(device)

        is_better = evaluate(champion, challenger)

        if is_better:
            ckpt_name = f"checkpoint_iter_{iteration}.pt"
            # Saves checkpoints locally inside Lightning Studio workspace
            save_checkpoint(champion, f"./{ckpt_name}") 
            
            with open(VERSION_FILE, "w") as f:
                f.write(ckpt_name)
                
            print(f"Champion updated and saved locally as {ckpt_name}")
            clean_old_checkpoints("./", keep_latest=3)
            
            # Use Git to automatically push the new version pointers to GitHub
            os.system(f'git add {VERSION_FILE} {ckpt_name}')
            os.system(f'git commit -m "checkpoint iteration {iteration}"')
            os.system('git push')
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
