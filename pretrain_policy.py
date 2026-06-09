import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
import chess
import os
import sys
sys.path.append('..')

from Feature_extractor import HalfKPExtractor
from policy import PolicyValueNet


class LichessDataset(Dataset):

    def __init__(self, data_path: str):
        print(f"Loading Lichess data from {data_path}...")
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Loaded {len(self.data)} positions.")
        self.extractor = HalfKPExtractor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move_uci = self.data[idx]
        board = chess.Board(fen)

        features = self.extractor.board_to_tensor_769(board)

        move = chess.Move.from_uci(move_uci)
        move_idx = self.extractor.move_to_idx(move)

        return features, torch.tensor(move_idx, dtype=torch.long)


def pretrain_policy(
    data_path: str = "data/lichess_data.pkl",
    epochs: int = 3,
    batch_size: int = 512,
    learning_rate: float = 0.001,
    save_path: str = "weights/pretrained_policy.pth"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = LichessDataset(data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )

    model = PolicyValueNet(n_res_blocks=6, channels=128).to(device)
    print(f"Policy parameters: {model.count_params():,}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss_fn = nn.CrossEntropyLoss()

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, (features, targets) in enumerate(dataloader):
            features = features.to(device)
            targets  = targets.to(device)

            policy_logits, _ = model(features)

            loss = policy_loss_fn(policy_logits, targets)


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            predicted = policy_logits.argmax(dim=1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)

            if batch_idx % 200 == 0:
                avg_loss = total_loss / num_batches
                accuracy = correct / total * 100
                print(f"Epoch {epoch+1} | "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Accuracy: {accuracy:.2f}%")

        epoch_loss = total_loss / num_batches
        epoch_acc  = correct / total * 100
        print(f"Epoch {epoch+1} done | "
              f"Loss: {epoch_loss:.4f} | "
              f"Accuracy: {epoch_acc:.2f}%")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            model.save(save_path)
            print(f"Best model saved. Loss: {best_loss:.4f}")

    print(f"Policy pretraining done. Best loss: {best_loss:.4f}")
    return model


if __name__ == "__main__":
    pretrain_policy()
