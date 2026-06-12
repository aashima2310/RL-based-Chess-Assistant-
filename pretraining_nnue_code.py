
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pickle
import chess
import numpy as np
from tqdm import tqdm
import os

class HalfKPExtractor:
    input_size = 40960

    PIECE_TYPE_INDEX = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.PAWN,   chess.BLACK): 1,
        (chess.KNIGHT, chess.WHITE): 2,
        (chess.KNIGHT, chess.BLACK): 3,
        (chess.BISHOP, chess.WHITE): 4,
        (chess.BISHOP, chess.BLACK): 5,
        (chess.ROOK,   chess.WHITE): 6,
        (chess.ROOK,   chess.BLACK): 7,
        (chess.QUEEN,  chess.WHITE): 8,
        (chess.QUEEN,  chess.BLACK): 9,
    }

    def get_halfkp_indices(self, board: chess.Board, turn: bool) -> list:
        king_sq = board.king(turn)
        if turn == chess.BLACK:
            king_sq = chess.square_mirror(king_sq)

        active = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None or piece.piece_type == chess.KING:
                continue  

            if turn == chess.BLACK:
                relative_color = not piece.color
                sq = chess.square_mirror(square)
            else:
                relative_color = piece.color
                sq = square

            pt_idx = self.PIECE_TYPE_INDEX[(piece.piece_type, relative_color)]
            feature_idx = king_sq * 640 + sq * 10 + pt_idx
            active.append(feature_idx)

        return active

class NNUE(nn.Module):
    def __init__(self, input_size=40960):
        super().__init__()
        self.input_weights = nn.Parameter(torch.randn(input_size, 512) * 0.01)
        self.input_bias = nn.Parameter(torch.zeros(512))
        self.l2 = nn.Linear(1024, 32)
        self.l3 = nn.Linear(32, 32)
        self.l4 = nn.Linear(32, 1)

    def clipped_relu(self, x):
        return torch.clamp(x, min=0, max=1)

    def forward(self, w_acc: torch.Tensor, b_acc: torch.Tensor) -> torch.Tensor:
        if w_acc.dim() == 1:
            w_acc = w_acc.unsqueeze(0)
        if b_acc.dim() == 1:
            b_acc = b_acc.unsqueeze(0)
            
        x = torch.cat([w_acc, b_acc], dim=1)
        x = self.clipped_relu(x)
        x = self.clipped_relu(self.l2(x))
        x = self.clipped_relu(self.l3(x))
        x = self.l4(x)
        return x

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)

class ChessDataset(Dataset):
    def __init__(self, pkl_path, extractor):
        print(f"Loading master dataset from {pkl_path}...")
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.extractor = extractor
        print(f"Dataset completely loaded! Total positions: {len(self.data):,}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, score = self.data[idx]
        board = chess.Board(fen)
        
        w_idx = self.extractor.get_halfkp_indices(board, chess.WHITE)
        b_idx = self.extractor.get_halfkp_indices(board, chess.BLACK)
        
        w_idx_tensor = torch.tensor(w_idx, dtype=torch.long)
        b_idx_tensor = torch.tensor(b_idx, dtype=torch.long)
        target_tensor = torch.tensor([score], dtype=torch.float32)
        
        return w_idx_tensor, b_idx_tensor, target_tensor

def collate_fn(batch):
    """Handles variable length feature lists within a training batch."""
    w_indices, b_indices, targets = zip(*batch)
    targets = torch.stack(targets)
    return w_indices, b_indices, targets


def train_nnue(model, train_loader, val_loader, epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training actively running on hardware accelerator: {device}\n")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        
        for w_idx_batch, b_idx_batch, targets in train_pbar:
            targets = targets.to(device)
            optimizer.zero_grad()
            
            w_accs = []
            b_accs = []
            

            for w_idx, b_idx in zip(w_idx_batch, b_idx_batch):
                w_acc = model.input_bias + model.input_weights[w_idx.to(device)].sum(dim=0)
                b_acc = model.input_bias + model.input_weights[b_idx.to(device)].sum(dim=0)
                w_accs.append(w_acc)
                b_accs.append(b_acc)
                
            w_acc_tensor = torch.stack(w_accs)
            b_acc_tensor = torch.stack(b_accs)

            predictions = model(w_acc_tensor, b_acc_tensor)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for w_idx_batch, b_idx_batch, targets in val_loader:
                targets = targets.to(device)
                
                w_accs = []
                b_accs = []
                for w_idx, b_idx in zip(w_idx_batch, b_idx_batch):
                    w_acc = model.input_bias + model.input_weights[w_idx.to(device)].sum(dim=0)
                    b_acc = model.input_bias + model.input_weights[b_idx.to(device)].sum(dim=0)
                    w_accs.append(w_acc)
                    b_accs.append(b_acc)
                    
                w_acc_tensor = torch.stack(w_accs)
                b_acc_tensor = torch.stack(b_accs)

                predictions = model(w_acc_tensor, b_acc_tensor)
                val_loss = criterion(predictions, targets)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        
        print(f"\n✨ Epoch {epoch+1} Results -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Save structural checkpoint
        save_path = f"/content/drive/MyDrive/chess_data/nnue_epoch_{epoch+1}.pth"
        model.save(save_path)
        print(f"💾 Saved model checkpoint safely to: {save_path}\n" + "-"*50)

if __name__ == "__main__":

    extractor = HalfKPExtractor()
    nnue_model = NNUE(input_size=40960)
    
    dataset_path = "/content/drive/MyDrive/chess_data/master_training_dataset.pkl"
    full_dataset = ChessDataset(dataset_path, extractor) 
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=512, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    train_nnue(nnue_model, train_loader, val_loader, epochs=10, lr=0.001)
