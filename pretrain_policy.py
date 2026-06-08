import pickle, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import chess, os
from Feature_extractor import FeatureExtractor
from policy import PolicyValueNet

class LichessDataset(Dataset):
    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            self.data = pickle.load(f)
        self.extractor = FeatureExtractor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fen, move_uci = self.data[idx]
        board = chess.Board(fen)
        state = self.extractor.board_to_tensor(board)
        move = chess.Move.from_uci(move_uci)
        target = self.extractor.move_to_idx(move)
        # Value is 0.0 for now; we rely on policy (move prediction)
        return state, torch.tensor(target, dtype=torch.long), torch.tensor(0.0)

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("weights", exist_ok=True)
    
    # Load data
    dataset = LichessDataset("lichess_data.pkl")
    loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=2)
    
    # Model & Opt
    model = PolicyValueNet(n_res_blocks=10, channels=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    p_criterion = nn.CrossEntropyLoss()
    
    # Resume Checkpoint if it exists
    start_epoch = 0
    if os.path.exists("weights/checkpoint.pth"):
        checkpoint = torch.load("weights/checkpoint.pth")
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optim_state'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming from epoch {start_epoch}")

    # Training Loop
    model.train()
    for epoch in range(start_epoch, 3):
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/3")
        
        for i, (states, targets, _) in enumerate(pbar):
            states, targets = states.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits, _ = model(states)
            loss = p_criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{running_loss/(i+1):.4f}")
            
        # Save checkpoint
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optim_state': optimizer.state_dict(),
        }, "weights/checkpoint.pth")
        print(f"Checkpoint saved.")

    torch.save(model.state_dict(), "weights/pretrained_policy.pth")
    print("Training finished.")

if __name__ == "__main__":
    train()
