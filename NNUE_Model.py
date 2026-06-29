import torch
import torch.nn as nn
import os

class NNUE(nn.Module):
    def __init__(self,input_size=40960):
      super().__init__()
      self.input_weights=nn.Parameter(torch.randn(input_size,512)*0.01)
      self.input_bias=nn.Parameter(torch.zeros(512))
      self.l2=nn.Linear(1024,32)
      self.l3=nn.Linear(32,32)
      self.l4=nn.Linear(32,1)
    def clipped_relu(self,x):
        return torch.clamp(x,min=0,max=1)
    def refresh_accumulator(self,active_features : list)-> torch.Tensor:
        acc=self.input_bias.clone()
        for i in active_features:
          acc=acc+self.input_weights[i]
        return acc
    def update_accumulator(self,accumulator : torch.Tensor,added_features: list,removed_features: list) -> torch.Tensor:
        acc=accumulator.clone()
        for i in added_features:
          acc=acc+self.input_weights[i]
        for i in removed_features:
          acc=acc-self.input_weights[i]
        return acc
    def forward(self, w_acc : torch.Tensor, b_acc : torch.Tensor) -> torch.Tensor:
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
    def evaluate_board(
        self,
        board,
        extractor
    ) -> float:
        self.eval()
        with torch.no_grad():
            w_idx = extractor.get_halfkp_indices(board, True)
            b_idx = extractor.get_halfkp_indices(board, False)
            w_acc = self.refresh_accumulator(w_idx)
            b_acc = self.refresh_accumulator(b_idx)
            score = self.forward(w_acc, b_acc)
            return score.item()

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.state_dict(), path)
        print(f"NNUE saved to {path}")

    def load_weights(self, path: str):
        self.load_state_dict(
            torch.load(path, map_location='cpu')
        )
        print(f"NNUE loaded from {path}")
