import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append('/content/repo')
from RL.chess_env.features import HalfKPExtractor

_extractor = HalfKPExtractor()


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

    def refresh_accumulator(self, active_features: list) -> torch.Tensor:
        acc = self.input_bias.clone()
        for i in active_features:
            acc = acc + self.input_weights[i]
        return acc

    def forward_features(self, w_acc: torch.Tensor, b_acc: torch.Tensor) -> torch.Tensor:
        if w_acc.dim() == 1:
            w_acc = w_acc.unsqueeze(0)
        if b_acc.dim() == 1:
            b_acc = b_acc.unsqueeze(0)
        x = torch.cat([w_acc, b_acc], dim=1)
        x = self.clipped_relu(x)
        x = self.clipped_relu(self.l2(x))
        x = self.clipped_relu(self.l3(x))   # (B, 32)
        return x


class NNUE_AlphaZero(nn.Module):
    def __init__(self, input_size=40960, policy_size=4672):
        super().__init__()
        self.backbone = NNUE(input_size=input_size)

        self.trunk = nn.Linear(32, 128)

        # OPTION A: Upgraded high-capacity policy head with LayerNorm blocks
        self.policy_head = nn.Sequential(
            nn.Linear(128, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Linear(1024, policy_size),
        )
        
        # Keeping the value head consistent so it maps seamlessly to value checkpoints
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def _build_legal_mask_batch(self, boards, device):
        masks = []
        for b in boards:
            m = _extractor.get_legal_moves(b)
            masks.append(m.bool())
        return torch.stack(masks).to(device)

    def forward(self, w_acc: torch.Tensor, b_acc: torch.Tensor, board=None):
        feats = self.backbone.forward_features(w_acc, b_acc)
        trunk_out = F.relu(self.trunk(feats))

        policy_logits = self.policy_head(trunk_out)
        value = self.value_head(trunk_out)

        if board is not None:
            mask = self._build_legal_mask_batch(board, policy_logits.device)
            policy_logits = policy_logits.masked_fill(~mask, -1e9)

        policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value

    def save(self, path, optimizer=None, scheduler=None, epoch=None, batch=None, best_val=None):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ckpt = {"model_state": self.state_dict()}
        if optimizer is not None: ckpt["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None: ckpt["scheduler_state"] = scheduler.state_dict()
        if epoch is not None: ckpt["epoch"] = epoch
        if batch is not None: ckpt["batch"] = batch
        if best_val is not None: ckpt["best_val"] = best_val
        torch.save(ckpt, path)
        print(f"Checkpoint saved to {path}")

    def load_weights(self, path, device="cpu"):
        ckpt = torch.load(path, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing: print("MISSING keys:", missing)
        if unexpected: print("UNEXPECTED keys:", unexpected)
        return ckpt


def alphazero_loss(policy_preds, value_preds, policy_targets, value_targets, model=None, l2_lambda=1e-4):
    eps = 1e-8
    policy_loss = -(policy_targets * torch.log(policy_preds + eps)).sum(dim=1).mean()
    value_loss = F.mse_loss(value_preds, value_targets)
    
    l2_reg = 0.0
    if model is not None and l2_lambda > 0:
        l2_reg = l2_lambda * sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
        
    total_loss = policy_loss + value_loss + l2_reg
    return total_loss, policy_loss, value_loss
