import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)
from RL.chess_env.features import HalfKPExtractor

halfkp_extractor = HalfKPExtractor()


class NNUE_AlphaZero(nn.Module):
    def __init__(self, pretrained_nnue, num_moves=4672, freeze_backbone=True):
        super().__init__()
        self.backbone = pretrained_nnue
        self.trunk = nn.Linear(32, 128)
        self.policy_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_moves)
        )
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        if freeze_backbone:
            self._freeze_backbone()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, lr_scale=0.1):
        for param in self.backbone.parameters():
            param.requires_grad = True
        return [
            {'params': self.backbone.parameters(),   'lr': 1e-4 * lr_scale},
            {'params': self.trunk.parameters(),       'lr': 1e-4},
            {'params': self.policy_head.parameters(), 'lr': 1e-4},
            {'params': self.value_head.parameters(),  'lr': 1e-4},
        ]

    def forward(self, w_acc, b_acc, board=None):
        device = w_acc.device
        legal_move_mask = None
        if board is not None:
            legal_move_mask = torch.zeros(4672, dtype=torch.bool, device=w_acc.device)
            for move in board.legal_moves:
                legal_move_mask[halfkp_extractor.move_to_idx(move)] = True
            if w_acc.dim() == 2:
                legal_move_mask = legal_move_mask.unsqueeze(0).expand(w_acc.size(0), -1)
        input_weights = self.backbone.input_weights.to(device)
        input_bias = self.backbone.input_bias.to(device)

        w_processed = torch.matmul(w_acc, input_weights) + input_bias
        b_processed = torch.matmul(b_acc, input_weights) +input_bias
        x = torch.cat([w_processed, b_processed], dim=1)
        x = self.backbone.clipped_relu(x)
        x = self.backbone.clipped_relu(self.backbone.l2(x))
        x = self.backbone.clipped_relu(self.backbone.l3(x))

        trunk = F.relu(self.trunk(x))

        policy_logits = self.policy_head(trunk)
        if legal_move_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_move_mask, -1e9)
        policy = F.softmax(policy_logits, dim=-1)

        value = self.value_head(trunk)

        return policy, value

    def refresh_accumulator(self, active_features):
        return self.backbone.refresh_accumulator(active_features)

    def update_accumulator(self, accumulator, added, removed):
        return self.backbone.update_accumulator(accumulator, added, removed)


def alphazero_loss(policy_pred, value_pred, policy_target, value_target, model, l2_lambda=1e-4):
    policy_loss = -(policy_target * torch.log(policy_pred + 1e-8)).sum(dim=-1).mean()
    value_loss = F.mse_loss(value_pred.squeeze(-1), value_target.squeeze(-1))
    l2_loss = sum(p.pow(2).sum() for p in model.parameters() if p.requires_grad)
    return policy_loss + value_loss + l2_lambda * l2_loss
