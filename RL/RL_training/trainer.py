import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from chess_env.features import board_to_tensor_769
from config import Config
from model import alphazero_loss


class Trainer:

    def __init__(self, network):
        self.network = network
        self.device = next(network.parameters()).device
        self.optimizer = optim.Adam([
            {'params': self.network.trunk.parameters(),        'lr': Config.lr},
            {'params': self.network.policy_head.parameters(),  'lr': Config.lr},
            {'params': self.network.value_head.parameters(),   'lr': Config.lr * 0.5},
        ], weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=Config.num_iterations if hasattr(Config, 'num_iterations') else 200,
            eta_min=1e-5
        )

    def train(self, replay_buffer):

        if len(replay_buffer) < Config.batch_size:
            return None
        self.network.train()
        total_loss = 0
        total_policy = 0.0
        total_value  = 0.0

        for epoch in range(Config.epochs):

            states, policies, values = replay_buffer.sample(Config.batch_size)
            w_accs, b_accs = [], []
            for s in states:
                w, b = halfkp_extractor.board_to_halfkp(s)
                w_accs.append(w)
                b_accs.append(b)
            
            w_acc = torch.stack(w_accs).to(self.device, non_blocking=True)
            b_acc = torch.stack(b_accs).to(self.device, non_blocking=True)
            policy_targets = torch.tensor(np.array(policies), dtype=torch.float32).to(self.device, non_blocking=True)
            value_targets = torch.tensor(np.array(values), dtype=torch.float32 ).unsqueeze(1).to(self.device, non_blocking=True)

            policy_preds, value_preds = self.network(w_acc, b_acc)

            loss, policy_loss, value_loss = alphazero_loss(
                policy_preds, value_preds,
                policy_targets, value_targets,
                model=self.network
            )
            self.optimizer.zero_grad()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
            total_policy += policy_loss.item()
            total_value  += value_loss.item()

        avg_loss   = total_loss   / Config.epochs
        avg_policy = total_policy / Config.epochs
        avg_value  = total_value  / Config.epochs
        current_lr = self.scheduler.get_last_lr()[0]
        print(f"  Policy loss: {avg_policy:.4f} | "
              f"Value loss: {avg_value:.4f} | "
              f"LR: {current_lr:.6f}")
 
        return avg_loss
