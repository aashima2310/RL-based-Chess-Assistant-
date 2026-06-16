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
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=Config.lr
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

    def train(self, replay_buffer):

        if len(replay_buffer) < Config.batch_size:
            return None

        total_loss = 0

        for epoch in range(Config.epochs):

            states, policies, values = replay_buffer.sample(Config.batch_size)
            w_accs, b_accs = [], []
            for s in states:
                w, b = halfkp_extractor.board_to_halfkp(s)
                w_accs.append(w)
                b_accs.append(b)
            
            w_acc = torch.stack(w_accs).to(self.device)
            b_acc = torch.stack(b_accs).to(self.device)
            policy_targets = torch.tensor(policies, dtype=torch.float32)
            value_targets = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

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


        return total_loss / Config.epochs
