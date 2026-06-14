import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from chess_env.features import board_to_tensor_769
from config import Config


class Trainer:

    def __init__(self, network):
        self.network = network
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
            state_tensors = torch.stack([
                board_to_tensor_769(s) for s in states
            ])

            policy_targets = torch.tensor(policies, dtype=torch.float32)
            value_targets = torch.tensor(values, dtype=torch.float32).unsqueeze(1)

            policy_preds, value_preds = self.network(state_tensors)

            policy_loss = self.policy_loss_fn(policy_preds, policy_targets)
            value_loss = self.value_loss_fn(value_preds, value_targets)
          
            loss = policy_loss + value_loss
            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()


        return total_loss / Config.epochs
