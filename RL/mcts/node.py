import numpy as np
import math
import chess
import torch
from RL.chess_env.features import HalfKPExtractor
halfkp_extractor = HalfKPExtractor()

class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None, policy_from_nn=None):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.policy = policy_from_nn
        self.valid_moves = game.get_valid_moves(state).numpy()

    def select(self):
        best_child = None
        best_ucb = -np.inf
        legal_actions = np.where(self.valid_moves == 1)[0]

        unvisited = [a for a in legal_actions if a not in self.children]
        if unvisited:
            return max(unvisited,
                key=lambda a: self.policy[a] if self.policy is not None else 0)

        for action in legal_actions:
            if self.policy is not None and self.policy[action] == 0:
                continue  
            if action in self.children:
                child = self.children[action]
                q_value = -child.value_sum / child.visit_count if child.visit_count > 0 else 0
                ucb = q_value + self.args['C'] * self.policy[action] * (
                    math.sqrt(self.visit_count) / (1 + child.visit_count))
            else:
                ucb = self.args['C'] * (self.policy[action] if self.policy is not None else 1.0) * math.sqrt(self.visit_count)

            if ucb > best_ucb:
                best_ucb = ucb
                best_child = action

        if best_child is None:
            if len(legal_actions) > 0:
                return np.random.choice(legal_actions)
            return None
        return best_child

    def expand(self, policy_probs, value=None):
        self.policy = policy_probs

    def back_propagate(self, value):
        self.value_sum += value
        self.visit_count += 1
