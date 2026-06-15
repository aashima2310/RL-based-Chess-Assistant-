
import numpy as np
import math


import chess
import torch
import numpy as np
from RL.chess_env.features 
import HalfKPExtractor
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
    self.valid_moves = game.get_valid_moves(state)

  def select(self):
    best_child = None
    best_ucb = -np.inf


    for action in range(self.game.action_size):
      if self.valid_moves[action] == 1:
        if action in self.children:
          child = self.children[action]
          q_value = child.value_sum / child.visit_count if child.visit_count > 0 else 0
          if self.policy[action] == 0:
            ucb = -np.inf
          else:
            ucb = q_value + self.args['C'] * self.policy[action] * (math.sqrt(self.visit_count) / (1 + child.visit_count))
        else:
          q_value = 0
          if self.policy is None or self.policy[action] == 0:
            ucb = -np.inf
          else:
            ucb = q_value + self.args['C'] * self.policy[action] * (math.sqrt(self.visit_count) / (1 + 0))

        if ucb > best_ucb:
          best_ucb = ucb
          best_child = action

    if best_child is None:
      legal_actions = np.where(self.valid_moves == 1)[0]
      if len(legal_actions) > 0:
        return np.random.choice(legal_actions)
      else:
        return None

    return best_child

  def expand(self, policy_probs, value):
    self.policy = policy_probs
    self.value_sum = value
    self.visit_count = 1

  def back_propagate(self, value):
    self.value_sum += value
    self.visit_count += 1

    if self.parent is not None:
      self.parent.back_propagate(self.game.get_opponent_value(value))
