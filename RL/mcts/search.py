import chess
import torch
import numpy as np
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
from RL.mcts.node import Node
from RL.chess_env.features import HalfKPExtractor

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model=model
        self.device = next(model.parameters()).device
        self.nodes = {}
        self.extractor = HalfKPExtractor()

    def get_node(self, board_fen):
        if board_fen not in self.nodes:
            self.nodes[board_fen] = Node(self.game, self.args, chess.Board(board_fen))
        return self.nodes[board_fen]

    def search(self, state):
     original_turn = state.turn
     board_fen_for_root = state.board_fen()
     root = self.get_node(board_fen_for_root)
     root.visit_count = 0
    
     if root.policy is None:
         device = self.args.get('device', torch.device('cpu'))
         w_acc, b_acc = self.extractor.board_to_halfkp(state)
         w_acc = w_acc.to(self.device)
         b_acc = b_acc.to(self.device)
         
        
         with torch.no_grad():
            policy_probs, root_value = self.model(w_acc.unsqueeze(0), b_acc.unsqueeze(0))
         policy_probs = policy_probs.squeeze(0).cpu().numpy()
         policy_probs *= root.valid_moves
         total = policy_probs.sum()
         if total > 0:
            policy_probs /= total
         else:
            legal = np.where(root.valid_moves == 1)[0]
            policy_probs[legal] = 1.0 / len(legal)

         if self.args.get('add_noise', True):
            alpha = self.args.get('dirichlet_alpha', 0.3)
            epsilon = self.args.get('dirichlet_epsilon', 0.25)
            legal_indices = np.where(root.valid_moves == 1)[0]
            noise = np.random.dirichlet([alpha] * len(legal_indices))
            noise_full = np.zeros_like(policy_probs)
            noise_full[legal_indices] = noise
            policy_probs = (1 - epsilon) * policy_probs + epsilon * noise_full

         root.expand(policy_probs, root_value.item())

     for _ in range(self.args["num_searches"]):
        node = root
        path = [node]

        while node.visit_count > 0 and node.policy is not None:
            action = node.select()
            if action is None:
                break
            if action not in node.children:
                next_abs_state = self.game.get_next_state(node.state, action)
                node.children[action] = Node(self.game, self.args, next_abs_state,
                                              parent=node, action_taken=action)
            node = node.children[action]
            path.append(node)

        value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

        if not is_terminal:
            w_acc, b_acc = self.extractor.board_to_halfkp(node.state)
            w_acc = w_acc.to(device)
            b_acc = b_acc.to(device)
            with torch.no_grad():
                policy_probs, value = self.model(w_acc.unsqueeze(0), b_acc.unsqueeze(0))
            policy_probs = policy_probs.squeeze(0).cpu().numpy()
            value = value.item()

            policy_probs *= node.valid_moves
            total_policy = np.sum(policy_probs)
            if total_policy > 0:
                policy_probs /= total_policy
            else:
                legal_actions_indices = np.where(node.valid_moves == 1)[0]
                if len(legal_actions_indices) > 0:
                    policy_probs[legal_actions_indices] = 1.0 / len(legal_actions_indices)
                else:
                    value = 0
                    policy_probs = np.zeros(self.game.action_size)

            node.expand(policy_probs, value)

        for node_to_update in reversed(path):
            node_to_update.back_propagate(value)
            value = self.game.get_opponent_value(value)

     action_probs = np.zeros(self.game.action_size)
     for action, child in root.children.items():
        action_probs[action] = child.visit_count

     total_visits = np.sum(action_probs)
     if total_visits > 0:
        action_probs /= total_visits
     else:
        legal_actions_indices = np.where(root.valid_moves == 1)[0]
        if len(legal_actions_indices) > 0:
            action_probs[legal_actions_indices] = 1.0 / len(legal_actions_indices)

     return action_probs

