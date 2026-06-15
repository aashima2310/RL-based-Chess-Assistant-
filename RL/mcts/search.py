import chess
import torch
import numpy as np
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
from RL.mcts.node import Node
from RL.chess_env.features import HalfKPExtractor

halfkp_extractor = HalfKPExtractor()

class MCTS:
    def __init__(self, game, args, model):
        self.game = game
        self.args = args
        self.model = model
        self.nodes = {}

    def get_node(self, board_fen):
        if board_fen not in self.nodes:
            self.nodes[board_fen] = Node(self.game, self.args, chess.Board(board_fen))
        return self.nodes[board_fen]

    def search(self, state):
        original_turn = state.turn
        canonical_state_for_nn_input = state.copy()
        if original_turn == chess.BLACK:
            canonical_state_for_nn_input = canonical_state_for_nn_input.mirror()

        board_fen_for_root = canonical_state_for_nn_input.board_fen()
        root = self.get_node(board_fen_for_root)
        root.visit_count = 0

        if root.policy is None:
            w_acc, b_acc = halfkp_extractor.board_to_halfkp(canonical_state_for_nn_input)
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
                alpha   = self.args.get('dirichlet_alpha', 0.3)
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
                    canonical_next_state = self.game.change_perspective(next_abs_state)
                    node.children[action] = Node(self.game, self.args, canonical_next_state,
                                                  parent=node, action_taken=action)
                node = node.children[action]
                path.append(node)

            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken)

            if not is_terminal:
                w_acc, b_acc = halfkp_extractor.board_to_halfkp(node.state)
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

        action_probs_canonical = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs_canonical[action] = child.visit_count

        total_visits = np.sum(action_probs_canonical)
        if total_visits > 0:
            action_probs_canonical /= total_visits
        else:
            legal_actions_indices = np.where(root.valid_moves == 1)[0]
            if len(legal_actions_indices) > 0:
                action_probs_canonical[legal_actions_indices] = 1.0 / len(legal_actions_indices)

        if original_turn == chess.BLACK:
            action_probs_original_perspective = np.zeros(self.game.action_size)
            for canonical_action_idx in range(self.game.action_size):
                if action_probs_canonical[canonical_action_idx] > 0:
                    canonical_move = halfkp_extractor.idx_to_move(canonical_action_idx)
                    original_move = chess.Move(
                        chess.square_mirror(canonical_move.from_square),
                        chess.square_mirror(canonical_move.to_square),
                        promotion=canonical_move.promotion
                    )
                    original_action_idx = halfkp_extractor.move_to_idx(original_move)
                    action_probs_original_perspective[original_action_idx] = action_probs_canonical[canonical_action_idx]
            return action_probs_original_perspective
        else:
            return action_probs_canonical

