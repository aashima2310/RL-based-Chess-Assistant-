import chess
import torch
import numpy as np
import sys
sys.path.append('/content/RL-based-Chess-Assistant-')
from RL.mcts.node import Node
from RL.chess_env.features import HalfKPExtractor


class MCTS:
    def __init__(self, game, args, model):
        self.game    = game
        self.args    = args
        self.model   = model
        self.device  = next(model.parameters()).device
        self.extractor = HalfKPExtractor()
    def _evaluate_node(self, node):
        w_acc, b_acc = self.extractor.board_to_halfkp(node.state)
        w_acc = w_acc.unsqueeze(0).to(self.device)
        b_acc = b_acc.unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            policy_probs, value = self.model(w_acc, b_acc)

        policy_probs = policy_probs.squeeze(0).cpu().numpy()
        value        = value.item()

        policy_probs *= node.valid_moves
        total = policy_probs.sum()
        if total > 0:
            policy_probs /= total
        else:
            legal = np.where(node.valid_moves == 1)[0]
            policy_probs[legal] = 1.0 / len(legal)

        return policy_probs, value
    def _batch_evaluate(self, nodes):
        valid_nodes = []
        for n in nodes:
            _, is_term = self.game.get_value_and_terminated(
                n.state, n.action_taken
            )
            if not is_term:
                valid_nodes.append(n)

        if not valid_nodes:
            return

        w_list, b_list = [], []
        for node in valid_nodes:
            w, b = self.extractor.board_to_halfkp(node.state)
            w_list.append(w)
            b_list.append(b)
        w_batch = torch.stack(w_list).to(self.device, non_blocking=True)
        b_batch = torch.stack(b_list).to(self.device, non_blocking=True)

        self.model.eval()
        with torch.no_grad():
            policies, values = self.model(w_batch, b_batch)

        policies = policies.cpu().numpy()
        values   = values.cpu().numpy()

        for i, node in enumerate(valid_nodes):
            p     = policies[i] * node.valid_moves
            total = p.sum()
            if total > 0:
                p /= total
            else:
                legal = np.where(node.valid_moves == 1)[0]
                if len(legal) > 0:
                    p[legal] = 1.0 / len(legal)
                else:
                    p = np.zeros(self.game.action_size)


            node.value = values[i].item()
            node.expand(p, node.value)


    def search(self, state):
        root = Node(self.game, self.args, chess.Board(state.board_fen()))
        policy_probs, root_value = self._evaluate_node(root)
        if self.args.get('add_noise', True):
            alpha   = self.args.get('dirichlet_alpha', 0.3)
            epsilon = self.args.get('dirichlet_epsilon', 0.25)
            legal   = np.where(root.valid_moves == 1)[0]
            if len(legal) > 0:
                noise      = np.random.dirichlet([alpha] * len(legal))
                noise_full = np.zeros_like(policy_probs)
                noise_full[legal] = noise
                policy_probs = (1 - epsilon) * policy_probs + epsilon * noise_full

        root.expand(policy_probs, root_value)

        for _ in range(self.args['num_searches']):
            node = root
            path = [node]

            while node.visit_count > 0 and node.policy is not None:
                action = node.select()
                if action is None:
                    break
                if action not in node.children:
                    next_state = self.game.get_next_state(node.state, action)
                    node.children[action] = Node(
                        self.game, self.args, next_state,
                        parent=node, action_taken=action
                    )
                node = node.children[action]
                path.append(node)
            value, is_terminal = self.game.get_value_and_terminated(
                node.state, node.action_taken
            )

            if not is_terminal:
                # FIX 2: Use batched evaluation even for single leaf —
                # keeps all eval logic in one place and GPU-ready
                self._batch_evaluate([node])
                value = node.value
            for n in reversed(path):
                n.back_propagate(value)
                value = self.game.get_opponent_value(value)
        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        total = action_probs.sum()
        if total > 0:
            action_probs /= total
        else:
            legal = np.where(root.valid_moves == 1)[0]
            if len(legal) > 0:
                action_probs[legal] = 1.0 / len(legal)

        return action_probs
