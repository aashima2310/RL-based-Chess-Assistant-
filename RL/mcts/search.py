import chess, torch, numpy as np, math
from RL.chess_env.features import HalfKPExtractor
from RL.mcts.node import Node

class MCTS:
    def __init__(self, game, args, model):
        self.game      = game
        self.args      = args
        self.model     = model
        self.device    = next(model.parameters()).device
        self.extractor = HalfKPExtractor()

    def _get_policy_value(self, board):
        w = self.extractor.indices_to_tensor(
            self.extractor.get_halfkp_indices(board, chess.WHITE)
        ).unsqueeze(0).to(self.device)
        b = self.extractor.indices_to_tensor(
            self.extractor.get_halfkp_indices(board, chess.BLACK)
        ).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pol, val = self.model(w, b, [board])
        pol = pol.squeeze(0).cpu().numpy()
        # mask to legal only
        mask = self.game.get_valid_moves(board).numpy()
        pol  = pol * mask
        s    = pol.sum()
        if s > 0:
            pol /= s
        else:
            legal = np.where(mask == 1)[0]
            pol[legal] = 1.0 / len(legal)
        # flip value if black to move (value is always from current player perspective)
        v = val.item()
        if board.turn == chess.BLACK:
            v = -v
        return pol, v

    def search(self, state):
        root = Node(self.game, self.args,
                    chess.Board(state.fen()))
        pol, val = self._get_policy_value(root.state)

        if self.args.get("add_noise", False):
            alpha   = self.args.get("dirichlet_alpha", 0.3)
            eps     = self.args.get("dirichlet_epsilon", 0.25)
            legal   = np.where(self.game.get_valid_moves(root.state).numpy()==1)[0]
            noise   = np.random.dirichlet([alpha]*len(legal))
            nf      = np.zeros_like(pol); nf[legal] = noise
            pol     = (1-eps)*pol + eps*nf

        root.expand(pol, val)

        for _ in range(self.args["num_searches"]):
            node = root
            path = [node]

            # --- SELECT ---
            while node.visit_count > 1 and node.policy is not None:
                action = node.select()
                if action is None:
                    break
                if action not in node.children:
                    ns = self.game.get_next_state(node.state, action)
                    node.children[action] = Node(
                        self.game, self.args, ns,
                        parent=node, action_taken=action)
                node = node.children[action]
                path.append(node)

            board = node.state
            if board.is_game_over(claim_draw=True):
                outcome = board.outcome(claim_draw=True)
                if outcome and outcome.winner is not None:
                    value = 1.0  
                else:
                    value = 0.0
            else:
                pol2, value = self._get_policy_value(board)
                node.expand(pol2, value)

            for n in reversed(path):
                n.back_propagate(value)
                value = -value   

        action_probs = np.zeros(self.game.action_size)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count
        s = action_probs.sum()
        if s > 0:
            action_probs /= s
        else:
            legal = np.where(
                self.game.get_valid_moves(root.state).numpy()==1)[0]
            action_probs[legal] = 1.0 / len(legal)
        return action_probs
