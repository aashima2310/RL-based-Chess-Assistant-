
import chess
import torch
import numpy as np
import math
import sys
sys.path.insert(0, "/content/repo")
from RL.chess_env.features import HalfKPExtractor, resolve_move
from RL.chess_env.board import Chess_game
from RL.config import Config

extractor = HalfKPExtractor()
game_env = Chess_game()


def get_policy_value(board, model, device):
    w_idx = extractor.get_halfkp_indices(board, chess.WHITE)
    b_idx = extractor.get_halfkp_indices(board, chess.BLACK)

    w_acc = model.backbone.refresh_accumulator(w_idx).unsqueeze(0).to(device)
    b_acc = model.backbone.refresh_accumulator(b_idx).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        pol, val = model(w_acc, b_acc, [board])
    pol = pol.squeeze(0).cpu().numpy()
    mask = game_env.get_valid_moves(board).numpy()
    pol = pol * mask
    s = pol.sum()
    if s > 0:
        pol /= s
    else:
        legal = np.where(mask == 1)[0]
        pol[legal] = 1.0 / len(legal)
    return pol, val.item()


def mcts_search(root_board, model, device, num_searches=None, c=None,
                 add_noise=False, dirichlet_alpha=None, dirichlet_epsilon=None):

    num_searches = num_searches if num_searches is not None else Config.num_simulations
    c = c if c is not None else Config.c_puct
    dirichlet_alpha = dirichlet_alpha if dirichlet_alpha is not None else Config.dirichlet_alpha
    dirichlet_epsilon = dirichlet_epsilon if dirichlet_epsilon is not None else Config.dirichlet_epsilon

    class N:
        def __init__(self):
            self.n = 0
            self.w = 0.0
            self.p = None
            self.children = {}

    root = N()
    pol, val = get_policy_value(root_board, model, device)

    if add_noise:
        mask = game_env.get_valid_moves(root_board).numpy()
        legal = np.where(mask == 1)[0]
        noise = np.random.dirichlet([dirichlet_alpha] * len(legal))
        noisy_pol = np.zeros_like(pol)
        noisy_pol[legal] = noise
        pol = (1 - dirichlet_epsilon) * pol + dirichlet_epsilon * noisy_pol

    root.p = pol
    root.n = 1
    root.w = val

    for _ in range(num_searches):
        node = root
        board = chess.Board(root_board.fen())
        path = [node]

        while True:
            mask = game_env.get_valid_moves(board).numpy()
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break

            unvisited = [a for a in legal if a not in node.children]
            if unvisited:
                best_a = max(unvisited, key=lambda a: node.p[a] if node.p is not None else 0)
                move, true_idx = resolve_move(best_a, board, node.p)
                node.children[true_idx] = N()
                board.push(move)
                node = node.children[true_idx]
                path.append(node)
                break 
            best_a, best_u = None, -1e9
            for a in legal:
                p = node.p[a] if node.p is not None else 1.0 / len(legal)
                c_node = node.children.get(a)
                if c_node is None:
                    continue  
                q = -c_node.w / c_node.n if c_node.n > 0 else 0
                u = q + c * p * math.sqrt(node.n) / (1 + c_node.n)
                if u > best_u:
                    best_u, best_a = u, a

            if best_a is None:
                break

            move, true_idx = resolve_move(best_a, board, node.p)
            board.push(move)
            node = node.children[true_idx]
            path.append(node)

        if board.is_game_over(claim_draw=True):
            outcome = board.outcome(claim_draw=True)
            leaf_value = -1.0 if (outcome and outcome.winner is not None) else 0.0
        else:
            pol2, leaf_value = get_policy_value(board, model, device)
            node.p = pol2

        node.n += 1
        node.w += leaf_value
        value = -leaf_value
        for n in reversed(path[:-1]):
            n.n += 1
            n.w += value
            value = -value

    action_probs = np.zeros(4672)
    for a, child in root.children.items():
        action_probs[a] = child.n
    s = action_probs.sum()
    if s > 0:
        action_probs /= s
    else:
        mask = game_env.get_valid_moves(root_board).numpy()
        legal = np.where(mask == 1)[0]
        action_probs[legal] = 1.0 / len(legal)
    return action_probs
