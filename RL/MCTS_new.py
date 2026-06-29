import chess, torch, numpy as np, math
import sys
sys.path.insert(0, "/content/repo")
from RL.chess_env.features import HalfKPExtractor
from RL.chess_env.board import Chess_game

extractor = HalfKPExtractor()
game_env  = Chess_game()

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
    pol  = pol * mask
    s    = pol.sum()
    if s > 0:
        pol /= s
    else:
        legal = np.where(mask==1)[0]
        pol[legal] = 1.0/len(legal)
    return pol, val.item()


def mcts_search(root_board, model, device, num_searches=400, c=1.5):

    class N:
        def __init__(self):
            self.n = 0
            self.w = 0.0
            self.p = None
            self.children = {}

    root = N()
    pol, val = get_policy_value(root_board, model, device)
    root.p   = pol
    root.n   = 1
    root.w   = val

    for _ in range(num_searches):
        node  = root
        board = chess.Board(root_board.fen())
        path  = [node]
        while True:
            mask  = game_env.get_valid_moves(board).numpy()
            legal = np.where(mask == 1)[0]
            if len(legal) == 0:
                break
            unvisited = [a for a in legal if a not in node.children]
            if unvisited:
                best_a = max(unvisited,
                    key=lambda a: node.p[a] if node.p is not None else 0)
                node.children[best_a] = N()
                move = extractor.idx_to_move(best_a, board)
                if not board.is_legal(move):
                    legal_list = list(board.legal_moves)
                    if not legal_list:
                        break
                    move = max(legal_list,
                        key=lambda m: node.p[extractor.move_to_idx(m)]
                        if node.p is not None else 0)
                board.push(move)
                node = node.children[best_a]
                path.append(node)
                break  # evaluate this new node
            best_a, best_u = None, -1e9
            for a in legal:
                p      = node.p[a] if node.p is not None else 1.0 / len(legal)
                c_node = node.children[a]
                q      = -c_node.w / c_node.n if c_node.n > 0 else 0
                u      = q + c * p * math.sqrt(node.n) / (1 + c_node.n)
                if u > best_u:
                    best_u, best_a = u, a

            if best_a is None:
                break

            move = extractor.idx_to_move(best_a, board)
            if not board.is_legal(move):
                legal_list = list(board.legal_moves)
                if not legal_list:
                    break
                move = max(legal_list,
                    key=lambda m: node.p[extractor.move_to_idx(m)]
                    if node.p is not None else 0)

            board.push(move)
            node = node.children[best_a]
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
        mask  = game_env.get_valid_moves(root_board).numpy()
        legal = np.where(mask == 1)[0]
        action_probs[legal] = 1.0 / len(legal)
    return action_probs


print("✅ MCTS defined")
