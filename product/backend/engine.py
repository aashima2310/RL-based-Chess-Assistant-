import os
import math

import chess
import torch

from mcts import MCTS
from features import HalfKPExtractor
from combined_network import NNUE_AlphaZero
from opening_book import OpeningBook
from hybrid_search import TacticalSearch, MATE, PIECE_CP


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "value_clean_best.pt")

# ---- hybrid tuning (override with environment variables if you like) ----
TOP_K          = int(os.getenv("RL_TOP_K", "6"))            # best moves by network policy that are always searched
MAX_CANDIDATES = int(os.getenv("RL_MAX_CANDIDATES", "14"))  # cap on moves searched (keeps it fast)
MAX_DEPTH      = int(os.getenv("RL_MAX_DEPTH", "4"))        # deepest search, in plies
TIME_LIMIT     = float(os.getenv("RL_TIME_LIMIT", "0.5"))   # seconds of search per move
POLICY_W       = float(os.getenv("RL_POLICY_W", "0.5"))     # weight of network policy (log-prob, in pawns)
VALUE_W        = float(os.getenv("RL_VALUE_W", "2.0"))      # weight of network value head (pawns per 1.0)
DEBUG          = os.getenv("RL_ENGINE_DEBUG") == "1"


class StockfishEngine:

    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.mcts = MCTS(difficulty)

    def get_move(self, board):
        move = self.mcts.search(board)
        if isinstance(move, str):
            move = chess.Move.from_uci(move)
        return move


class CustomEngine:
    """Hybrid engine: network policy + value head propose and judge moves,
    a short tactical search (alpha-beta + quiescence) vetoes blunders and finds tactics."""

    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.extractor = HalfKPExtractor()
        self.model = NNUE_AlphaZero()
        self.loaded = False
        self._fallback = None
        self.searcher = TacticalSearch()
        self.book = OpeningBook(os.path.join(BASE_DIR, "models", "opening_book.bin"))

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path, device="cpu")
                self.model.eval()
                self.loaded = True
                print(f"Loaded custom RL model from {self.model_path}")
            except Exception as e:
                print(f"Failed to load custom model weights: {e}")
        else:
            print(f"Model file not found at '{self.model_path}'. Ensure 'value_clean_best.pt' is in 'backend/models/'.")

        if not self.loaded:
            print("RL engine unavailable: falling back to Stockfish (medium) instead of random moves.")

    # ------------------------------------------------------------------ network helpers
    @staticmethod
    def _canonical(board):
        """Network sees positions with the side to move as White."""
        return board.mirror() if board.turn == chess.BLACK else board.copy(stack=False)

    @staticmethod
    def _to_canon_move(move, black):
        if not black:
            return move
        return chess.Move(chess.square_mirror(move.from_square),
                          chess.square_mirror(move.to_square),
                          promotion=move.promotion)

    def _accumulator(self, indices):
        backbone = self.model.backbone
        with torch.no_grad():
            if not indices:
                return backbone.input_bias.clone()
            idx = torch.tensor(indices, dtype=torch.long)
            return backbone.input_weights[idx].sum(0) + backbone.input_bias

    def _policy(self, board, legal):
        """Network policy probability for every legal move: {move: prob}."""
        black = board.turn == chess.BLACK
        canon = self._canonical(board)
        w_acc = self._accumulator(self.extractor.get_halfkp_indices(canon, chess.WHITE))
        b_acc = self._accumulator(self.extractor.get_halfkp_indices(canon, chess.BLACK))
        with torch.no_grad():
            probs, _ = self.model(w_acc, b_acc, board=[canon])
        probs = probs[0].tolist()
        return {m: probs[self.extractor.move_to_idx(self._to_canon_move(m, black))] for m in legal}

    def _child_values(self, board, moves):
        """Value head's opinion of the position after each move, from the MOVER's point of view
        (in [-1, 1]). One batched forward pass. Terminal positions are skipped."""
        board = board.copy()                             # never modify the live game board
        w_list, b_list, order = [], [], []
        for m in moves:
            board.push(m)
            if not board.is_game_over():
                canon = self._canonical(board)          # opponent to move -> mirrored to 'White to move'
                w_list.append(self._accumulator(self.extractor.get_halfkp_indices(canon, chess.WHITE)))
                b_list.append(self._accumulator(self.extractor.get_halfkp_indices(canon, chess.BLACK)))
                order.append(m)
            board.pop()
        if not order:
            return {}
        with torch.no_grad():
            _, values = self.model(torch.stack(w_list), torch.stack(b_list), board=None)
        values = values.view(-1).tolist()
        return {m: -v for m, v in zip(order, values)}   # network scores the opponent's side -> negate

    # ------------------------------------------------------------------ move selection
    @staticmethod
    def _victim_order(board, move):
        victim = PIECE_CP.get(board.piece_type_at(move.to_square), 100 if board.is_en_passant(move) else 0)
        return victim * 10 - PIECE_CP.get(board.piece_type_at(move.from_square), 0) // 10

    def _select_candidates(self, board, legal, probs):
        if len(legal) <= MAX_CANDIDATES:
            return list(legal)
        ranked = sorted(legal, key=lambda m: probs[m], reverse=True)
        chosen = ranked[:TOP_K]
        tactical = [m for m in legal
                    if m not in chosen and (board.is_capture(m) or m.promotion or board.gives_check(m))]
        tactical.sort(key=lambda m: self._victim_order(board, m), reverse=True)
        for m in tactical + ranked:
            if len(chosen) >= MAX_CANDIDATES:
                break
            if m not in chosen:
                chosen.append(m)
        return chosen

    @staticmethod
    def _search_pawns(cp):
        """Search score (centipawns) -> pawns, with mates mapped to large, distance-aware values."""
        if cp >= MATE - 200:
            return 100.0 - (MATE - cp)          # faster mate is better
        if cp <= -MATE + 200:
            return -100.0 + (MATE + cp)         # slower mate is better
        return max(-30.0, min(30.0, cp / 100.0))

    def get_move(self, board):
        book_move = self.book.pick(board)
        if book_move is not None:
            return book_move

        if not self.loaded:
            if self._fallback is None:
                self._fallback = StockfishEngine("medium")
            return self._fallback.get_move(board)

        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]

        # 1) network policy
        try:
            probs = self._policy(board, legal)
        except Exception as e:
            print(f"Network policy failed ({e}); using search only.")
            probs = {m: 1.0 / len(legal) for m in legal}

        # 2) tactical search over the candidates
        cands = self._select_candidates(board, legal, probs)
        scores, depth = self.searcher.score_moves(board, cands, MAX_DEPTH, TIME_LIMIT)

        # safety net: if every candidate looks bad, the real defence may be outside the shortlist
        if scores and max(scores.values()) < -250 and len(cands) < len(legal):
            rest = [m for m in legal if m not in scores]
            extra, _ = self.searcher.score_moves(board, rest, max(2, depth), TIME_LIMIT)
            scores.update(extra)

        if not scores:                                   # should not happen; stay safe
            return max(legal, key=lambda m: probs[m])

        best_cp = max(scores.values())
        if best_cp >= MATE - 50:                         # forced mate found
            return max(scores, key=scores.get)

        # 3) value head, only for moves that are still in the running
        contenders = [m for m, s in scores.items() if s >= best_cp - 150]
        try:
            values = self._child_values(board, contenders) if len(contenders) > 1 else {}
        except Exception as e:
            print(f"Value head failed ({e}); ignoring it.")
            values = {}

        # 4) blend: tactics + network preference + network position judgement
        def blended(m):
            return (self._search_pawns(scores[m])
                    + POLICY_W * math.log(max(probs[m], 1e-3))
                    + VALUE_W * values.get(m, 0.0))

        move = max(scores, key=blended)
        if DEBUG:
            top = sorted(scores, key=blended, reverse=True)[:4]
            print(f"[hybrid] depth {depth}, nodes {self.searcher.nodes}, picks:",
                  [(board.san(m), round(scores[m]), round(probs[m], 3), round(values.get(m, 0), 2)) for m in top])
        return move


_CUSTOM_CACHE = {}


class ChessEngine:

    def __init__(self, engine_type="stockfish", difficulty="easy", model_path=DEFAULT_MODEL_PATH):
        if str(engine_type).lower() == "custom":
            if model_path not in _CUSTOM_CACHE:
                _CUSTOM_CACHE[model_path] = CustomEngine(model_path=model_path)
            self.engine = _CUSTOM_CACHE[model_path]
        else:
            self.engine = StockfishEngine(difficulty=difficulty)

    def get_move(self, board):
        return self.engine.get_move(board)


def get_engine(engine_type="stockfish", difficulty="easy"):
    if str(engine_type).lower() == "custom":
        return CustomEngine()
    return StockfishEngine(difficulty)
