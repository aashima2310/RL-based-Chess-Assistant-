import os
import math
import chess
import torch
from mcts import MCTS
from features import HalfKPExtractor
from combined_network import NNUE_AlphaZero
from opening_book import OpeningBook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "value_clean_best.pt")

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
    def __init__(self, model_path=DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.extractor = HalfKPExtractor()
        self.model = NNUE_AlphaZero()
        self.loaded = False
        self._fallback = None
        self.book = OpeningBook(os.path.join(BASE_DIR, "models", "opening_book.bin"))

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path, device="cpu")
                self.model.eval()
                self.loaded = True
                print(f"✅ Loaded custom RL model from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load custom model weights: {e}")
        else:
            print(f"⚠️ Model file not found at '{self.model_path}'. Ensure 'value_clean_best.pt' is in 'backend/models/'.")

        if not self.loaded:
            print("⚠️ RL engine unavailable: falling back to Stockfish (medium) instead of random moves.")

    @staticmethod
    def _piece_value(piece_type):
        return {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.2,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 100.0,
        }.get(piece_type, 0.0)

    @staticmethod
    def _captured_piece(board, move):
        """Return the piece captured by a move, including en-passant."""
        piece = board.piece_at(move.to_square)
        if piece is not None:
            return piece

        if board.is_en_passant(move):
            return chess.Piece(chess.PAWN, not board.turn)

        return None

    def _is_hanging(self, board, square):
        """A piece is hanging if attacked and not defended."""
        piece = board.piece_at(square)
        if piece is None:
            return False

        attackers = board.attackers(board.turn, square)
        defenders = board.attackers(piece.color, square)
        defenders.discard(square)

        return bool(attackers) and not defenders

    def _tactical_bonus(self, board, move):
        """Bias the trained policy toward valuable free material."""
        captured = self._captured_piece(board, move)
        if captured is None:
            return 0.0

        victim_value = self._piece_value(captured.piece_type)

        # Small bonus for every capture.
        bonus = 0.20 * victim_value

        # Strong bonus when the captured piece is genuinely hanging.
        # This is deliberately large enough to override a bad policy preference
        # for a quiet move when a queen/rook/etc. is simply free.
        if not board.is_en_passant(move) and self._is_hanging(board, move.to_square):
            bonus += 3.5 + 0.75 * victim_value

        # Do not blindly take material if our capturing piece is immediately
        # attacked after the capture.
        next_board = board.copy(stack=False)
        next_board.push(move)
        moved_piece = next_board.piece_at(move.to_square)

        if moved_piece is not None:
            opponent_attackers = next_board.attackers(
                next_board.turn, move.to_square
            )
            if opponent_attackers:
                bonus -= 0.35 * self._piece_value(moved_piece.piece_type)

        # Always strongly prefer a move that actually checkmates.
        if next_board.is_checkmate():
            bonus += 100.0

        return bonus

    def get_move(self, board):
        # Standard openings first: avoids early mistakes. Falls through once out of book.
        book_move = self.book.pick(board)
        if book_move is not None:
            return book_move

        if not self.loaded:
            if self._fallback is None:
                self._fallback = StockfishEngine("medium")
            return self._fallback.get_move(board)

        # The network was trained on "canonical" positions (side to move = White).
        # For Black, mirror the board, pick the move, then mirror the move back.
        black = board.turn == chess.BLACK
        canon = board.mirror() if black else board

        w_idx = self.extractor.get_halfkp_indices(canon, chess.WHITE)
        b_idx = self.extractor.get_halfkp_indices(canon, chess.BLACK)
        w_acc = self.model.backbone.refresh_accumulator(w_idx)
        b_acc = self.model.backbone.refresh_accumulator(b_idx)

        with torch.no_grad():
            policy_probs, _ = self.model(w_acc, b_acc, board=[canon])

        policy_probs = policy_probs[0]

        # Previously the engine used argmax(policy), which could ignore a
        # hanging piece. Now every legal move keeps its learned policy score
        # and receives a tactical bonus for taking valuable free material.
        legal_moves = list(canon.legal_moves)
        scored_moves = []

        for candidate in legal_moves:
            action_idx = self.extractor.move_to_idx(candidate)
            policy_prob = float(policy_probs[action_idx].item())

            score = math.log(max(policy_prob, 1e-9))
            score += self._tactical_bonus(canon, candidate)

            scored_moves.append((score, policy_prob, candidate))

        _, _, move = max(scored_moves, key=lambda item: item[0])

        if black:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )

        if not board.is_legal(move):
            q = chess.Move(
                move.from_square,
                move.to_square,
                promotion=chess.QUEEN
            )
            move = q if board.is_legal(q) else next(iter(board.legal_moves))

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
