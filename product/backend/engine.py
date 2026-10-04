import os
import chess
import torch
from mcts import MCTS
from features import HalfKPExtractor
from combined_network import NNUE_AlphaZero

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

    def get_move(self, board):
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
            policy_probs, value = self.model(w_acc, b_acc, board=[canon])

        policy_probs = policy_probs[0]
        best_action_idx = torch.argmax(policy_probs).item()

        move, _ = self.extractor.resolve_move(
            action_idx=best_action_idx,
            board=canon,
            policy=policy_probs.tolist()
        )
        if black:
            move = chess.Move(
                chess.square_mirror(move.from_square),
                chess.square_mirror(move.to_square),
                promotion=move.promotion,
            )
        if not board.is_legal(move):
            q = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
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
