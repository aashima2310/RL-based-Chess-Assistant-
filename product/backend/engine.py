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

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path, device="cpu")
                self.model.eval()
                print(f"✅ Loaded custom RL model from {self.model_path}")
            except Exception as e:
                print(f"❌ Failed to load custom model weights: {e}")
        else:
            print(f"⚠️ Model file not found at '{self.model_path}'. Ensure 'value_clean_best.pt' is in 'backend/models/'.")

    def get_move(self, board):
        w_idx = self.extractor.get_halfkp_indices(board, chess.WHITE)
        b_idx = self.extractor.get_halfkp_indices(board, chess.BLACK)

        w_acc = self.model.backbone.refresh_accumulator(w_idx)
        b_acc = self.model.backbone.refresh_accumulator(b_idx)

        with torch.no_grad():
            policy_probs, value = self.model(w_acc, b_acc, board=[board])

        policy_probs = policy_probs[0]
        best_action_idx = torch.argmax(policy_probs).item()

        best_move, _ = self.extractor.resolve_move(
            action_idx=best_action_idx,
            board=board,
            policy=policy_probs.tolist()
        )

        return best_move

class ChessEngine:
    def __init__(self, engine_type="stockfish", difficulty="easy", model_path=DEFAULT_MODEL_PATH):
        if str(engine_type).lower() == "custom":
            self.engine = CustomEngine(model_path=model_path)
        else:
            self.engine = StockfishEngine(difficulty=difficulty)

    def get_move(self, board):
        return self.engine.get_move(board)

def get_engine(engine_type="stockfish", difficulty="easy"):
    if str(engine_type).lower() == "custom":
        return CustomEngine()
    return StockfishEngine(difficulty)
