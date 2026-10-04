import os
import time
import chess
import torch

from mcts import MCTS
from features import HalfKPExtractor
from combined_network import NNUE_AlphaZero
from opening_book import OpeningBook

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "value_clean_best.pt")

MAX_DEPTH = 5
NODE_LIMIT = 12000
TIME_LIMIT = 1.0
QUIESCENCE_DEPTH = 3
TT_MAX_SIZE = 75000
VALUE_CACHE_MAX = 3000
MATE_SCORE = 100000


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
        self._value_cache = {}
        self._tt = {}
        self._nodes = 0
        self._start_time = 0.0
        self._stop = False
        self._killers = {}
        self._history = {}

        self.book = OpeningBook(
            os.path.join(BASE_DIR, "models", "opening_book.bin")
        )

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(self.model_path, device="cpu")
                self.model.eval()
                self.loaded = True
                print(f"Loaded custom RL model from {self.model_path}")
            except Exception as e:
                print(f"Failed to load custom model weights: {e}")
        else:
            print(f"Model file not found at '{self.model_path}'.")

    @staticmethod
    def _piece_value(piece_type):
        return {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.2,
            chess.BISHOP: 3.3,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 100.0,
        }.get(piece_type, 0.0)

    @staticmethod
    def _captured_piece(board, move):
        piece = board.piece_at(move.to_square)
        if piece is not None:
            return piece
        if board.is_en_passant(move):
            return chess.Piece(chess.PAWN, not board.turn)
        return None

    def _capture_score(self, board, move):
        captured = self._captured_piece(board, move)
        if captured is None:
            return 0.0

        attacker = board.piece_at(move.from_square)
        victim = self._piece_value(captured.piece_type)
        attacker_value = self._piece_value(attacker.piece_type) if attacker else 0.0
        score = victim * 100.0 - attacker_value * 5.0

        if victim >= attacker_value:
            score += 80.0

        return score

    def _quick_see(self, board, move):
        captured = self._captured_piece(board, move)
        if captured is None:
            return 0.0

        gain = self._piece_value(captured.piece_type)
        child = board.copy(stack=False)
        child.push(move)
        recapture = 0.0

        for reply in child.legal_moves:
            if reply.to_square != move.to_square:
                continue
            attacker = child.piece_at(reply.from_square)
            if attacker is None:
                continue
            value = self._piece_value(attacker.piece_type)
            if recapture == 0.0 or value < recapture:
                recapture = value

        return gain - recapture

    def _network_value(self, board):
        key = board.fen()
        if key in self._value_cache:
            return self._value_cache[key]

        black = board.turn == chess.BLACK
        canon = board.mirror() if black else board

        try:
            w_idx = self.extractor.get_halfkp_indices(canon, chess.WHITE)
            b_idx = self.extractor.get_halfkp_indices(canon, chess.BLACK)
            w_acc = self.model.backbone.refresh_accumulator(w_idx)
            b_acc = self.model.backbone.refresh_accumulator(b_idx)

            with torch.no_grad():
                _, value = self.model(w_acc, b_acc, board=[canon])

            result = float(value.reshape(-1)[0].item())
            result = max(-1.0, min(1.0, result))
            if black:
                result = -result
        except Exception:
            result = 0.0

        self._value_cache[key] = result
        if len(self._value_cache) > VALUE_CACHE_MAX:
            self._value_cache.pop(next(iter(self._value_cache)))
        return result

    def _material_score(self, board):
        score = 0.0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            value = self._piece_value(piece.piece_type)
            score += value if piece.color == board.turn else -value
        return score

    def _positional_score(self, board):
        score = 0.0
        centers = (chess.D4, chess.E4, chess.D5, chess.E5)

        for square in centers:
            piece = board.piece_at(square)
            if piece is not None:
                score += 0.18 if piece.color == board.turn else -0.18
            score += 0.04 * (
                len(board.attackers(board.turn, square))
                - len(board.attackers(not board.turn, square))
            )

        own_moves = board.legal_moves.count()
        opponent = board.copy(stack=False)
        opponent.turn = not board.turn
        opponent_moves = opponent.legal_moves.count()
        score += (own_moves - opponent_moves) / 30.0

        if board.is_check():
            score -= 0.35

        return max(-1.0, min(1.0, score))

    def _evaluate(self, board):
        if board.is_checkmate():
            return -MATE_SCORE
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        return (
            self._material_score(board)
            + 1.2 * self._network_value(board)
            + 0.35 * self._positional_score(board)
        )

    def _should_stop(self):
        if self._stop:
            return True
        if self._nodes >= NODE_LIMIT:
            self._stop = True
            return True
        if time.perf_counter() - self._start_time >= TIME_LIMIT:
            self._stop = True
            return True
        return False

    def _ordered_moves(self, board, moves, tt_move=None, ply=0):
        scored = []
        for move in moves:
            score = 0.0

            if tt_move is not None and move == tt_move:
                score += 1000000.0

            if self._killers.get(ply) == move:
                score += 750000.0

            score += self._history.get(move, 0) * 0.01

            if board.is_capture(move):
                score += 500000.0 + self._capture_score(board, move)

            if move.promotion is not None:
                score += 400000.0

            if board.gives_check(move):
                score += 300000.0

            if not board.is_capture(move):
                piece = board.piece_at(move.from_square)
                if piece is not None:
                    score += 10.0 - self._piece_value(piece.piece_type)

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored]

    def _terminal_score(self, board, ply):
        if board.is_checkmate():
            return -MATE_SCORE + ply
        if (
            board.is_stalemate()
            or board.is_insufficient_material()
            or board.halfmove_clock >= 100
        ):
            return 0.0
        return None

    def _quiescence(self, board, alpha, beta, ply, depth):
        if self._should_stop():
            return self._evaluate(board)

        self._nodes += 1
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal

        stand_pat = self._evaluate(board)
        if depth <= 0:
            return stand_pat

        if not board.is_check() and stand_pat >= beta:
            return stand_pat

        if board.is_check():
            moves = list(board.legal_moves)
        else:
            moves = [
                move for move in board.legal_moves
                if board.is_capture(move)
                or move.promotion is not None
                or board.gives_check(move)
            ]

        if not moves:
            return stand_pat

        moves = self._ordered_moves(board, moves)

        for move in moves[:6]:
            if self._should_stop():
                break

            if not board.is_check() and board.is_capture(move):
                if self._quick_see(board, move) < -1.0:
                    continue

            child = board.copy(stack=False)
            child.push(move)
            score = -self._quiescence(child, -beta, -alpha, ply + 1, depth - 1)

            if score >= beta:
                return score
            if score > alpha:
                alpha = score

        return alpha

    def _search(self, board, depth, alpha, beta, ply):
        if self._should_stop():
            return self._evaluate(board)

        self._nodes += 1
        terminal = self._terminal_score(board, ply)
        if terminal is not None:
            return terminal

        if depth <= 0:
            return self._quiescence(board, alpha, beta, ply, QUIESCENCE_DEPTH)

        key = board.fen()
        entry = self._tt.get(key)
        original_alpha = alpha

        if entry is not None and entry[0] >= depth:
            stored_depth, value, flag, stored_move = entry
            if flag == 0:
                return value
            if flag == 1:
                alpha = max(alpha, value)
            elif flag == 2:
                beta = min(beta, value)
            if alpha >= beta:
                return value
        else:
            stored_move = None

        moves = self._ordered_moves(board, list(board.legal_moves), stored_move, ply)
        if not moves:
            return self._evaluate(board)

        best_value = -float("inf")
        best_move = moves[0]

        for move in moves:
            if self._should_stop():
                break

            child = board.copy(stack=False)
            child.push(move)
            score = -self._search(child, depth - 1, -beta, -alpha, ply + 1)

            if score > best_value:
                best_value = score
                best_move = move

            alpha = max(alpha, score)
            if alpha >= beta:
                if not board.is_capture(move):
                    self._killers[ply] = move
                    self._history[move] = min(100000, self._history.get(move, 0) + depth * depth)
                break

        if best_value == -float("inf"):
            best_value = self._evaluate(board)

        if best_value <= original_alpha:
            flag = 2
        elif best_value >= beta:
            flag = 1
        else:
            flag = 0

        self._tt[key] = (depth, best_value, flag, best_move)
        if len(self._tt) > TT_MAX_SIZE:
            self._tt.pop(next(iter(self._tt)))

        return best_value

    def _root_policy(self, board):
        black = board.turn == chess.BLACK
        canon = board.mirror() if black else board

        w_idx = self.extractor.get_halfkp_indices(canon, chess.WHITE)
        b_idx = self.extractor.get_halfkp_indices(canon, chess.BLACK)
        w_acc = self.model.backbone.refresh_accumulator(w_idx)
        b_acc = self.model.backbone.refresh_accumulator(b_idx)

        with torch.no_grad():
            policy_probs, _ = self.model(w_acc, b_acc, board=[canon])

        return canon, policy_probs[0]

    def _root_order(self, board, policy_probs):
        scored = []

        for move in board.legal_moves:
            idx = self.extractor.move_to_idx(move)
            score = float(policy_probs[idx].item())

            if board.is_capture(move):
                score += 4.0 + self._capture_score(board, move) / 700.0
            if board.gives_check(move):
                score += 1.0
            if move.promotion is not None:
                score += 5.0

            scored.append((score, move))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in scored]

    def _root_search(self, board, depth, moves):
        alpha = -float("inf")
        beta = float("inf")
        best_value = -float("inf")
        best_move = moves[0]

        for move in moves:
            if self._should_stop():
                break

            child = board.copy(stack=False)
            child.push(move)
            score = -self._search(child, depth - 1, -beta, -alpha, 1)

            if score > best_value:
                best_value = score
                best_move = move

            alpha = max(alpha, score)

        return best_move, best_value

    def get_move(self, board):
        book_move = self.book.pick(board)
        if book_move is not None:
            return book_move

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        if not self.loaded:
            if self._fallback is None:
                self._fallback = StockfishEngine("medium")
            return self._fallback.get_move(board)

        self._nodes = 0
        self._stop = False
        self._start_time = time.perf_counter()

        try:
            canon, policy_probs = self._root_policy(board)
        except Exception:
            return legal_moves[0]

        ordered_moves = self._root_order(canon, policy_probs)
        if not ordered_moves:
            return legal_moves[0]

        best_move = ordered_moves[0]

        for depth in range(1, MAX_DEPTH + 1):
            if self._should_stop():
                break

            candidate, _ = self._root_search(canon, depth, ordered_moves)

            if candidate is not None and not self._stop:
                best_move = candidate
                ordered_moves = [best_move] + [m for m in ordered_moves if m != best_move]

            if self._should_stop():
                break

        if board.turn == chess.BLACK:
            best_move = chess.Move(
                chess.square_mirror(best_move.from_square),
                chess.square_mirror(best_move.to_square),
                promotion=best_move.promotion
            )

        if board.is_legal(best_move):
            return best_move

        queen_move = chess.Move(
            best_move.from_square,
            best_move.to_square,
            promotion=chess.QUEEN
        )

        if board.is_legal(queen_move):
            return queen_move

        return legal_moves[0]


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


def get_engine(engine_type="stockfish", difficulty="easy", model_path=DEFAULT_MODEL_PATH):
    if str(engine_type).lower() == "custom":
        if model_path not in _CUSTOM_CACHE:
            _CUSTOM_CACHE[model_path] = CustomEngine(model_path=model_path)
        return _CUSTOM_CACHE[model_path]
    return StockfishEngine(difficulty)
