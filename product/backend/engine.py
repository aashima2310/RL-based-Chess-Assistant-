
import os
import math
import chess
import torch

from mcts import MCTS
from features import HalfKPExtractor
from combined_network import NNUE_AlphaZero
from opening_book import OpeningBook


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "value_clean_best.pt"
)


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

        self.book = OpeningBook(
            os.path.join(
                BASE_DIR,
                "models",
                "opening_book.bin"
            )
        )

        if os.path.exists(self.model_path):
            try:
                self.model.load_weights(
                    self.model_path,
                    device="cpu"
                )

                self.model.eval()
                self.loaded = True

                print(
                    f"Loaded custom RL model from "
                    f"{self.model_path}"
                )

            except Exception as e:
                print(
                    f"Failed to load custom model weights: {e}"
                )

        else:
            print(
                f"Model file not found at "
                f"'{self.model_path}'. Ensure "
                f"'value_clean_best.pt' is in "
                f"'backend/models/'."
            )

        if not self.loaded:
            print(
                "RL engine unavailable: falling back to "
                "Stockfish (medium) instead of random moves."
            )

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

        piece = board.piece_at(move.to_square)

        if piece is not None:
            return piece

        if board.is_en_passant(move):
            return chess.Piece(
                chess.PAWN,
                not board.turn
            )

        return None

    def _static_exchange_gain(self, board, move):

        captured = self._captured_piece(
            board,
            move
        )

        if captured is None:
            return 0.0

        captured_value = self._piece_value(
            captured.piece_type
        )

        target = move.to_square

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        opponent_recapture = self._see_recapture(
            next_board,
            target
        )

        return captured_value - opponent_recapture

    def _see_recapture(self, board, target):

        target_piece = board.piece_at(target)

        if target_piece is None:
            return 0.0

        best_gain = 0.0

        captures = [
            move
            for move in board.legal_moves
            if move.to_square == target
        ]

        for capture in captures:

            captured_piece = board.piece_at(
                target
            )

            if captured_piece is None:
                continue

            captured_value = self._piece_value(
                captured_piece.piece_type
            )

            next_board = board.copy(
                stack=False
            )

            next_board.push(capture)

            opponent_gain = self._see_recapture(
                next_board,
                target
            )

            gain = (
                captured_value
                - opponent_gain
            )

            if gain > best_gain:
                best_gain = gain

        return best_gain

    def _is_piece_hanging(self, board, square):

        piece = board.piece_at(square)

        if piece is None:
            return False

        captures = [
            move
            for move in board.legal_moves
            if move.to_square == square
        ]

        if not captures:
            return False

        for capture in captures:

            gain = self._static_exchange_gain(
                board,
                capture
            )

            if gain > 0.5:
                return True

        return False

    def _capture_is_safe(self, board, move):

        captured = self._captured_piece(
            board,
            move
        )

        if captured is None:
            return True

        gain = self._static_exchange_gain(
            board,
            move
        )

        return gain >= 0.0

    def _move_material_score(self, board):

        score = 0.0

        for square in chess.SQUARES:

            piece = board.piece_at(square)

            if piece is None:
                continue

            value = self._piece_value(
                piece.piece_type
            )

            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value

        return score

    def _tactical_moves(self, board):

        moves = []

        for move in board.legal_moves:

            if (
                board.is_capture(move)
                or board.gives_check(move)
                or move.promotion is not None
            ):
                moves.append(move)

        return moves

    def _tactical_move_order_score(self, board, move):

        score = 0.0

        captured = self._captured_piece(
            board,
            move
        )

        if captured is not None:

            score += (
                10.0
                * self._piece_value(
                    captured.piece_type
                )
            )

            score += self._static_exchange_gain(
                board,
                move
            )

        if board.gives_check(move):
            score += 8.0

        if move.promotion is not None:
            score += 10.0

        return score

    def _shallow_tactical_search(
        self,
        board,
        depth,
        root_color
    ):

        if board.is_checkmate():

            if board.turn == root_color:
                return -1000.0

            return 1000.0

        if board.is_stalemate():
            return 0.0

        if depth <= 0:
            material = self._move_material_score(
                board
            )

            if root_color == chess.WHITE:
                return material

            return -material

        moves = self._tactical_moves(board)

        if not moves:

            material = self._move_material_score(
                board
            )

            if root_color == chess.WHITE:
                return material

            return -material

        moves.sort(
            key=lambda move:
            self._tactical_move_order_score(
                board,
                move
            ),
            reverse=True
        )

        moves = moves[:12]

        maximizing = (
            board.turn == root_color
        )

        if maximizing:

            best = -float("inf")

            for move in moves:

                next_board = board.copy(
                    stack=False
                )

                next_board.push(move)

                value = self._shallow_tactical_search(
                    next_board,
                    depth - 1,
                    root_color
                )

                if value > best:
                    best = value

            return best

        best = float("inf")

        for move in moves:

            next_board = board.copy(
                stack=False
            )

            next_board.push(move)

            value = self._shallow_tactical_search(
                next_board,
                depth - 1,
                root_color
            )

            if value < best:
                best = value

        return best

    def _three_ply_tactical_score(
        self,
        board,
        move
    ):

        root_color = board.turn

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        immediate_material = (
            self._static_exchange_gain(
                board,
                move
            )
        )

        future_score = self._shallow_tactical_search(
            next_board,
            2,
            root_color
        )

        score = future_score

        if immediate_material > 0:
            score += (
                1.5
                * immediate_material
            )

        elif immediate_material < 0:
            score += (
                2.0
                * immediate_material
            )

        if next_board.is_check():
            score += 2.0

        return score

    def _tactical_bonus(self, board, move):

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        if next_board.is_checkmate():
            return 1000.0

        captured = self._captured_piece(
            board,
            move
        )

        gives_check = next_board.is_check()

        if captured is None:

            tactical_lookahead = (
                self._three_ply_tactical_score(
                    board,
                    move
                )
            )

            bonus = tactical_lookahead * 0.8

            if gives_check:
                bonus += 3.0

            return bonus

        victim_value = self._piece_value(
            captured.piece_type
        )

        see_gain = self._static_exchange_gain(
            board,
            move
        )

        bonus = 2.0 + victim_value

        if see_gain > 0:

            bonus += 3.0 * see_gain

            if see_gain >= victim_value - 0.1:

                if captured.piece_type == chess.QUEEN:
                    bonus += 45.0

                elif captured.piece_type == chess.ROOK:
                    bonus += 32.0

                elif captured.piece_type in (
                    chess.BISHOP,
                    chess.KNIGHT
                ):
                    bonus += 22.0

                elif captured.piece_type == chess.PAWN:
                    bonus += 6.0

                else:
                    bonus += 15.0

            if see_gain >= 5.0:
                bonus += 15.0

            if see_gain >= 8.0:
                bonus += 25.0

        elif see_gain < 0:

            bonus += 4.0 * see_gain

            if see_gain <= -2.0:
                bonus -= 8.0

        lookahead_score = (
            self._three_ply_tactical_score(
                board,
                move
            )
        )

        bonus += (
            2.0
            * lookahead_score
        )

        if gives_check:
            bonus += 3.0

        return bonus

    def get_move(self, board):

        book_move = self.book.pick(
            board
        )

        if book_move is not None:
            return book_move

        if not self.loaded:

            if self._fallback is None:
                self._fallback = StockfishEngine(
                    "medium"
                )

            return self._fallback.get_move(
                board
            )

        black = (
            board.turn == chess.BLACK
        )

        canon = (
            board.mirror()
            if black
            else board
        )

        w_idx = self.extractor.get_halfkp_indices(
            canon,
            chess.WHITE
        )

        b_idx = self.extractor.get_halfkp_indices(
            canon,
            chess.BLACK
        )

        w_acc = (
            self.model.backbone.refresh_accumulator(
                w_idx
            )
        )

        b_acc = (
            self.model.backbone.refresh_accumulator(
                b_idx
            )
        )

        with torch.no_grad():

            policy_probs, _ = self.model(
                w_acc,
                b_acc,
                board=[canon]
            )

        policy_probs = policy_probs[0]

        legal_moves = list(
            canon.legal_moves
        )

        scored_moves = []

        for candidate in legal_moves:

            action_idx = (
                self.extractor.move_to_idx(
                    candidate
                )
            )

            policy_prob = float(
                policy_probs[
                    action_idx
                ].item()
            )

            policy_score = math.log(
                max(
                    policy_prob,
                    1e-9
                )
            )

            tactical_score = (
                self._tactical_bonus(
                    canon,
                    candidate
                )
            )

            total_score = (
                policy_score
                + tactical_score
            )

            scored_moves.append(
                (
                    total_score,
                    policy_prob,
                    tactical_score,
                    candidate
                )
            )

        (
            _,
            _,
            _,
            move
        ) = max(
            scored_moves,
            key=lambda item: item[0]
        )

        if black:

            move = chess.Move(
                chess.square_mirror(
                    move.from_square
                ),
                chess.square_mirror(
                    move.to_square
                ),
                promotion=move.promotion
            )

        if not board.is_legal(move):

            q = chess.Move(
                move.from_square,
                move.to_square,
                promotion=chess.QUEEN
            )

            if board.is_legal(q):
                move = q

            else:
                move = next(
                    iter(
                        board.legal_moves
                    )
                )

        return move


_CUSTOM_CACHE = {}


class ChessEngine:

    def __init__(
        self,
        engine_type="stockfish",
        difficulty="easy",
        model_path=DEFAULT_MODEL_PATH
    ):

        if str(engine_type).lower() == "custom":

            if model_path not in _CUSTOM_CACHE:

                _CUSTOM_CACHE[
                    model_path
                ] = CustomEngine(
                    model_path=model_path
                )

            self.engine = _CUSTOM_CACHE[
                model_path
            ]

        else:

            self.engine = StockfishEngine(
                difficulty=difficulty
            )

    def get_move(self, board):
        return self.engine.get_move(board)


def get_engine(
    engine_type="stockfish",
    difficulty="easy"
):

    if str(engine_type).lower() == "custom":
        return CustomEngine()

    return StockfishEngine(
        difficulty
    )
