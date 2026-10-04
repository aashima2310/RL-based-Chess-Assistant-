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
        self._value_cache = {}

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

        captures.sort(
            key=lambda move:
            self._piece_value(
                board.piece_at(
                    move.from_square
                ).piece_type
            )
        )

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

        for capture in captures:

            if self._static_exchange_gain(
                board,
                capture
            ) > 0.5:
                return True

        return False

    def _network_value(self, board):

        key = board.fen()

        if key in self._value_cache:
            return self._value_cache[key]

        black = board.turn == chess.BLACK

        canon = (
            board.mirror()
            if black
            else board
        )

        try:

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

                _, value = self.model(
                    w_acc,
                    b_acc,
                    board=[canon]
                )

            value = float(
                value.reshape(-1)[0].item()
            )

            value = max(
                -1.0,
                min(1.0, value)
            )

            if black:
                value = -value

        except Exception:
            value = 0.0

        self._value_cache[key] = value

        if len(self._value_cache) > 5000:
            self._value_cache.pop(
                next(iter(self._value_cache))
            )

        return value

    def _material_score(self, board):

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

        if board.turn == chess.BLACK:
            score = -score

        return score

    def _mobility_score(self, board):

        own = board.turn

        own_moves = board.legal_moves.count()

        opponent = board.copy(
            stack=False
        )

        opponent.turn = not own

        if opponent.is_check():
            opponent_moves = 0
        else:
            opponent_moves = opponent.legal_moves.count()

        score = (
            own_moves - opponent_moves
        ) / 20.0

        return max(
            -1.0,
            min(1.0, score)
        )

    def _center_score(self, board):

        centers = [
            chess.D4,
            chess.E4,
            chess.D5,
            chess.E5
        ]

        score = 0.0

        for square in centers:

            piece = board.piece_at(square)

            if piece is not None:

                if piece.color == board.turn:
                    score += 0.20
                else:
                    score -= 0.20

            own_attackers = len(
                board.attackers(
                    board.turn,
                    square
                )
            )

            enemy_attackers = len(
                board.attackers(
                    not board.turn,
                    square
                )
            )

            score += (
                0.05
                * (
                    own_attackers
                    - enemy_attackers
                )
            )

        return max(
            -1.0,
            min(1.0, score)
        )

    def _king_safety_score(self, board):

        score = 0.0

        own_king = board.king(
            board.turn
        )

        enemy_king = board.king(
            not board.turn
        )

        if own_king is None or enemy_king is None:
            return 0.0

        own_zone = chess.SquareSet(
            chess.BB_KING_ATTACKS[own_king]
        )

        enemy_zone = chess.SquareSet(
            chess.BB_KING_ATTACKS[enemy_king]
        )

        own_attackers = 0
        enemy_attackers = 0

        for square in own_zone:
            enemy_attackers += len(
                board.attackers(
                    not board.turn,
                    square
                )
            )

        for square in enemy_zone:
            own_attackers += len(
                board.attackers(
                    board.turn,
                    square
                )
            )

        score += (
            own_attackers
            - enemy_attackers
        ) * 0.12

        if board.is_check():
            score -= 0.35

        return max(
            -1.0,
            min(1.0, score)
        )

    def _positional_score(self, board):

        mobility = self._mobility_score(
            board
        )

        center = self._center_score(
            board
        )

        king = self._king_safety_score(
            board
        )

        score = (
            0.40 * mobility
            + 0.30 * center
            + 0.30 * king
        )

        return max(
            -1.0,
            min(1.0, score)
        )

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

    def _tactical_order(self, board, move):

        score = 0.0

        captured = self._captured_piece(
            board,
            move
        )

        if captured is not None:

            victim = self._piece_value(
                captured.piece_type
            )

            attacker = board.piece_at(
                move.from_square
            )

            attacker_value = (
                self._piece_value(
                    attacker.piece_type
                )
                if attacker is not None
                else 0.0
            )

            score += (
                12.0
                * victim
            )

            score += (
                3.0
                * self._static_exchange_gain(
                    board,
                    move
                )
            )

            score -= (
                0.15
                * attacker_value
            )

        if board.gives_check(move):
            score += 10.0

        if move.promotion is not None:
            score += 20.0

        return score

    def _quiescence(
        self,
        board,
        alpha,
        beta,
        root_color,
        depth=2
    ):

        if board.is_checkmate():

            if board.turn == root_color:
                return -1000.0

            return 1000.0

        stand_pat = self._material_score(
            board
        )

        if depth <= 0:
            return stand_pat

        if board.turn == root_color:

            if stand_pat >= beta:
                return beta

            alpha = max(
                alpha,
                stand_pat
            )

            moves = self._tactical_moves(
                board
            )

            moves.sort(
                key=lambda move:
                self._tactical_order(
                    board,
                    move
                ),
                reverse=True
            )

            for move in moves[:8]:

                next_board = board.copy(
                    stack=False
                )

                next_board.push(move)

                score = self._quiescence(
                    next_board,
                    alpha,
                    beta,
                    root_color,
                    depth - 1
                )

                if score >= beta:
                    return beta

                alpha = max(
                    alpha,
                    score
                )

            return alpha

        if stand_pat <= alpha:
            return alpha

        beta = min(
            beta,
            stand_pat
        )

        moves = self._tactical_moves(
            board
        )

        moves.sort(
            key=lambda move:
            self._tactical_order(
                board,
                move
            ),
            reverse=True
        )

        for move in moves[:8]:

            next_board = board.copy(
                stack=False
            )

            next_board.push(move)

            score = self._quiescence(
                next_board,
                alpha,
                beta,
                root_color,
                depth - 1
            )

            if score <= alpha:
                return alpha

            beta = min(
                beta,
                score
            )

        return beta

    def _tactical_search(
        self,
        board,
        depth,
        root_color,
        alpha=-float("inf"),
        beta=float("inf")
    ):

        if board.is_checkmate():

            if board.turn == root_color:
                return -1000.0

            return 1000.0

        if board.is_stalemate():
            return 0.0

        if depth <= 0:

            return self._quiescence(
                board,
                alpha,
                beta,
                root_color,
                depth=2
            )

        moves = self._tactical_moves(
            board
        )

        if not moves:

            return self._quiescence(
                board,
                alpha,
                beta,
                root_color,
                depth=2
            )

        moves.sort(
            key=lambda move:
            self._tactical_order(
                board,
                move
            ),
            reverse=True
        )

        moves = moves[:10]

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

                score = self._tactical_search(
                    next_board,
                    depth - 1,
                    root_color,
                    alpha,
                    beta
                )

                best = max(
                    best,
                    score
                )

                alpha = max(
                    alpha,
                    best
                )

                if alpha >= beta:
                    break

            return best

        best = float("inf")

        for move in moves:

            next_board = board.copy(
                stack=False
            )

            next_board.push(move)

            score = self._tactical_search(
                next_board,
                depth - 1,
                root_color,
                alpha,
                beta
            )

            best = min(
                best,
                score
            )

            beta = min(
                beta,
                best
            )

            if alpha >= beta:
                break

        return best

    def _tactical_score(
        self,
        board,
        move
    ):

        root_color = board.turn

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        immediate_see = (
            self._static_exchange_gain(
                board,
                move
            )
        )

        future = self._tactical_search(
            next_board,
            2,
            root_color
        )

        raw = (
            0.9 * immediate_see
            + 0.8 * future
        )

        return math.tanh(
            raw / 4.0
        )

    def _hanging_score(
        self,
        board,
        move
    ):

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        own_gain = max(
            0.0,
            self._static_exchange_gain(
                board,
                move
            )
        )

        lost = 0.0

        for reply in next_board.legal_moves:

            captured = self._captured_piece(
                next_board,
                reply
            )

            if captured is None:
                continue

            gain = self._static_exchange_gain(
                next_board,
                reply
            )

            if gain > lost:
                lost = gain

        raw = (
            own_gain
            - lost
        )

        return math.tanh(
            raw / 3.0
        )

    def _forcing_count(self, board):

        checks = 0
        captures = 0

        for move in board.legal_moves:

            if board.is_capture(move):
                captures += 1

            if board.gives_check(move):
                checks += 1

        return (
            captures
            + 2 * checks
        )

    def _weights(self, board):

        volatility = (
            self._forcing_count(board)
        )

        tactical = volatility >= 5

        if tactical:

            return {
                "policy": 0.10,
                "tactical": 0.40,
                "value": 0.20,
                "hanging": 0.20,
                "king": 0.08,
                "positional": 0.02
            }

        return {
            "policy": 0.25,
            "tactical": 0.20,
            "value": 0.25,
            "hanging": 0.05,
            "king": 0.05,
            "positional": 0.20
        }

    def _blunder_penalty(
        self,
        board,
        move
    ):

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        worst = 0.0

        for reply in next_board.legal_moves:

            captured = self._captured_piece(
                next_board,
                reply
            )

            if captured is None:
                continue

            gain = self._static_exchange_gain(
                next_board,
                reply
            )

            worst = max(
                worst,
                gain
            )

        if worst >= 9.0:
            return 8.0

        if worst >= 5.0:
            return 4.0

        if worst >= 3.0:
            return 2.0

        if worst >= 2.0:
            return 1.0

        return 0.0

    def _candidate_moves(
        self,
        board,
        policy_probs
    ):

        legal_moves = list(
            board.legal_moves
        )

        scored = []

        for move in legal_moves:

            idx = self.extractor.move_to_idx(
                move
            )

            probability = float(
                policy_probs[idx].item()
            )

            scored.append(
                (
                    probability,
                    move
                )
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        selected = [
            move
            for _, move
            in scored[:12]
        ]

        for move in legal_moves:

            if (
                board.is_capture(move)
                or board.gives_check(move)
                or move.promotion is not None
            ):

                if move not in selected:
                    selected.append(move)

        return selected

    def _policy_normalized(
        self,
        probability,
        minimum,
        maximum
    ):

        if maximum - minimum < 1e-9:
            return 0.5

        value = (
            math.log(
                max(
                    probability,
                    1e-9
                )
            )
            - math.log(
                max(
                    minimum,
                    1e-9
                )
            )
        )

        denominator = (
            math.log(
                max(
                    maximum,
                    1e-9
                )
            )
            - math.log(
                max(
                    minimum,
                    1e-9
                )
            )
        )

        return max(
            0.0,
            min(
                1.0,
                value / denominator
            )
        )

    def _move_score(
        self,
        board,
        move,
        policy_value,
        policy_min,
        policy_max
    ):

        next_board = board.copy(
            stack=False
        )

        next_board.push(move)

        weights = self._weights(
            board
        )

        policy_score = self._policy_normalized(
            policy_value,
            policy_min,
            policy_max
        )

        tactical_score = self._tactical_score(
            board,
            move
        )

        hanging_score = self._hanging_score(
            board,
            move
        )

        value_score = self._network_value(
            next_board
        )

        positional_score = self._positional_score(
            next_board
        )

        king_score = self._king_safety_score(
            next_board
        )

        penalty = self._blunder_penalty(
            board,
            move
        )

        score = (
            weights["policy"]
            * policy_score
            +
            weights["tactical"]
            * tactical_score
            +
            weights["value"]
            * value_score
            +
            weights["hanging"]
            * hanging_score
            +
            weights["king"]
            * king_score
            +
            weights["positional"]
            * positional_score
            -
            penalty
        )

        if next_board.is_checkmate():
            score += 20.0

        if board.gives_check(move):
            score += 0.08

        return score

    def _evaluate_candidates(
        self,
        board,
        candidates,
        policy_probs
    ):

        values = []

        policy_values = []

        for move in candidates:

            idx = self.extractor.move_to_idx(
                move
            )

            probability = float(
                policy_probs[idx].item()
            )

            policy_values.append(
                probability
            )

        policy_min = min(
            policy_values
        )

        policy_max = max(
            policy_values
        )

        for move, probability in zip(
            candidates,
            policy_values
        ):

            score = self._move_score(
                board,
                move,
                probability,
                policy_min,
                policy_max
            )

            values.append(
                (
                    score,
                    probability,
                    move
                )
            )

        return values

    def get_move(self, board):

        self._value_cache.clear()

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

        candidates = self._candidate_moves(
            canon,
            policy_probs
        )

        scored = self._evaluate_candidates(
            canon,
            candidates,
            policy_probs
        )

        if not scored:
            move = next(
                iter(
                    canon.legal_moves
                )
            )

        else:

            scored.sort(
                key=lambda item: item[0],
                reverse=True
            )

            move = scored[0][2]

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

        return _CUSTOM_CACHE[
            model_path
        ]

    return StockfishEngine(
        difficulty
    )
