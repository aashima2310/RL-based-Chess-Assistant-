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

MATE_SCORE = 100000
INF = 1000000
MAX_DEPTH = 6
NODE_LIMIT = 60000
QUIESCENCE_DEPTH = 8
TT_MAX_SIZE = 200000


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

        self._value_cache = {}
        self._tt = {}
        self._history = {}
        self._killers = {}
        self._root_policy = {}
        self._nodes = 0
        self._stop = False

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
                "MCTS medium."
            )

    @staticmethod
    def _piece_value(piece_type):
        return {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }.get(piece_type, 0)

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

    @staticmethod
    def _position_key(board):
        return (
            board.board_fen(),
            board.turn,
            board.castling_rights,
            board.ep_square
        )

    @staticmethod
    def _move_key(move):
        return (
            move.from_square,
            move.to_square,
            move.promotion
        )

    def _network_value(self, board):
        key = board.fen()

        if key in self._value_cache:
            return self._value_cache[key]

        black = board.turn == chess.BLACK
        canon = board.mirror() if black else board

        try:
            w_idx = self.extractor.get_halfkp_indices(
                canon,
                chess.WHITE
            )

            b_idx = self.extractor.get_halfkp_indices(
                canon,
                chess.BLACK
            )

            w_acc = self.model.backbone.refresh_accumulator(
                w_idx
            )

            b_acc = self.model.backbone.refresh_accumulator(
                b_idx
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

        if len(self._value_cache) > 10000:
            self._value_cache.pop(
                next(iter(self._value_cache))
            )

        return value

    def _material_cp(self, board):
        score = 0

        for square in chess.SQUARES:
            piece = board.piece_at(square)

            if piece is None:
                continue

            value = self._piece_value(
                piece.piece_type
            )

            if piece.color == board.turn:
                score += value
            else:
                score -= value

        return score

    def _center_cp(self, board):
        centers = (
            chess.D4,
            chess.E4,
            chess.D5,
            chess.E5
        )

        score = 0

        for square in centers:
            piece = board.piece_at(square)

            if piece is not None:
                if piece.color == board.turn:
                    score += 20
                else:
                    score -= 20

            own = len(
                board.attackers(
                    board.turn,
                    square
                )
            )

            enemy = len(
                board.attackers(
                    not board.turn,
                    square
                )
            )

            score += 5 * (own - enemy)

        return score

    def _mobility_cp(self, board):
        own_moves = board.legal_moves.count()

        other = board.copy(stack=False)
        other.turn = not board.turn

        if other.is_check():
            enemy_moves = 0
        else:
            enemy_moves = other.legal_moves.count()

        return max(
            -100,
            min(
                100,
                4 * (own_moves - enemy_moves)
            )
        )

    def _king_safety_cp(self, board):
        own_king = board.king(board.turn)
        enemy_king = board.king(not board.turn)

        if own_king is None or enemy_king is None:
            return 0

        score = 0

        if board.is_check():
            score -= 180

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

        score += 15 * (
            own_attackers
            - enemy_attackers
        )

        return max(
            -250,
            min(250, score)
        )

    def _static_eval(self, board):
        if board.is_checkmate():
            return -MATE_SCORE

        if board.is_stalemate():
            return 0

        if board.is_insufficient_material():
            return 0

        material = self._material_cp(board)
        network = self._network_value(board)

        network_cp = network * 600

        mobility = self._mobility_cp(board)
        center = self._center_cp(board)
        king = self._king_safety_cp(board)

        score = (
            0.48 * material
            + 0.38 * network_cp
            + 0.06 * mobility
            + 0.03 * center
            + 0.05 * king
        )

        return int(score)

    def _see(self, board, move):
        captured = self._captured_piece(
            board,
            move
        )

        if captured is None:
            return 0

        victim_value = self._piece_value(
            captured.piece_type
        )

        attacker = board.piece_at(
            move.from_square
        )

        if attacker is None:
            return 0

        attacker_value = self._piece_value(
            attacker.piece_type
        )

        next_board = board.copy(stack=False)
        next_board.push(move)

        target = move.to_square

        recaptures = [
            m
            for m in next_board.legal_moves
            if m.to_square == target
        ]

        if not recaptures:
            return victim_value

        best_reply = -INF

        for reply in recaptures:
            gain = self._see(
                next_board,
                reply
            )

            captured_again = self._captured_piece(
                next_board,
                reply
            )

            if captured_again is None:
                continue

            reply_gain = (
                self._piece_value(
                    captured_again.piece_type
                )
                - gain
            )

            best_reply = max(
                best_reply,
                reply_gain
            )

        if best_reply == -INF:
            return victim_value

        return victim_value - min(
            attacker_value,
            best_reply
        )

    def _is_quiet(self, board, move):
        return not (
            board.is_capture(move)
            or move.promotion is not None
            or board.gives_check(move)
        )

    def _ordered_moves(
        self,
        board,
        tt_move=None,
        ply=0,
        quiescence=False
    ):
        moves = list(board.legal_moves)

        scored = []

        killer_moves = self._killers.get(
            ply,
            []
        )

        for move in moves:
            score = 0

            if tt_move is not None and move == tt_move:
                score += 100000000

            if move in killer_moves:
                score += 500000

            if board.gives_check(move):
                score += 400000

            if move.promotion is not None:
                score += 350000

            if board.is_capture(move):
                captured = self._captured_piece(
                    board,
                    move
                )

                attacker = board.piece_at(
                    move.from_square
                )

                victim_value = (
                    self._piece_value(
                        captured.piece_type
                    )
                    if captured is not None
                    else 0
                )

                attacker_value = (
                    self._piece_value(
                        attacker.piece_type
                    )
                    if attacker is not None
                    else 0
                )

                see = self._see(
                    board,
                    move
                )

                score += 300000
                score += victim_value * 100
                score -= attacker_value
                score += see * 100

                if see >= 0:
                    score += 50000

            key = self._move_key(move)

            score += self._history.get(
                key,
                0
            )

            policy = self._root_policy.get(
                move.uci(),
                0.0
            )

            score += int(
                policy * 10000
            )

            if quiescence:
                if not (
                    board.is_capture(move)
                    or move.promotion is not None
                    or board.gives_check(move)
                ):
                    continue

            scored.append(
                (score, move)
            )

        scored.sort(
            key=lambda x: x[0],
            reverse=True
        )

        return [
            move
            for _, move in scored
        ]

    def _store_killer(self, ply, move):
        if not self._is_quiet(
            self._current_board,
            move
        ):
            return

        killers = self._killers.setdefault(
            ply,
            []
        )

        if move in killers:
            return

        killers.insert(
            0,
            move
        )

        if len(killers) > 2:
            del killers[2:]

    def _update_history(self, move, depth):
        key = self._move_key(move)

        self._history[key] = (
            self._history.get(key, 0)
            + depth * depth
        )

        if self._history[key] > 100000:
            self._history[key] //= 2

    def _terminal_score(self, board, ply):
        if board.is_checkmate():
            return -MATE_SCORE + ply

        if board.is_stalemate():
            return 0

        if board.is_insufficient_material():
            return 0

        if board.halfmove_clock >= 100:
            return 0

        return None

    def _quiescence(
        self,
        board,
        alpha,
        beta,
        ply,
        qdepth,
        path
    ):
        self._nodes += 1

        if self._nodes >= NODE_LIMIT:
            self._stop = True
            return self._static_eval(board)

        terminal = self._terminal_score(
            board,
            ply
        )

        if terminal is not None:
            return terminal

        key = self._position_key(board)

        if path.get(key, 0) >= 2:
            return 0

        in_check = board.is_check()

        if not in_check:
            stand_pat = self._static_eval(
                board
            )

            if stand_pat >= beta:
                return stand_pat

            if stand_pat > alpha:
                alpha = stand_pat

            if qdepth <= 0:
                return stand_pat

            moves = self._ordered_moves(
                board,
                ply=ply,
                quiescence=True
            )
        else:
            moves = self._ordered_moves(
                board,
                ply=ply,
                quiescence=False
            )

        if not moves:
            return self._static_eval(board)

        best = -INF

        for move in moves:
            if self._stop:
                break

            if (
                not in_check
                and board.is_capture(move)
                and move.promotion is None
            ):
                if self._see(board, move) < -100:
                    continue

            child = board.copy(
                stack=False
            )

            child.push(move)

            child_key = self._position_key(
                child
            )

            path[child_key] = (
                path.get(child_key, 0) + 1
            )

            score = -self._quiescence(
                child,
                -beta,
                -alpha,
                ply + 1,
                qdepth - 1,
                path
            )

            path[child_key] -= 1

            if path[child_key] == 0:
                del path[child_key]

            if score > best:
                best = score

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        if best == -INF:
            return self._static_eval(board)

        return best

    def _search(
        self,
        board,
        depth,
        alpha,
        beta,
        ply,
        path
    ):
        self._nodes += 1

        if self._nodes >= NODE_LIMIT:
            self._stop = True
            return self._static_eval(board)

        terminal = self._terminal_score(
            board,
            ply
        )

        if terminal is not None:
            return terminal

        key = self._position_key(board)

        if path.get(key, 0) >= 2:
            return 0

        alpha_original = alpha

        tt_entry = self._tt.get(key)

        tt_move = None

        if tt_entry is not None:
            tt_depth = tt_entry[0]
            tt_score = tt_entry[1]
            tt_flag = tt_entry[2]
            tt_move = tt_entry[3]

            if tt_depth >= depth:
                if tt_flag == 0:
                    return tt_score

                if tt_flag == 1:
                    alpha = max(
                        alpha,
                        tt_score
                    )

                elif tt_flag == 2:
                    beta = min(
                        beta,
                        tt_score
                    )

                if alpha >= beta:
                    return tt_score

        if depth <= 0:
            return self._quiescence(
                board,
                alpha,
                beta,
                ply,
                QUIESCENCE_DEPTH,
                path
            )

        moves = self._ordered_moves(
            board,
            tt_move=tt_move,
            ply=ply
        )

        if not moves:
            return self._static_eval(board)

        best_score = -INF
        best_move = moves[0]

        for move_index, move in enumerate(moves):
            if self._stop:
                break

            child = board.copy(
                stack=False
            )

            child.push(move)

            child_key = self._position_key(
                child
            )

            path[child_key] = (
                path.get(child_key, 0) + 1
            )

            gives_check = child.is_check()

            extension = 1 if gives_check else 0

            next_depth = depth - 1 + extension

            if (
                move_index >= 4
                and depth >= 3
                and self._is_quiet(board, move)
                and not gives_check
            ):
                reduced_depth = max(
                    1,
                    next_depth - 1
                )

                score = -self._search(
                    child,
                    reduced_depth,
                    -alpha - 1,
                    -alpha,
                    ply + 1,
                    path
                )

                if (
                    score > alpha
                    and score < beta
                ):
                    score = -self._search(
                        child,
                        next_depth,
                        -beta,
                        -alpha,
                        ply + 1,
                        path
                    )
            else:
                score = -self._search(
                    child,
                    next_depth,
                    -beta,
                    -alpha,
                    ply + 1,
                    path
                )

            path[child_key] -= 1

            if path[child_key] == 0:
                del path[child_key]

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                if self._is_quiet(board, move):
                    killers = self._killers.setdefault(
                        ply,
                        []
                    )

                    if move not in killers:
                        killers.insert(
                            0,
                            move
                        )

                        if len(killers) > 2:
                            del killers[2:]

                    self._update_history(
                        move,
                        depth
                    )

                break

        if self._stop:
            return best_score

        if best_score <= alpha_original:
            flag = 2
        elif best_score >= beta:
            flag = 1
        else:
            flag = 0

        self._tt[key] = (
            depth,
            best_score,
            flag,
            best_move
        )

        if len(self._tt) > TT_MAX_SIZE:
            for _ in range(
                max(1, TT_MAX_SIZE // 20)
            ):
                try:
                    self._tt.pop(
                        next(iter(self._tt))
                    )
                except StopIteration:
                    break

        return best_score

    def _root_search(
        self,
        board,
        depth,
        alpha,
        beta
    ):
        key = self._position_key(board)

        tt_entry = self._tt.get(key)

        tt_move = None

        if tt_entry is not None:
            tt_move = tt_entry[3]

        moves = self._ordered_moves(
            board,
            tt_move=tt_move,
            ply=0
        )

        if not moves:
            return None, 0

        best_move = moves[0]
        best_score = -INF

        path = {
            key: 1
        }

        for move in moves:
            if self._stop:
                break

            child = board.copy(
                stack=False
            )

            child.push(move)

            child_key = self._position_key(
                child
            )

            path[child_key] = (
                path.get(child_key, 0) + 1
            )

            gives_check = child.is_check()

            extension = 1 if gives_check else 0

            score = -self._search(
                child,
                depth - 1 + extension,
                -beta,
                -alpha,
                1,
                path
            )

            path[child_key] -= 1

            if path[child_key] == 0:
                del path[child_key]

            if self._stop:
                break

            if score > best_score:
                best_score = score
                best_move = move

            if score > alpha:
                alpha = score

            if alpha >= beta:
                break

        self._tt[key] = (
            depth,
            best_score,
            0,
            best_move
        )

        return best_move, best_score

    def _root_policy(self, board):
        try:
            black = board.turn == chess.BLACK

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

            self._root_policy = {}

            for move in canon.legal_moves:
                try:
                    idx = self.extractor.move_to_idx(
                        move
                    )

                    self._root_policy[
                        move.uci()
                    ] = float(
                        policy_probs[idx].item()
                    )

                except Exception:
                    continue

        except Exception:
            self._root_policy = {}

    def _book_move(self, board):
        try:
            return self.book.pick(board)
        except Exception:
            return None

    def get_move(self, board):
        if not self.loaded:
            if self._fallback is None:
                self._fallback = StockfishEngine(
                    "medium"
                )

            return self._fallback.get_move(
                board
            )

        book_move = self._book_move(board)

        if book_move is not None:
            return book_move

        legal_moves = list(
            board.legal_moves
        )

        if not legal_moves:
            return None

        if len(legal_moves) == 1:
            return legal_moves[0]

        self._value_cache.clear()
        self._tt.clear()
        self._history.clear()
        self._killers.clear()
        self._root_policy.clear()

        self._nodes = 0
        self._stop = False

        self._root_policy(board)

        black = board.turn == chess.BLACK

        canon = (
            board.mirror()
            if black
            else board
        )

        best_move = next(
            iter(canon.legal_moves)
        )

        best_score = -INF

        for depth in range(
            1,
            MAX_DEPTH + 1
        ):
            if self._stop:
                break

            self._stop = False

            alpha = -INF
            beta = INF

            move, score = self._root_search(
                canon,
                depth,
                alpha,
                beta
            )

            if self._stop:
                break

            if move is not None:
                best_move = move
                best_score = score

        if black:
            best_move = chess.Move(
                chess.square_mirror(
                    best_move.from_square
                ),
                chess.square_mirror(
                    best_move.to_square
                ),
                promotion=best_move.promotion
            )

        if board.is_legal(best_move):
            return best_move

        for move in board.legal_moves:
            return move

        return None


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
