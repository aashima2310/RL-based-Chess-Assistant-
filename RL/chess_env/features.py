import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class HalfKPExtractor:

    input_size = 40960
    QUEEN_DIRS = [
        (1, 0), (1, 1), (0, 1), (-1, 1),
        (-1, 0), (-1, -1), (0, -1), (1, -1)
    ]
    KNIGHT_MOVES = [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]
    UNDER_PROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
    UNDER_PROMO_DIRS   = [-1, 0, 1]

    PIECE_TYPE_INDEX = {
        (chess.PAWN,   chess.WHITE): 0,
        (chess.PAWN,   chess.BLACK): 1,
        (chess.KNIGHT, chess.WHITE): 2,
        (chess.KNIGHT, chess.BLACK): 3,
        (chess.BISHOP, chess.WHITE): 4,
        (chess.BISHOP, chess.BLACK): 5,
        (chess.ROOK,   chess.WHITE): 6,
        (chess.ROOK,   chess.BLACK): 7,
        (chess.QUEEN,  chess.WHITE): 8,
        (chess.QUEEN,  chess.BLACK): 9,
    }

    def get_halfkp_indices(self, board: chess.Board, turn: bool) -> list:
        king_sq = board.king(turn)
        if king_sq is None:
            return []
        if turn == chess.BLACK:
            king_sq = chess.square_mirror(king_sq)
        active = []
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            if piece.piece_type == chess.KING:
                continue
            if turn == chess.BLACK:
                relative_color = not piece.color
                sq = chess.square_mirror(square)
            else:
                relative_color = piece.color
                sq = square
            pt_idx = self.PIECE_TYPE_INDEX[(piece.piece_type, relative_color)]
            feature_idx = king_sq * 640 + sq * 10 + pt_idx
            active.append(feature_idx)
        return active

    def indices_to_tensor(self, indices: list) -> torch.Tensor:
        vec = torch.zeros(self.input_size, dtype=torch.float32)
        if indices:
            vec[indices] = 1.0
        return vec

    def board_to_halfkp(self, board: chess.Board) -> tuple:
        w_idx = self.get_halfkp_indices(board, chess.WHITE)
        b_idx = self.get_halfkp_indices(board, chess.BLACK)
        return self.indices_to_tensor(w_idx), self.indices_to_tensor(b_idx)

    def board_to_tensor_769(self, board: chess.Board) -> torch.Tensor:
        PIECE_TO_PLANES = {
            (chess.PAWN,   chess.WHITE): 0,
            (chess.KNIGHT, chess.WHITE): 1,
            (chess.BISHOP, chess.WHITE): 2,
            (chess.ROOK,   chess.WHITE): 3,
            (chess.QUEEN,  chess.WHITE): 4,
            (chess.KING,   chess.WHITE): 5,
            (chess.PAWN,   chess.BLACK): 6,
            (chess.KNIGHT, chess.BLACK): 7,
            (chess.BISHOP, chess.BLACK): 8,
            (chess.ROOK,   chess.BLACK): 9,
            (chess.QUEEN,  chess.BLACK): 10,
            (chess.KING,   chess.BLACK): 11,
        }
        vector = np.zeros(769, dtype=np.float32)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                plane = PIECE_TO_PLANES[(piece.piece_type, piece.color)]
                vector[plane * 64 + square] = 1.0
        if board.turn == chess.WHITE:
            vector[768] = 1.0
        return torch.tensor(vector, dtype=torch.float32)

    def get_legal_moves(self, board: chess.Board) -> torch.Tensor:
        mask = torch.zeros(4672, dtype=torch.float32)
        for move in board.legal_moves:
            if move.promotion is None:
                piece = board.piece_at(move.from_square)
                to_rank = chess.square_rank(move.to_square)
                if piece and piece.piece_type == chess.PAWN and to_rank in (0, 7):
                    move = chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)
            try:
                mask[self.move_to_idx(move)] = 1.0
            except (ValueError, IndexError):
                pass
        return mask

    def move_to_idx(self, move: chess.Move) -> int:
        from_sq = move.from_square
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)
        to_sq   = move.to_square
        to_rank = chess.square_rank(to_sq)
        to_file = chess.square_file(to_sq)

        dr = to_rank - from_rank
        df = to_file - from_file
        if move.promotion is not None and move.promotion != chess.QUEEN:
            piece_idx = self.UNDER_PROMO_PIECES.index(move.promotion)
            dir_idx   = self.UNDER_PROMO_DIRS.index(df)
            move_type = 64 + piece_idx * 3 + dir_idx
            return from_sq * 73 + move_type
        if (abs(dr), abs(df)) in [(2,1),(1,2)]:
            knight_idx = self.KNIGHT_MOVES.index((dr, df))
            return from_sq * 73 + 56 + knight_idx
        steps = max(abs(dr), abs(df))
        unit_dr = dr // steps
        unit_df = df // steps
        dir_idx  = self.QUEEN_DIRS.index((unit_dr, unit_df))
        dist_idx = steps - 1
        move_type = dir_idx * 7 + dist_idx
        return from_sq * 73 + move_type

    def idx_to_move(self, idx: int, board: chess.Board | None = None) -> chess.Move:
        from_sq = idx // 73
        move_type = idx % 73
        from_rank = chess.square_rank(from_sq)
        from_file = chess.square_file(from_sq)

        if move_type >= 64:
            offset = move_type - 64
            piece_idx = offset // 3
            dir_idx = offset % 3
            promotion = self.UNDER_PROMO_PIECES[piece_idx]
            df = self.UNDER_PROMO_DIRS[dir_idx]
            dr = 1 if from_rank == 6 else -1
            to_file = from_file + df
            to_rank = from_rank + dr
            if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
                return chess.Move.null()
            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq, promotion=promotion)

        if move_type >= 56:
            dr, df = self.KNIGHT_MOVES[move_type - 56]
            to_file = from_file + df
            to_rank = from_rank + dr
            if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
                return chess.Move.null()
            to_sq = chess.square(to_file, to_rank)
            return chess.Move(from_sq, to_sq)

        dir_idx = move_type // 7
        dist = move_type % 7 + 1
        dr, df = self.QUEEN_DIRS[dir_idx]
        to_file = from_file + df * dist
        to_rank = from_rank + dr * dist
        if not (0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return chess.Move.null()
        to_sq = chess.square(to_file, to_rank)

        promotion = None
        if board is not None:
            piece = board.piece_at(from_sq)
            to_rank_val = chess.square_rank(to_sq)
            if piece and piece.piece_type == chess.PAWN and to_rank_val in (0, 7):
                promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    def move_to_policy_target(self, mcts_visit_counts: dict, board: chess.Board) -> torch.Tensor:
        policy = torch.zeros(4672, dtype=torch.float32)
        total_visits = sum(mcts_visit_counts.values())
        if total_visits == 0:
            legal_moves = list(board.legal_moves)
            num_moves = len(legal_moves)
            for move in legal_moves:
                policy[self.move_to_idx(move)] = 1.0 / num_moves
        else:
            for move, count in mcts_visit_counts.items():
                policy[self.move_to_idx(move)] = count / total_visits
        return policy


_default_extractor = HalfKPExtractor()

def move_to_index(move: chess.Move) -> int:
    return _default_extractor.move_to_idx(move)

def index_to_move(idx: int, board: chess.Board | None = None) -> chess.Move:
    return _default_extractor.idx_to_move(idx, board)
