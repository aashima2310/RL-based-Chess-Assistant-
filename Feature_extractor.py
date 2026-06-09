import chess
import torch
import numpy as np

class HalfKPExtractor:

    input_size = 40960

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
        """Returns (white_vec, black_vec) each of shape (41024,)"""
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
        mask = torch.zeros(4096, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move.from_square * 64 + move.to_square] = True
        return mask

    def move_to_idx(self, move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square

    def idx_to_move(self, idx: int) -> chess.Move:
        return chess.Move(idx // 64, idx % 64)

    def move_to_policy_target(
        self, mcts_visit_counts: dict,board: chess.Board ) -> torch.Tensor:
        
        policy = torch.zeros(4096, dtype=torch.float32)
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


