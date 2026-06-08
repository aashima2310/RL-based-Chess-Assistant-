import chess
import torch
import numpy as np

class HalfKPExtractor:

    input_size = 41024   

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

    def get_halfkp_indices(self, board: chess.Board, turn: bool):
       
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
            relative_color = piece.color if turn == chess.WHITE else not piece.color
            
            pt_idx = self.PIECE_TYPE_INDEX[(piece.piece_type, relative_color)]

           
            sq = chess.square_mirror(square) if turn == chess.BLACK else square

            feature_idx = king_sq * 641 + sq * 10 + pt_idx
            active.append(feature_idx)        

        return active

    def indices_to_tensor(self, indices: list) -> torch.Tensor:
        
        vec = torch.zeros(self.input_size, dtype=torch.float32)
        if indices:
            vec[indices] = 1.0
        return vec

    def board_to_halfkp(self, board: chess.Board):
        """Returns (white_vec, black_vec) each of shape (41024,)"""
        w_idx = self.get_halfkp_indices(board, chess.WHITE)
        b_idx = self.get_halfkp_indices(board, chess.BLACK)
        return self.indices_to_tensor(w_idx), self.indices_to_tensor(b_idx)

    def get_legal_moves(self, board: chess.Board) -> torch.Tensor:
        mask = torch.zeros(4096, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move.from_square * 64 + move.to_square] = True
        return mask

    def move_to_idx(self, move: chess.Move) -> int:
        return move.from_square * 64 + move.to_square

    def idx_to_move(self, idx: int) -> chess.Move:
        return chess.Move(idx // 64, idx % 64)
