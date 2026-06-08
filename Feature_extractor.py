import chess
import torch 
import numpy as np

class FeatureExtractor:

    PIECE_TO_PLANES = {
        (chess.PAWN, chess.WHITE) :0,
        (chess.KNIGHT, chess.WHITE) :1,
        (chess.BISHOP, chess.WHITE) :2,
        (chess.ROOK, chess.WHITE) :3,
        (chess.QUEEN, chess.WHITE) :4,
        (chess.KING, chess.WHITE) :5,
        (chess.PAWN, chess.BLACK) :6,
        (chess.KNIGHT, chess.BLACK) :7,
        (chess.BISHOP, chess.BLACK) :8,
        (chess.ROOK, chess.BLACK) :9,
        (chess.QUEEN, chess.BLACK) :10,
        (chess.KING, chess.BLACK) :11,

    }

    def board_to_tensor(self, board : chess.Board) -> torch.Tensor :
        vector = np.zeros(769, dtype= np.float32)
        for square in chess.SQUARES :
            piece = board.piece_at(square)
            if piece is not None :
                plane = self.PIECE_TO_PLANES[(piece.piece_type, piece.color)]
                index = plane * 64 + square
                vector[index] = 1.0
        if board.turn == chess.WHITE :
            vector[768] = 1
        
        return torch.tensor(vector, dtype=torch.float32)
    
    def board_to_tensor_batch(self, boards : list) -> torch.Tensor:
        vectors = [self.board_to_tensor(b) for b in boards]
        return torch.stack(vectors)
    
    def get_legal_moves(self , board : chess.Board) -> torch.Tensor:
        mask = torch.zeros(4096, dtype = torch.bool)
        for move in board.legal_moves:
            idx = move.from_square*64 + move.to_square
            mask[idx] = True
        return mask
    
    def move_to_idx(self, move : chess.Move)-> int:
        return move.from_square*64 + move.to_square
    
    def idx_to_move(self, idx : int) -> chess.Move:
        from_square = idx//64
        to_square = idx%64
        return chess.Move(from_square, to_square )
    
    def move_to_policy_target(self, mcts_visit_counts : dict, board: chess.Board) -> torch.Tensor:
        policy = torch.zeros(4096, dtype = torch.float)
        total_visits = sum(mcts_visit_counts.values())

        # for fallback
        if total_visits == 0:
            for move in board.legal_moves:
                legal_moves = board.legal_moves
                idx = self.move_to_idx(move)
                policy[idx] = 1.0/len(legal_moves)
        else :
            for move, count in mcts_visit_counts.items():
                idx = self.move_to_idx(move)
                policy[idx] = count/total_visits
        return policy
    
