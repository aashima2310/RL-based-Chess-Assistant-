import chess
from mcts import MCTS

class ChessEngine:
    def __init__(self, difficulty="easy"):
        self.difficulty = difficulty
        self.mcts = MCTS(difficulty)

    def get_move(self, board):
        return self.mcts.search(board)
    