import chess
from engine import ChessEngine

class GameManager:
    def __init__(self, difficulty="easy", fen=None, engine_type="stockfish"):
        if fen:
            self.board = chess.Board(fen)
        else:
            self.board = chess.Board()
        self.engine = ChessEngine(difficulty=difficulty, engine_type=engine_type)

    def play_user_move(self, move_uci):
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            return {"success": False, "message": "Invalid move format"}
        if move not in self.board.legal_moves:
            return {"success": False, "message": "Illegal move"}
        self.board.push(move)
        return {"success": True, "message": "Move played"}

    def play_engine_move(self):
        engine_move = self.engine.get_move(self.board)
        if isinstance(engine_move, str):
            engine_move = chess.Move.from_uci(engine_move)
        self.board.push(engine_move)
        return engine_move.uci()

    def get_board(self):
        return self.board

    def get_fen(self):
        return self.board.fen()

    def is_game_over(self):
        return self.board.is_game_over()

    def get_result(self):
        if self.board.is_game_over():
            return self.board.result()
        return None
