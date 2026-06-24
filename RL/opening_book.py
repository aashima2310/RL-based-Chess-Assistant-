# RL/opening_book.py
import os
import chess
import chess.polyglot
BOOK_PATH = '/content/drive/MyDrive/chess_rl/opening_book.bin'

def get_book_move(board: chess.Board):

    if not os.path.exists(BOOK_PATH):
        return None
    try:
        with chess.polyglot.open_reader(BOOK_PATH) as reader:
            entry = reader.weighted_choice(board)
            if board.is_legal(entry.move):
                return entry.move
    except:
        return None
    return None
