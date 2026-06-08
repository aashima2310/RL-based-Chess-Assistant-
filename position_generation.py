import chess
import chess.engine
import random
import pickle
from tqdm import tqdm

def get_random_board():
    board = chess.Board()
    for _ in range(random.randint(6, 12)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        board.push(random.choice(legal_moves))
    return board

def collect_data(target=500000, stockfish_path="/usr/games/stockfish", depth=10):
    data = []
    with chess.engine.SimpleEngine.popen_uci(stockfish_path) as engine:
        with tqdm(total=target, desc="Collecting positions") as pbar:
            while len(data) < target:
                board = get_random_board()
         
                if board.is_game_over():
                    continue
                info = engine.analyse(board, chess.engine.Limit(depth=depth))
                score = info["score"].white()
                
                cp = score.score(mate_score=10000)
                
                data.append((board.fen(), cp))
                pbar.update(1)
                
    return data

