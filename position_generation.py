import chess
import chess.engine
import random
import pickle
from tqdm import tqdm

def get_random_board():
    board = chess.Board()
    for _ in range(random.randint(5,30)):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            board = chess.Board()
            break
        board.push(random.choice(legal_moves))
    return board

def collect_data(target=500000, stockfish_path="/usr/games/stockfish", depth=12):
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
                if cp is None:
                    continue
                
                data.append((board.fen(), cp))
                pbar.update(1)
                
    return data
if __name__ == "__main__":
    collected_data = collect_data(target=500000)
    
    os.makedirs("data", exist_ok=True)
    with open("data/stockfish_data.pkl", "wb") as f:
        pickle.dump(collected_data, f)
    print(f"Successfully saved {len(collected_data)} positions to data/stockfish_data.pkl")
