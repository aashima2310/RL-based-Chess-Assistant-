import chess
import chess.engine
import chess.pgn
import pickle
import os
from tqdm import tqdm


from google.colab import drive
drive.mount('/content/drive')

def extract_elite_data(pgn_path, stockfish_path="/usr/games/stockfish", target=1_000_000):
  
    save_dir = "/content/drive/MyDrive/chess_data"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_file = f"{save_dir}/elite_data_checkpoint.pkl"

    data = []
    games_to_skip = 0
    
    if os.path.exists(checkpoint_file):
        print("Found existing checkpoint on Drive. Loading...")
        with open(checkpoint_file, "rb") as f:
            saved_state = pickle.load(f)
            data = saved_state.get("data", [])
            games_to_skip = saved_state.get("games_processed", 0)
        print(f"Resuming from {len(data)} positions. Skipping first {games_to_skip} games to avoid duplicates.")

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    engine.configure({"Threads": 1, "Hash": 64})

    games_processed = 0
    pbar = tqdm(total=target, initial=len(data), desc="Extracting Elite Positions")

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        while len(data) < target:
            game = chess.pgn.read_game(f)
            if not game: 
                print("\nReached the end of the PGN file.")
                break
            
            games_processed += 1
            
            if games_processed <= games_to_skip:
                continue

            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                
                if 15 < board.fullmove_number < 50 and not board.is_check():
                    info = engine.analyse(board, chess.engine.Limit(depth=15))
                    score = info["score"].white()
                    cp = score.score(mate_score=10000)
                    
                    if cp is not None and abs(cp) < 300: 
                        data.append((board.fen(), cp / 100.0))
                        pbar.update(1)
                        
                        if len(data) >= target: 
                            break
            
            if games_processed % 100 == 0:
                with open(checkpoint_file, "wb") as cp_file:
                    pickle.dump({"data": data, "games_processed": games_processed}, cp_file)

    engine.quit()
    
    final_path = f"{save_dir}/elite_data_final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(data, f)
    print(f"\nSuccess! Saved exactly {len(data)} high-quality positions to {final_path}")
    return data

if __name__ == "__main__":

    pgn_filename = "lichess_elite_2024-01.pgn" 
    
    if not os.path.exists(pgn_filename):
        print(f"ERROR: Cannot find {pgn_filename}. Please upload the elite PGN to Colab.")
    else:
        extract_elite_data(pgn_filename)
