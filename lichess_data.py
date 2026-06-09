import chess.pgn
import pickle
import zstandard as zstd
import io
import os
from tqdm import tqdm

def collect_lichess_data(pgn_file_path="lichess_games.pgn.zst", target_positions=5000000, min_elo=1800):
    print(f"Loading games from {pgn_file_path}...")

    try:
        if pgn_file_path.endswith('.zst'):
            with open(pgn_file_path, 'rb') as f:
                dctx = zstd.ZstdDecompressor()
                with dctx.stream_reader(f) as reader:
                    text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                    data = extract_positions(text_stream, target_positions, min_elo)
        else:
            with open(pgn_file_path, 'r', encoding='utf-8') as text_stream:
                data = extract_positions(text_stream, target_positions, min_elo)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Final save just to make sure everything is completely synchronized
    with open("lichess_data.pkl", 'wb') as out_f:
        pickle.dump(data, out_f)
    print(f"Done! Total unique positions saved: {len(data)}")

def extract_positions(text_stream, target_positions, min_elo, checkpoint_path="lichess_data.pkl"):
    data = []
    
    if os.path.exists(checkpoint_path):
        print(f"Found existing checkpoint file at '{checkpoint_path}'. Loading progress...")
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            print(f"Successfully resumed! Starting from {len(data)} positions.")
        except Exception as e:
            print(f"Failed to read checkpoint ({e}). Starting fresh.")
            data = []

    if len(data) >= target_positions:
        return data[:target_positions]

    checkpoint_interval = 500000  
    next_checkpoint = ((len(data) // checkpoint_interval) + 1) * checkpoint_interval

    with tqdm(total=target_positions, initial=len(data), desc="Extracting positions") as pbar:
        while len(data) < target_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
                print("\nReached the end of your PGN archive file.")
                break 
            
            white_elo = game.headers.get("WhiteElo", "0")
            black_elo = game.headers.get("BlackElo", "0")
            
            if not (white_elo.isdigit() and black_elo.isdigit()):
                continue
                
            if int(white_elo) < min_elo or int(black_elo) < min_elo:
                continue

            board = game.board()
            for move in game.mainline_moves():
                if len(data) >= target_positions:
                    break
                
                data.append((board.fen(), move.uci()))
                board.push(move)
                pbar.update(1)
                
         
                if len(data) >= next_checkpoint:
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(data, f)
                    print(f"\n[Checkpoint Saved] securely backed up {len(data)} positions to disk.")
                    next_checkpoint += checkpoint_interval
                    
    return data

if __name__ == "__main__":
    collect_lichess_data()
