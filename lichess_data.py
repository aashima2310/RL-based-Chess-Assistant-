import chess.pgn
import pickle
import zstandard as zstd
import io
import os
from tqdm import tqdm

def collect_lichess_data(pgn_file_path, target_positions=5000000, min_elo=1800):
    data = []
    
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

    print(f"Saving {len(data)} positions to lichess_data.pkl...")

    os.makedirs("data", exist_ok=True)
    with open("lichess_data.pkl", 'wb') as out_f:
        pickle.dump(data, out_f)
    print("Done!")

def extract_positions(text_stream, target_positions, min_elo):
    data = []
    with tqdm(total=target_positions, desc="Extracting positions") as pbar:
        while len(data) < target_positions:
            game = chess.pgn.read_game(text_stream)
            if game is None:
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
                
    return data
