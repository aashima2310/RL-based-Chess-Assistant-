import pandas as pd
import os

import os
PUZZLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lichess_db_puzzle.csv")

WEAKNESS_TO_THEME = {
    "hanging_piece":  "hangingPiece",
    "king_safety":    "exposedKing",
    "missed_tactic":  "fork",
    "pawn_structure": "pawnEndgame",
    "endgame_error":  "endgame",
    "general":        "middlegame",
}

_puzzle_df = None

def _load_puzzles():
    global _puzzle_df
    if _puzzle_df is not None:
        return _puzzle_df
    if not os.path.exists(PUZZLE_CSV_PATH):
        return None
    print("Loading puzzles...")
    _puzzle_df = pd.read_csv(PUZZLE_CSV_PATH)
    print(f"Loaded {len(_puzzle_df):,} puzzles.")
    return _puzzle_df

def _format_puzzle(row, theme):
    moves_list = str(row.get("Moves", "")).split()
    return {
        "puzzle_id":  str(row.get("PuzzleId", "")),
        "fen":        str(row.get("FEN", "")),
        "moves":      moves_list,
        "first_move": moves_list[0] if moves_list else "",
        "solution":   moves_list[1:] if len(moves_list) > 1 else moves_list,
        "rating":     int(row.get("Rating", 0)),
        "theme":      theme,
        "themes_all": str(row.get("Themes", "")),
        "url":        str(row.get("GameUrl", "")),
    }

def _mock_puzzles(weakness, user_elo, n):
    # Fallback puzzles (used when the Lichess CSV is not deployed). All lines verified legal.
    theme = WEAKNESS_TO_THEME.get(weakness, "middlegame")
    return [
        {
            "puzzle_id": "mock_001",
            "fen": "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
            "moves": ["g8f6", "h5f7"],
            "first_move": "g8f6",
            "solution": ["h5f7"],
            "rating": user_elo,
            "theme": theme,
            "themes_all": "mateIn1 short",
            "url": "https://lichess.org",
        },
        {
            "puzzle_id": "mock_002",
            "fen": "rn1qkbnr/ppp2ppp/3p4/4p3/2B1P1b1/2N2N2/PPPP1PPP/R1BQK2R b KQkq - 3 4",
            "moves": ["g7g6", "f3e5", "g4d1", "c4f7", "e8e7", "c3d5"],
            "first_move": "g7g6",
            "solution": ["f3e5", "g4d1", "c4f7", "e8e7", "c3d5"],
            "rating": user_elo + 50,
            "theme": theme,
            "themes_all": "mateIn3 sacrifice",
            "url": "https://lichess.org",
        },
    ][:n]

def get_puzzles(weakness, user_elo, n=5, elo_range=150):
    df = _load_puzzles()
    if df is None:
        return _mock_puzzles(weakness, user_elo, n)

    theme = WEAKNESS_TO_THEME.get(weakness, "middlegame")
    filtered = df[df["Themes"].str.contains(theme, na=False, case=False)]
    filtered = filtered[filtered["Rating"].between(user_elo - elo_range, user_elo + elo_range)]

    if len(filtered) < n:
        filtered = df[df["Themes"].str.contains(theme, na=False, case=False)]

    if len(filtered) == 0:
        filtered = df[df["Rating"].between(user_elo - 300, user_elo + 300)]
    if len(filtered) == 0:
        filtered = df
    if len(filtered) == 0:
        return _mock_puzzles(weakness, user_elo, n)

    sample = filtered.sample(min(n, len(filtered)))
    return [_format_puzzle(row, theme) for _, row in sample.iterrows()]

def preload_puzzles():
    _load_puzzles()
