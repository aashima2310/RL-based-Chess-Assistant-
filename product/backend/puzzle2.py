import pandas as pd
import os

import os
PUZZLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "lichess_db_puzzle.csv")

WEAKNESS_TO_THEME = {
    "hanging_piece":  "hangingPiece",
    "king_safety":    "kingSafety",
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
    return [
        {
            "puzzle_id": "mock_001",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "moves": ["f3e5", "d8g5", "e5f7"],
            "first_move": "f3e5",
            "solution": ["d8g5", "e5f7"],
            "rating": user_elo,
            "theme": WEAKNESS_TO_THEME.get(weakness, "middlegame"),
            "themes_all": "fork hangingPiece",
            "url": "https://lichess.org",
        },
        {
            "puzzle_id": "mock_002",
            "fen": "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 6",
            "moves": ["d1a4", "c6d4", "a4d4"],
            "first_move": "d1a4",
            "solution": ["c6d4", "a4d4"],
            "rating": user_elo + 50,
            "theme": WEAKNESS_TO_THEME.get(weakness, "middlegame"),
            "themes_all": "pin hangingPiece",
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
        return _mock_puzzles(weakness, user_elo, n)

    sample = filtered.sample(min(n, len(filtered)))
    return [_format_puzzle(row, theme) for _, row in sample.iterrows()]