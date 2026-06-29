import chess
import chess.pgn
import chess.engine
import io
import os
import shutil


def get_engine_path():
    rl_engine = os.path.join(os.path.dirname(__file__), "..", "..", "engine", "chessrl_engine")
    if os.path.exists(rl_engine):
        return rl_engine

    stockfish = shutil.which("stockfish")
    if stockfish:
        return stockfish

    windows_paths = [
        r"C:\Users\vedik\stockfish\stockfish-windows-x86-64-avx2.exe",
        r"C:\stockfish\stockfish.exe",
        r"C:\Program Files\stockfish\stockfish.exe",
    ]
    for path in windows_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Stockfish not found. Install it and add to PATH, or place in C:\\stockfish\\"
    )


ENGINE_PATH = get_engine_path()

THRESHOLDS = {
    "Good":       (0,   50),
    "Inaccuracy": (50,  100),
    "Mistake":    (100, 200),
    "Blunder":    (200, float("inf")),
}


def classify(cp_loss):
    for label, (lo, hi) in THRESHOLDS.items():
        if lo <= cp_loss < hi:
            return label
    return "Blunder"


def tag_mistake_type(board, move):
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "general"

    board.push(move)
    attackers = board.attackers(not board.turn, move.to_square)
    defenders = board.attackers(board.turn, move.to_square)
    board.pop()

    if attackers and not defenders:
        return "hanging_piece"
    if piece.piece_type == chess.PAWN:
        king_sq = board.king(board.turn)
        if king_sq and chess.square_distance(move.from_square, king_sq) <= 2:
            return "king_safety"
    if piece.piece_type == chess.QUEEN and board.fullmove_number <= 10:
        return "missed_tactic"
    if len(board.piece_map()) <= 10:
        return "endgame_error"
    return "general"


def analyze_pgn(pgn_string):
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_string))
    except Exception as e:
        return {"error": f"PGN parse failed: {str(e)}"}

    if game is None:
        return {"error": "Invalid PGN"}

    try:
        engine = chess.engine.SimpleEngine.popen_uci(ENGINE_PATH)
    except Exception as e:
        return {"error": f"Engine failed: {str(e)}"}

    moves_analysis = []
    board = game.board()
    mistake_counts = {
        "hanging_piece": 0, "king_safety": 0,
        "missed_tactic": 0, "pawn_structure": 0,
        "endgame_error": 0, "general": 0,
    }

    try:
        for move in game.mainline_moves():
            info_before = engine.analyse(board, chess.engine.Limit(depth=12))
            score_before = info_before["score"].white().score(mate_score=10000) or 0

            best_move = engine.play(board, chess.engine.Limit(depth=12)).move

            multi_info = engine.analyse(board, chess.engine.Limit(depth=10), multipv=3)
            alternatives = [str(info["pv"][0]) for info in multi_info if "pv" in info]

            mistake_type = tag_mistake_type(board, move)
            san_move = board.san(move)
            best_san = board.san(best_move) if best_move and board.is_legal(best_move) else str(best_move)
            board.push(move)

            info_after = engine.analyse(board, chess.engine.Limit(depth=12))
            score_after = info_after["score"].white().score(mate_score=10000) or 0

            if board.turn == chess.BLACK:
                cp_loss = max(0, score_before - score_after)
            else:
                cp_loss = max(0, score_after - score_before)

            classification = classify(cp_loss)

            if classification == "Good":
                mistake_type = None
            else:
                mistake_counts[mistake_type] = mistake_counts.get(mistake_type, 0) + 1

            moves_analysis.append({
                "move_number": board.fullmove_number,
                "move": san_move,
                "best_move": best_san,
                "alternatives": alternatives,
                "cp_loss": cp_loss,
                "classification": classification,
                "mistake_type": mistake_type,
                "fen_before": board.fen(),
            })
    finally:
        engine.quit()

    blunders = [m for m in moves_analysis if m["classification"] == "Blunder"]
    mistakes = [m for m in moves_analysis if m["classification"] == "Mistake"]
    inaccuracies = [m for m in moves_analysis if m["classification"] == "Inaccuracy"]
    avg_cp_loss = sum(m["cp_loss"] for m in moves_analysis) / len(moves_analysis) if moves_analysis else 0
    primary_weakness = max(mistake_counts, key=mistake_counts.get) if any(mistake_counts.values()) else "none"

    return {
        "moves": moves_analysis,
        "summary": {
            "total_moves": len(moves_analysis),
            "blunders": len(blunders),
            "mistakes": len(mistakes),
            "inaccuracies": len(inaccuracies),
            "avg_cp_loss": round(avg_cp_loss, 1),
            "mistake_counts": mistake_counts,
            "primary_weakness": primary_weakness,
        }
    }
