import os
import chess
import chess.syzygy


class SyzygyProbe:
    def __init__(self, tablebase_dir: str, max_pieces: int = 6):
        if not os.path.isdir(tablebase_dir):
            raise FileNotFoundError(f"Tablebase dir not found: {tablebase_dir}")
        self.tablebase = chess.syzygy.open_tablebase(tablebase_dir)
        self.max_pieces = max_pieces

    def is_available(self, board: chess.Board) -> bool:
        if board.castling_rights:
            return False
        return chess.popcount(board.occupied) <= self.max_pieces

    def probe_wdl(self, board: chess.Board):
        try:
            return self.tablebase.probe_wdl(board)
        except (chess.syzygy.MissingTableError, KeyError, ValueError):
            return None

    def probe_value(self, board: chess.Board):
        wdl = self.probe_wdl(board)
        if wdl is None:
            return None
        return wdl / 2.0  # 2->1.0, 1->0.5, 0->0.0, -1->-0.5, -2->-1.0

    def probe_dtz(self, board: chess.Board):
        try:
            return self.tablebase.probe_dtz(board)
        except (chess.syzygy.MissingTableError, KeyError, ValueError):
            return None

    def close(self):
        self.tablebase.close()
