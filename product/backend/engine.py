"""Built-in opening book for the RL engine (no external file needed).

Positions are matched by placement/turn/castling/en-passant, so transpositions work.
If `models/opening_book.bin` (Polyglot format) exists it is used first, then the built-in lines.
The engine leaves the book as soon as the opponent plays something outside it.
"""
import os
import random
from collections import defaultdict

import chess

try:
    import chess.polyglot
except Exception:  # pragma: no cover
    chess.polyglot = None

MAX_BOOK_PLY = 16   # stop using the book after this many half-moves

# One line per row, SAN moves from the starting position.
LINES = """
e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Bb3 Ba7
e4 e5 Nf3 Nc6 Bc4 Nf6 d3 Bc5 O-O d6 c3 O-O Re1 a6 Bb3 Ba7
e4 e5 Nf3 Nc6 Bc4 Nf6 Ng5 d5 exd5 Na5 Bb5+ c6 dxc6 bxc6 Be2 h6
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Nxe4 d4 b5 Bb3 d5 dxe5 Be6 c3 Be7
e4 e5 Nf3 Nc6 Bb5 Nf6 O-O Nxe4 d4 Nd6 Bxc6 dxc6 dxe5 Nf5 Qxd8+ Kxd8
e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Nf6 Nxc6 bxc6 e5 Qe7 Qe2 Nd5 c4 Ba6
e4 e5 Nf3 Nc6 d4 exd4 Nxd4 Bc5 Be3 Qf6 c3 Nge7 Bc4 O-O O-O Ne5
e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 d4 d5 Bd3 Be7 O-O Nc6 Re1 Bg4
e4 e5 Nf3 Nf6 Nxe5 d6 Nf3 Nxe4 Nc3 Nxc3 dxc3 Be7 Be3 O-O Qd2 Nd7
e4 e5 Nc3 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d3 Nxc3 bxc3 O-O
e4 e5 Nf3 d6 d4 Nf6 Nc3 Nbd7 Bc4 Be7 O-O O-O Re1 c6 a4 exd4
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be3 e5 Nb3 Be6 f3 Be7 Qd2 O-O O-O-O Nbd7
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 g6 Be3 Bg7 f3 O-O Qd2 Nc6 Bc4 Bd7 O-O-O Rc8
e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Be3 Be6
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 Nf6 Nc3 d6 Bg5 e6 Qd2 a6 O-O-O Bd7 f4 Be7
e4 c5 Nf3 Nc6 d4 cxd4 Nxd4 g6 Nc3 Bg7 Be3 Nf6 Bc4 O-O Bb3 d6
e4 c5 Nf3 e6 d4 cxd4 Nxd4 a6 Bd3 Nc6 Nxc6 bxc6 O-O d5 c4 Nf6 Nc3 Be7
e4 c5 Nf3 e6 d4 cxd4 Nxd4 Nc6 Nc3 Qc7 Be3 a6 Qd2 Nf6 O-O-O Bb4
e4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 d3 d6 Be3 e6 Qd2 Nge7
e4 c5 c3 Nf6 e5 Nd5 d4 cxd4 Nf3 Nc6 cxd4 d6 Bc4 Nb6 Bb5 dxe5 Nxe5 Bd7
e4 e6 d4 d5 Nc3 Nf6 Bg5 Be7 e5 Nfd7 Bxe7 Qxe7 f4 O-O Nf3 c5 Qd2 Nc6
e4 e6 d4 d5 Nc3 Bb4 e5 c5 a3 Bxc3+ bxc3 Ne7 Qg4 O-O Bd3 Nbc6
e4 e6 d4 d5 e5 c5 c3 Nc6 Nf3 Qb6 a3 Nh6 b4 cxd4 cxd4 Nf5
e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 f6
e4 e6 d4 d5 Nd2 c5 exd5 exd5 Ngf3 Nc6 Bb5 Bd6 dxc5 Bxc5 O-O Nge7
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6
e4 c6 d4 d5 Nc3 dxe4 Nxe4 Nd7 Ng5 Ngf6 Bd3 e6 N1f3 Bd6 Qe2 h6 Nxf7 Kxf7
e4 c6 d4 d5 e5 Bf5 Nf3 e6 Be2 c5 Be3 cxd4 Nxd4 Nc6 O-O Nge7
e4 c6 d4 d5 exd5 cxd5 c4 Nf6 Nc3 e6 Nf3 Be7 cxd5 Nxd5 Bd3 Nc6 O-O O-O
e4 d5 exd5 Qxd5 Nc3 Qa5 d4 Nf6 Nf3 c6 Bc4 Bf5 Bd2 e6 Qe2 Bb4
e4 d5 exd5 Nf6 d4 Nxd5 Nf3 Bg4 Be2 e6 O-O Nc6 c4 Nb6 Nc3 Be7
e4 d6 d4 Nf6 Nc3 g6 Nf3 Bg7 Be2 O-O O-O c6 a4 Nbd7 Re1 e5
e4 d6 d4 Nf6 Nc3 g6 f4 Bg7 Nf3 O-O Bd3 Na6 O-O c5 d5 Nc7
e4 g6 d4 Bg7 Nc3 d6 Be3 a6 Qd2 b5 f3 Nd7 Nge2 Ngf6
e4 Nf6 e5 Nd5 d4 d6 Nf3 Bg4 Be2 e6 O-O Be7 c4 Nb6 Nc3 O-O
d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 Nbd7 Rc1 c6 Bd3 dxc4 Bxc4 Nd5
d4 d5 c4 e6 Nc3 Nf6 Nf3 Be7 Bf4 O-O e3 Nbd7 c5 c6 Bd3 b6
d4 d5 c4 dxc4 Nf3 Nf6 e3 e6 Bxc4 c5 O-O a6 Qe2 b5 Bb3 Bb7 Rd1 Nbd7
d4 d5 c4 dxc4 e3 Nf6 Bxc4 e6 Nf3 c5 O-O a6 Qe2 b5 Bb3 Nc6
d4 d5 c4 c6 Nf3 Nf6 Nc3 dxc4 a4 Bf5 e3 e6 Bxc4 Bb4 O-O Nbd7
d4 d5 c4 c6 Nf3 Nf6 Nc3 e6 e3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6 e4 c5
d4 d5 c4 c6 Nc3 Nf6 e3 e6 Nf3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 Bb7
d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7
d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 f3 O-O Be3 e5 Nge2 c6 Qd2 Nbd7 O-O-O a6
d4 Nf6 c4 g6 Nc3 d5 cxd5 Nxd5 e4 Nxc3 bxc3 Bg7 Nf3 c5 Be3 Qa5 Qd2 O-O
d4 Nf6 c4 e6 Nc3 Bb4 e3 O-O Bd3 d5 Nf3 c5 O-O Nc6 a3 Bxc3 bxc3 dxc4 Bxc4 Qc7
d4 Nf6 c4 e6 Nc3 Bb4 Qc2 O-O a3 Bxc3+ Qxc3 b6 Bg5 Bb7 e3 d6 f3 Nbd7
d4 Nf6 c4 e6 Nf3 b6 g3 Bb7 Bg2 Be7 O-O O-O Nc3 Ne4 Qc2 Nxc3 Qxc3 c5 dxc5 bxc5
d4 Nf6 c4 e6 g3 d5 Bg2 Be7 Nf3 O-O O-O dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
d4 Nf6 c4 e6 Nf3 d5 Nc3 Be7 Bg5 h6 Bh4 O-O e3 Ne4 Bxe7 Qxe7 Rc1 c6 Be2 Nxc3 Rxc3 dxc4 Bxc4 Nd7
d4 d5 Nf3 Nf6 Bf4 e6 e3 c5 c3 Nc6 Nbd2 Bd6 Bg3 O-O Bd3 b6
d4 d5 Nf3 Nf6 Bf4 c5 e3 Nc6 c3 Qb6 Qb3 c4 Qxb6 axb6 Nbd2 Bf5
d4 Nf6 Nf3 e6 Bf4 c5 e3 Nc6 Nbd2 cxd4 exd4 Bb4 a3 Bxd2+ Qxd2 d5
d4 Nf6 Nf3 g6 Bf4 Bg7 e3 O-O h3 d6 Be2 Nbd7 O-O b6
d4 f5 g3 Nf6 Bg2 e6 Nf3 Be7 O-O O-O c4 d6 Nc3 Qe8 Re1 Qh5
d4 e6 c4 f5 g3 Nf6 Bg2 Be7 Nc3 O-O Nf3 d6 O-O Qe8
d4 d6 Nf3 Nf6 c4 g6 Nc3 Bg7 e4 O-O Be2 e5 O-O Nc6 d5 Ne7
c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 a3 O-O b4 Be6
c4 e5 Nc3 Nf6 Nf3 Nc6 e3 Bb4 Qc2 Bxc3 bxc3 O-O Be2 Re8 O-O d6
c4 e5 g3 Nf6 Bg2 d5 cxd5 Nxd5 Nc3 Nb6 d3 Be7 Nf3 Nc6 O-O O-O
c4 c5 Nc3 Nc6 g3 g6 Bg2 Bg7 Nf3 e6 O-O Nge7 d3 O-O Rb1 d5
c4 Nf6 Nc3 e6 e4 d5 e5 d4 exf6 dxc3 bxc3 Qxf6 Nf3 Nc6 Bb2 Bd7
c4 Nf6 Nc3 g6 e4 d6 d4 Bg7 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7
c4 e6 Nf3 d5 g3 Nf6 Bg2 Be7 O-O O-O d4 dxc4 Qc2 a6 Qxc4 b5 Qc2 Bb7
Nf3 d5 g3 Nf6 Bg2 e6 O-O Be7 d3 O-O Nbd2 c5 e4 Nc6 Re1 b5
Nf3 d5 c4 d4 b4 c5 bxc5 e6 e3 dxe3 dxe3 Bxc5 Bb2 Nf6
Nf3 Nf6 c4 g6 Nc3 Bg7 e4 d6 d4 O-O Be2 e5 O-O Nc6 d5 Ne7
Nf3 Nf6 g3 g6 Bg2 Bg7 O-O O-O d3 d6 Nbd2 Nc6 e4 e5 c3 a5
Nf3 c5 c4 Nf6 Nc3 e6 g3 b6 Bg2 Bb7 O-O Be7 d4 cxd4 Qxd4 d6
Nf3 g6 e4 Bg7 d4 d6 Nc3 Nf6 Be2 O-O O-O Nc6
b3 d5 Bb2 Nf6 e3 e6 Nf3 Be7 Be2 O-O O-O c5 d3 Nc6
f4 d5 Nf3 Nf6 e3 g6 Be2 Bg7 O-O O-O d3 c5
Nc3 d5 d4 Nf6 Bf4 e6 e3 Bd6 Bxd6 cxd6 f4 Nc6
Nc3 e5 e4 Nf6 f4 d5 fxe5 Nxe4 Nf3 Be7 d3 Nxc3 bxc3 O-O
g3 d5 Bg2 e5 d3 Nf6 Nf3 Nc6 O-O Be7 Nbd2 O-O e4
e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 O-O c3 d5 exd5 Nxd5
d4 d5 c4 e6 Nc3 c6 e3 Nf6 Nf3 Nbd7 Bd3 dxc4 Bxc4 b5 Bd3 a6 e4 c5
""".strip().splitlines()


def _key(board):
    return " ".join(board.fen().split()[:4])


def _build():
    book = defaultdict(lambda: defaultdict(int))
    for line in LINES:
        board = chess.Board()
        for san in line.split():
            move = board.parse_san(san)           # raises if a line is illegal
            book[_key(board)][move.uci()] += 1
            board.push(move)
    return book


class OpeningBook:
    def __init__(self, polyglot_path=None, max_ply=MAX_BOOK_PLY):
        self.max_ply = max_ply
        self.book = _build()
        self.reader = None
        if polyglot_path and os.path.exists(polyglot_path) and chess.polyglot is not None:
            try:
                self.reader = chess.polyglot.open_reader(polyglot_path)
                print(f"[opening_book] using polyglot book {polyglot_path}")
            except Exception as e:
                print(f"[opening_book] could not open {polyglot_path}: {e}")
        print(f"[opening_book] built-in book ready: {len(self.book)} positions")

    def pick(self, board):
        """Return a book chess.Move for this position, or None if out of book."""
        if board.ply() >= self.max_ply:
            return None
        if self.reader is not None:
            try:
                move = self.reader.weighted_choice(board).move
                if board.is_legal(move):
                    return move
            except Exception:
                pass
        options = self.book.get(_key(board))
        if not options:
            return None
        moves, weights = zip(*options.items())
        move = chess.Move.from_uci(random.choices(moves, weights=weights)[0])
        return move if board.is_legal(move) else None
