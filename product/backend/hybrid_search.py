
import time

import chess

MATE = 100_000
INF = 10 ** 9
PIECE_CP = {
    chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0,
}
ROOT_MARGIN = 400      # candidates worse than best by more than this are only bounded, not scored exactly


class SearchTimeout(Exception):
    pass


def evaluate(board):
    """Material (+ bishop pair) in centipawns from the side to move's point of view."""
    w, b = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]
    score = 0
    for pt, val in ((chess.PAWN, 100), (chess.KNIGHT, 320), (chess.BISHOP, 330),
                    (chess.ROOK, 500), (chess.QUEEN, 900)):
        mask = board.pieces_mask(pt, chess.WHITE)
        score += val * (chess.popcount(mask) - chess.popcount(board.pieces_mask(pt, chess.BLACK)))
    if chess.popcount(board.bishops & w) >= 2:
        score += 30
    if chess.popcount(board.bishops & b) >= 2:
        score -= 30
    return score if board.turn == chess.WHITE else -score


def _victim_cp(board, move):
    if board.is_en_passant(move):
        return 100
    return PIECE_CP.get(board.piece_type_at(move.to_square), 0)


def _order_key(board, move):
    if board.is_capture(move):
        return 10_000 + 10 * _victim_cp(board, move) - PIECE_CP[board.piece_type_at(move.from_square)] // 10
    if move.promotion:
        return 9_000 + PIECE_CP[move.promotion]
    return 0


class TacticalSearch:
    def __init__(self):
        self.nodes = 0
        self.deadline = INF

    def _tick(self):
        self.nodes += 1
        if (self.nodes & 127) == 0 and time.perf_counter() > self.deadline:
            raise SearchTimeout

    # ---------- quiescence: only forcing moves, so the score is not taken mid-capture ----------
    def qsearch(self, board, alpha, beta, ply, qd):
        self._tick()
        in_check = board.is_check()
        if in_check and qd < 4:
            moves = list(board.legal_moves)
            if not moves:
                return -MATE + ply
            best = -INF
            stand_cp = 0
        else:
            stand = stand_cp = evaluate(board)
            if stand >= beta:
                return stand
            alpha = max(alpha, stand)
            if qd >= 8:
                return stand
            best = stand
            moves = list(board.generate_legal_captures())
        moves.sort(key=lambda m: _order_key(board, m), reverse=True)
        for m in moves:
            if not in_check and not m.promotion and stand_cp + _victim_cp(board, m) + 200 < alpha:
                continue                     # delta pruning: capture can't raise alpha
            board.push(m)
            score = -self.qsearch(board, -beta, -alpha, ply + 1, qd + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    def negamax(self, board, depth, alpha, beta, ply):
        self._tick()
        if board.is_insufficient_material():
            return 0
        if ply <= 2 and ply > 0 and board.is_repetition(2):
            return 0
        if depth <= 0:
            return self.qsearch(board, alpha, beta, ply, 0)
        moves = list(board.legal_moves)
        if not moves:
            return -MATE + ply if board.is_check() else 0
        moves.sort(key=lambda m: _order_key(board, m), reverse=True)
        best = -INF
        for m in moves:
            board.push(m)
            score = -self.negamax(board, depth - 1, -beta, -alpha, ply + 1)
            board.pop()
            if score > best:
                best = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta:
                        break
        return best

    def score_moves(self, board, moves, max_depth=4, time_limit=0.6):
        """Iterative deepening over `moves` (all legal in `board`).
        Returns ({move: centipawns from the mover's view}, depth_reached)."""
        board = board.copy()
        moves = sorted(moves, key=lambda m: _order_key(board, m), reverse=True)
        results, reached = {}, 0
        start = time.perf_counter()
        for depth in range(1, max_depth + 1):
            # depth 1 always completes; deeper iterations are abandoned when time runs out
            self.deadline = INF if depth == 1 else start + time_limit
            cur, best = {}, -INF
            try:
                for m in moves:
                    floor = best - ROOT_MARGIN if best > -INF else None
                    board.push(m)
                    if floor is None:
                        v = -self.negamax(board, depth - 1, -INF, INF, 1)
                    else:
                        v = -self.negamax(board, depth - 1, -INF, -floor, 1)
                        if v <= floor:
                            v = floor - 100      # only a bound: clearly worse than the best move
                    board.pop()
                    cur[m] = v
                    best = max(best, v)
            except SearchTimeout:
                break
            results, reached = cur, depth
            moves.sort(key=lambda m: results[m], reverse=True)
            if best >= MATE - 50:
                break
        return results, reached
