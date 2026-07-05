import os
import chess
import chess.polyglot

class OpeningBook:

    def __init__(self, book_path, max_book_ply=20):
        self.book_path = book_path
        self.max_book_ply = max_book_ply
        self._reader = None

        if os.path.exists(book_path):
            try:
                self._reader = chess.polyglot.open_reader(book_path)
            except Exception as e:
                print(f"[opening_book] failed to open {book_path}: {e}")
                self._reader = None
        else:
            print(f"[opening_book] no book found at {book_path} — book moves disabled")

    def get_move(self, board: chess.Board, weighted: bool = True):
        """Returns a chess.Move from the book, or None if out of book,
        past max_book_ply, or no book loaded."""
        if self._reader is None:
            return None
        if board.ply() >= self.max_book_ply:
            return None

        try:
            entry = self._reader.weighted_choice(board) if weighted else self._reader.find(board)
        except IndexError:
            return None  
        except Exception as e:
            print(f"[opening_book] unexpected error probing book: {e}")
            return None

        if board.is_legal(entry.move):
            return entry.move
        return None

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_default_book(config):

    return OpeningBook(config.BOOK_PATH, max_book_ply=config.MAX_BOOK_PLY)
