"""
rulebook.py — Static Chess Knowledge Base
------------------------------------------
Categories: Openings, Tactical Patterns, Endgame Rules, Chess Rules
Each entry has: title, description, example FEN, key principles, common mistakes
Smart linking: maps mistake types to relevant rulebook entries
"""

# Mapping from weakness type to relevant rulebook entries
WEAKNESS_TO_RULES = {
    "hanging_piece":  ["hanging_pieces", "piece_safety", "tactical_awareness"],
    "king_safety":    ["king_safety", "castling", "pawn_shield"],
    "missed_tactic":  ["forks", "pins", "skewers", "discovered_attack"],
    "pawn_structure": ["pawn_structure", "isolated_pawn", "passed_pawn"],
    "endgame_error":  ["king_and_pawn", "rook_endgames", "opposition"],
    "general":        ["piece_development", "center_control", "piece_coordination"],
}

RULEBOOK = {
    # ── CHESS RULES ──────────────────────────────────────────
    "how_pieces_move": {
        "id": "how_pieces_move",
        "category": "Chess Rules",
        "title": "How Pieces Move",
        "description": "Each piece moves differently. The king moves one square in any direction. The queen moves any number of squares diagonally, horizontally, or vertically. The rook moves horizontally or vertically. The bishop moves diagonally. The knight moves in an L-shape. The pawn moves forward one square (or two from starting position) and captures diagonally.",
        "key_principles": [
            "The queen is the most powerful piece — combines rook and bishop movement",
            "Knights are the only pieces that can jump over other pieces",
            "Pawns can only capture diagonally, never straight ahead",
            "The king can never move to a square where it would be in check"
        ],
        "common_mistakes": [
            "Moving the bishop like a rook",
            "Forgetting pawns capture diagonally",
            "Trying to move through pieces with rooks or bishops"
        ],
        "example_fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    },

    "castling": {
        "id": "castling",
        "category": "Chess Rules",
        "title": "Castling",
        "description": "Castling is a special move involving the king and a rook. The king moves two squares toward the rook, and the rook jumps to the other side of the king. Kingside castling (O-O) and queenside castling (O-O-O) are both available if conditions are met.",
        "key_principles": [
            "Neither king nor rook can have moved previously",
            "No pieces can be between king and rook",
            "King cannot be in check, pass through check, or land in check",
            "Castling is usually the best way to secure your king early"
        ],
        "common_mistakes": [
            "Castling when king is in check",
            "Forgetting that moving the king or rook removes castling rights",
            "Castling into a dangerous position"
        ],
        "example_fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    },

    "en_passant": {
        "id": "en_passant",
        "category": "Chess Rules",
        "title": "En Passant",
        "description": "En passant is a special pawn capture. When a pawn advances two squares from its starting position and lands beside an opponent's pawn, the opponent can capture it as if it had only moved one square. This must be done immediately on the next move.",
        "key_principles": [
            "Only available immediately after the pawn moves two squares",
            "The capturing pawn moves diagonally to the square the pawn passed through",
            "The captured pawn is removed even though the capturing pawn doesn't land on its square",
            "If you don't take en passant immediately, you lose the right"
        ],
        "common_mistakes": [
            "Missing the en passant opportunity",
            "Trying to play en passant when the pawn only moved one square",
            "Playing en passant on the wrong move"
        ],
        "example_fen": "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
    },

    # ── TACTICAL PATTERNS ────────────────────────────────────
    "hanging_pieces": {
        "id": "hanging_pieces",
        "category": "Tactical Patterns",
        "title": "Hanging Pieces",
        "description": "A hanging piece is an undefended piece that can be captured for free. Always check if your pieces are defended before making a move. A piece is hanging if it is attacked but has no defenders.",
        "key_principles": [
            "Before every move, check if any of your pieces are undefended",
            "After moving, check if your move left any piece undefended",
            "A piece with more attackers than defenders is effectively hanging",
            "The queen is most valuable — never leave it undefended"
        ],
        "common_mistakes": [
            "Moving a piece without checking if it is defended",
            "Forgetting that a discovered attack can leave a piece hanging",
            "Leaving the queen undefended in the opening"
        ],
        "example_fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 4 4",
    },

    "forks": {
        "id": "forks",
        "category": "Tactical Patterns",
        "title": "Forks",
        "description": "A fork is a tactic where one piece attacks two or more enemy pieces simultaneously. Knights are especially effective at forks because of their unusual movement. The opponent can only save one piece, so you win the other.",
        "key_principles": [
            "Knights are the best forking pieces due to their L-shaped movement",
            "Look for squares where your piece can attack two valuable targets at once",
            "Royal forks (attacking king and queen) win material immediately",
            "Pawns can also create forks by threatening two pieces diagonally"
        ],
        "common_mistakes": [
            "Missing knight fork opportunities",
            "Not looking for fork squares before moving",
            "Allowing your opponent to set up forks against you"
        ],
        "example_fen": "r1bqkb1r/ppp2ppp/2n5/3pp3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4",
    },

    "pins": {
        "id": "pins",
        "category": "Tactical Patterns",
        "title": "Pins",
        "description": "A pin is a tactic where a piece cannot move without exposing a more valuable piece behind it to attack. An absolute pin is when the pinned piece cannot legally move because it would expose the king to check.",
        "key_principles": [
            "Absolute pins (against the king) mean the pinned piece cannot move legally",
            "Relative pins are against valuable pieces — moving is legal but costly",
            "Bishops and rooks are the best pieces for creating pins",
            "Exploit pins by attacking the pinned piece with more pieces"
        ],
        "common_mistakes": [
            "Moving a pinned piece and leaving the king in check",
            "Not recognizing when your piece is pinned",
            "Missing opportunities to pin opponent pieces"
        ],
        "example_fen": "r1bqk2r/ppp2ppp/2n2n2/1B1pp3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
    },

    "skewers": {
        "id": "skewers",
        "category": "Tactical Patterns",
        "title": "Skewers",
        "description": "A skewer is like a reverse pin. A valuable piece is attacked and must move, exposing a less valuable piece behind it to capture. The attacker wins the piece behind.",
        "key_principles": [
            "Skewers target the more valuable piece first",
            "The attacked piece must move, revealing the piece behind",
            "Rooks, bishops, and queens are the skewering pieces",
            "X-ray attacks are related — attacking through a piece"
        ],
        "common_mistakes": [
            "Not seeing the piece behind the attacked piece",
            "Missing skewer opportunities in the endgame",
            "Leaving your king and queen on the same diagonal or file"
        ],
        "example_fen": "4k3/8/8/8/8/8/4Q3/4K3 w - - 0 1",
    },

    "discovered_attack": {
        "id": "discovered_attack",
        "category": "Tactical Patterns",
        "title": "Discovered Attack",
        "description": "A discovered attack occurs when moving one piece reveals an attack by another piece behind it. The moving piece can make its own threat while the revealed piece attacks something else, creating two threats at once.",
        "key_principles": [
            "The moving piece creates one threat, the revealed piece creates another",
            "Discovered checks are especially powerful — opponent must deal with check first",
            "Look for pieces lined up behind other pieces",
            "Double check (two pieces checking simultaneously) is very powerful"
        ],
        "common_mistakes": [
            "Not seeing the piece behind before moving",
            "Missing discovered check opportunities",
            "Allowing your opponent to set up discovered attacks against you"
        ],
        "example_fen": "r1bqk2r/ppp2ppp/2n5/3p4/2BPn3/2N5/PPP2PPP/R1BQK2R w KQkq - 0 7",
    },

    # ── KING SAFETY ──────────────────────────────────────────
    "king_safety": {
        "id": "king_safety",
        "category": "King Safety",
        "title": "King Safety Principles",
        "description": "Keeping your king safe is one of the most important principles in chess. An exposed king is a target for attack. Castle early, maintain a pawn shield, and avoid unnecessary pawn moves in front of your king.",
        "key_principles": [
            "Castle early to tuck your king away safely",
            "Avoid moving pawns in front of your castled king unnecessarily",
            "Keep pieces near your king for defense",
            "Open files near your king are dangerous — rooks exploit them"
        ],
        "common_mistakes": [
            "Delaying castling too long",
            "Moving the f, g, or h pawns in front of your castled king",
            "Trading away pieces that defend your king",
            "Leaving your king in the center during the middlegame"
        ],
        "example_fen": "r1bq1rk1/ppp2ppp/2n2n2/3pp3/2B1P3/2NP1N2/PPP2PPP/R1BQR1K1 w - - 0 8",
    },

    "pawn_shield": {
        "id": "pawn_shield",
        "category": "King Safety",
        "title": "Pawn Shield",
        "description": "The pawns in front of a castled king form a pawn shield that protects it from attack. Maintaining this pawn structure is crucial for king safety. Each pawn pushed weakens the shield.",
        "key_principles": [
            "The three pawns in front of your castled king are your shield",
            "Avoid pushing these pawns unless absolutely necessary",
            "g3 or h3 can be played to give the king a luft (escape square)",
            "A broken pawn shield allows rooks and queens to attack the king"
        ],
        "common_mistakes": [
            "Pushing g or h pawn unnecessarily after castling",
            "Trading the f pawn and opening the f-file against your king",
            "Ignoring opponent attacks on your pawn shield"
        ],
        "example_fen": "r1bq1rk1/ppp2ppp/2n2n2/3p4/2PP4/2N2NP1/PP2PP1P/R1BQ1RK1 w - - 0 8",
    },

    # ── OPENING PRINCIPLES ───────────────────────────────────
    "piece_development": {
        "id": "piece_development",
        "category": "Opening Principles",
        "title": "Piece Development",
        "description": "Development means bringing your pieces from their starting squares to active positions where they control important squares. Good development is the foundation of a strong opening.",
        "key_principles": [
            "Develop knights before bishops",
            "Don't move the same piece twice in the opening without good reason",
            "Castle early to connect your rooks and protect your king",
            "Don't bring your queen out too early — it can be attacked"
        ],
        "common_mistakes": [
            "Moving pawns instead of developing pieces",
            "Bringing the queen out too early",
            "Moving the same piece multiple times in the opening",
            "Neglecting development to chase pawns"
        ],
        "example_fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/2N2N2/PPPP1PPP/R1BQKB1R w KQkq - 4 4",
    },

    "center_control": {
        "id": "center_control",
        "category": "Opening Principles",
        "title": "Center Control",
        "description": "The center (e4, d4, e5, d5 squares) is the most important area of the board. Pieces placed in or controlling the center have more mobility and power than pieces on the edges.",
        "key_principles": [
            "Control the center with pawns (e4 and d4 for White)",
            "Place knights in the center where they control the most squares",
            "Pieces on the edge of the board are generally weaker",
            "A space advantage in the center restricts opponent pieces"
        ],
        "common_mistakes": [
            "Ignoring the center while developing pieces to the edges",
            "Allowing opponent to dominate the center unchallenged",
            "Moving flank pawns instead of center pawns in the opening"
        ],
        "example_fen": "r1bqkb1r/ppp2ppp/2np1n2/4p3/2PPP3/2N2N2/PP3PPP/R1BQKB1R w KQkq - 0 5",
    },

    # ── ENDGAME ──────────────────────────────────────────────
    "king_and_pawn": {
        "id": "king_and_pawn",
        "category": "Endgame",
        "title": "King and Pawn Endgames",
        "description": "King and pawn endgames are the most fundamental endgames. The king becomes an active piece and must escort pawns to promotion. The concept of opposition is critical.",
        "key_principles": [
            "Activate your king immediately in the endgame",
            "Use the rule of the square to calculate if a pawn can promote",
            "Opposition (kings facing each other with one square between) is key",
            "Connected passed pawns are very powerful"
        ],
        "common_mistakes": [
            "Keeping the king passive in the endgame",
            "Not understanding opposition",
            "Allowing opponent king to blockade your passed pawn"
        ],
        "example_fen": "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    },

    "rook_endgames": {
        "id": "rook_endgames",
        "category": "Endgame",
        "title": "Rook Endgames",
        "description": "Rook endgames are the most common endgames. Key principles include the Lucena position (winning with rook and pawn) and the Philidor position (drawing with rook vs rook and pawn).",
        "key_principles": [
            "Rooks belong behind passed pawns (yours or opponent's)",
            "The Lucena position is the key winning technique",
            "The Philidor position is the key drawing technique",
            "Active rooks are much stronger than passive ones"
        ],
        "common_mistakes": [
            "Placing rook in front of your own passed pawn",
            "Keeping rook passive while opponent's is active",
            "Not knowing the Lucena and Philidor positions"
        ],
        "example_fen": "8/8/8/8/8/4k3/4p3/4K2R w - - 0 1",
    },

    "opposition": {
        "id": "opposition",
        "category": "Endgame",
        "title": "Opposition",
        "description": "Opposition occurs when two kings stand on the same rank, file, or diagonal with an odd number of squares between them. The player who does NOT have to move has the opposition and has a positional advantage.",
        "key_principles": [
            "Direct opposition: kings face each other with one square between",
            "The player to move loses the opposition",
            "Opposition is used to gain access to key squares",
            "Distant opposition can be converted to direct opposition"
        ],
        "common_mistakes": [
            "Not understanding who has the opposition",
            "Losing the opposition in a winning king and pawn endgame",
            "Not using the king actively enough to contest opposition"
        ],
        "example_fen": "8/8/8/3k4/8/3K4/8/8 w - - 0 1",
    },
}


def get_all_entries() -> list:
    """Return all rulebook entries as a list."""
    return list(RULEBOOK.values())


def get_entry(entry_id: str) -> dict:
    """Get a specific rulebook entry by ID."""
    return RULEBOOK.get(entry_id)


def get_by_category(category: str) -> list:
    """Get all entries in a category."""
    return [e for e in RULEBOOK.values() if e["category"] == category]


def get_categories() -> list:
    """Get all unique categories."""
    return list(set(e["category"] for e in RULEBOOK.values()))


def get_relevant_entries(weakness: str) -> list:
    """
    Get rulebook entries relevant to a player's weakness.
    Used for smart linking after game analysis.
    """
    entry_ids = WEAKNESS_TO_RULES.get(weakness, WEAKNESS_TO_RULES["general"])
    entries = []
    for eid in entry_ids:
        entry = RULEBOOK.get(eid)
        if entry:
            entries.append(entry)
    return entries


def search_rulebook(query: str) -> list:
    """Simple keyword search across all entries."""
    query = query.lower()
    results = []
    for entry in RULEBOOK.values():
        if (query in entry["title"].lower() or
            query in entry["description"].lower() or
            any(query in p.lower() for p in entry["key_principles"])):
            results.append(entry)
    return results