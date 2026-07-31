import streamlit as st
import requests
import chess
import chess.pgn
import chess.svg
import io
import time
import base64
from PIL import Image, ImageDraw, ImageFont
from streamlit_image_coordinates import streamlit_image_coordinates

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="ChessRL — Your Personal Chess Coach",
    page_icon="♟",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────

def piece_to_data(piece):
    """
    Converts a chess.Piece into simple storable data.
    This is used for captured pieces.
    """
    if piece is None:
        return None

    return {
        "piece_type": piece.piece_type,
        "color": piece.color
    }


def piece_data_to_piece(piece_data):
    """
    Converts stored piece data back into a chess.Piece.
    """
    if not piece_data:
        return None

    return chess.Piece(
        piece_data["piece_type"],
        piece_data["color"]
    )


def piece_data_to_unicode(piece_data):
    """
    Converts stored captured piece data into a chess symbol.
    """
    piece = piece_data_to_piece(piece_data)

    if piece is None:
        return ""

    return piece.unicode_symbol()


def get_captured_piece_data(board, move_uci):
    """
    Checks whether a move captures a piece.
    Handles normal captures and en passant.
    Returns captured piece data or None.
    """
    try:
        move = chess.Move.from_uci(move_uci)
    except ValueError:
        return None

    if move not in board.legal_moves:
        return None

    # Special capture: en passant
    if board.is_en_passant(move):
        captured_square = chess.square(
            chess.square_file(move.to_square),
            chess.square_rank(move.from_square)
        )
        captured_piece = board.piece_at(captured_square)
        return piece_to_data(captured_piece)

    # Normal capture
    captured_piece = board.piece_at(move.to_square)
    return piece_to_data(captured_piece)


def render_captured_piece_images(piece_data_list):
    """
    Displays captured pieces as chess piece images.
    If SVG rendering fails, it falls back to unicode symbols.
    """
    if not piece_data_list:
        st.caption("No captures yet.")
        return

    html = ""

    for piece_data in piece_data_list:
        piece = piece_data_to_piece(piece_data)

        if piece is None:
            continue

        try:
            svg = chess.svg.piece(piece, size=38)
            b64 = base64.b64encode(svg.encode()).decode()

            html += (
                f'<img src="data:image/svg+xml;base64,{b64}" '
                f'width="36" style="margin-right:6px; margin-bottom:6px;" />'
            )

        except Exception:
            html += (
                f'<span style="font-size:32px; margin-right:6px;">'
                f'{piece.unicode_symbol()}'
                f'</span>'
            )

    st.markdown(html, unsafe_allow_html=True)


def get_last_move_for_svg():
    """
    Returns the latest move so the visual board can highlight it.
    Engine move is preferred because it is usually the latest move.
    """
    if not st.session_state.play_history:
        return None

    last_item = st.session_state.play_history[-1]
    move_uci = last_item.get("engine_move") or last_item.get("user_move")

    if not move_uci:
        return None

    try:
        return chess.Move.from_uci(move_uci)
    except Exception:
        return None


def build_play_engine_context():
    """
    Builds context from the Play Engine page.
    This will be sent to the chatbot backend later.
    """
    try:
        board = chess.Board(st.session_state.play_fen)
        game_over = board.is_game_over()
        result = board.result() if game_over else None
    except Exception:
        game_over = False
        result = None

    return {
        "source": "play_engine",
        "difficulty": st.session_state.play_difficulty,
        "moves": st.session_state.play_history,
        "current_fen": st.session_state.play_fen,
        "game_over": game_over,
        "result": result,
        "captured_by_you": st.session_state.play_captured_by_you,
        "captured_by_engine": st.session_state.play_captured_by_engine,
    }


def play_move_with_backend(user_move):
    """
    Sends a clicked user move to the backend.
    Backend validates the move, engine replies, and frontend updates board.
    """
    old_fen = st.session_state.play_fen
    board_before_user = chess.Board(old_fen)

    user_captured = get_captured_piece_data(board_before_user, user_move)

    MIN_THINK_TIME = 2.0  # seconds

    try:
        with st.spinner("Engine thinking..."):
            start = time.time()

            resp = requests.post(
                f"{API_URL}/engine_move",
                json={
                    "fen": old_fen,
                    "move": user_move,
                    "difficulty": st.session_state.play_difficulty.lower(),
                    # Send configured engine type to backend
                    "engine_type": "stockfish" if st.session_state.play_engine_type == "Stockfish" else "custom"
                },
                timeout=120,
            )

            elapsed = time.time() - start
            if elapsed < MIN_THINK_TIME:
                time.sleep(MIN_THINK_TIME - elapsed)

        if resp.status_code == 200:
            data = resp.json()

            if data.get("success"):
                engine_move = data.get("engine_move")
                engine_captured = None

                try:
                    board_after_user = chess.Board(old_fen)
                    board_after_user.push(chess.Move.from_uci(user_move))

                    if engine_move is not None:
                        engine_captured = get_captured_piece_data(
                            board_after_user,
                            engine_move
                        )

                except Exception:
                    engine_captured = None

                st.session_state.play_fen = data.get("fen")

                if user_captured is not None:
                    st.session_state.play_captured_by_you.append(user_captured)

                if engine_captured is not None:
                    st.session_state.play_captured_by_engine.append(engine_captured)

                st.session_state.play_history.append({
                    "user_move": user_move,
                    "engine_move": engine_move,
                    "user_captured": user_captured,
                    "engine_captured": engine_captured,
                })

                st.session_state.chat_play_context = build_play_engine_context()
                st.session_state.play_selected_square = None

                if engine_move is None:
                    st.session_state.play_status_type = "success"
                    st.session_state.play_status_message = (
                        f"You played `{user_move}`. Game ended."
                    )
                else:
                    st.session_state.play_status_type = "success"
                    st.session_state.play_status_message = (
                        f"You played `{user_move}`. Engine replied `{engine_move}`."
                    )

                st.rerun()

            else:
                st.session_state.play_selected_square = None
                st.session_state.play_status_type = "error"
                st.session_state.play_status_message = data.get(
                    "message",
                    "Illegal move."
                )
                st.rerun()

        else:
            st.session_state.play_selected_square = None
            st.session_state.play_status_type = "error"
            st.session_state.play_status_message = "Backend returned an error."
            st.rerun()

    except requests.exceptions.ConnectionError:
        st.session_state.play_selected_square = None
        st.session_state.play_status_type = "error"
        st.session_state.play_status_message = (
            "Backend not running. Start it with: "
            "`python -m uvicorn backend.main:app --reload --port 8000`"
        )
        st.rerun()

    except Exception as e:
        st.session_state.play_selected_square = None
        st.session_state.play_status_type = "error"
        st.session_state.play_status_message = f"Error: {e}"
        st.rerun()


def handle_square_click(square_name):
    """
    Two-click movement:
    first click = source square
    second click = destination square
    """
    board = chess.Board(st.session_state.play_fen)

    if board.is_game_over():
        st.session_state.play_status_type = "info"
        st.session_state.play_status_message = "Game is already over."
        st.rerun()

    clicked_square = chess.parse_square(square_name)
    selected_square = st.session_state.play_selected_square

    # First click
    if selected_square is None:
        piece = board.piece_at(clicked_square)

        if piece is None:
            st.session_state.play_status_type = "error"
            st.session_state.play_status_message = "Please select a piece first."
            st.rerun()

        if piece.color != board.turn:
            st.session_state.play_status_type = "error"
            st.session_state.play_status_message = "That piece cannot move now."
            st.rerun()

        st.session_state.play_selected_square = square_name
        st.session_state.play_status_type = "info"
        st.session_state.play_status_message = (
            f"Selected `{square_name}`. Now click the destination square."
        )
        st.rerun()

    # Second click
    else:
        if selected_square == square_name:
            st.session_state.play_selected_square = None
            st.session_state.play_status_type = "info"
            st.session_state.play_status_message = "Selection cleared."
            st.rerun()

        move_uci = selected_square + square_name

        try:
            from_square = chess.parse_square(selected_square)
            to_square = chess.parse_square(square_name)
            moving_piece = board.piece_at(from_square)

            if (
                moving_piece is not None
                and moving_piece.piece_type == chess.PAWN
                and chess.square_rank(to_square) in [0, 7]
            ):
                promotion_map = {
                    "Queen": "q",
                    "Rook": "r",
                    "Bishop": "b",
                    "Knight": "n",
                }
                move_uci += promotion_map.get(
                    st.session_state.play_promotion_piece,
                    "q"
                )

        except Exception:
            pass

        play_move_with_backend(move_uci)


# ─────────────────────────────────────────────────────────────
# Clickable board rendering (PNG + pixel-coordinate click mapping)
# ─────────────────────────────────────────────────────────────

LIGHT_SQ = (240, 217, 181)
DARK_SQ = (181, 136, 99)
HILITE_SQ = (247, 236, 89)
LASTMOVE_SQ = (170, 200, 100)

PIECE_GLYPH = {
    (chess.PAWN, True): "♙", (chess.KNIGHT, True): "♘", (chess.BISHOP, True): "♗",
    (chess.ROOK, True): "♖", (chess.QUEEN, True): "♕", (chess.KING, True): "♔",
    (chess.PAWN, False): "♟", (chess.KNIGHT, False): "♞", (chess.BISHOP, False): "♝",
    (chess.ROOK, False): "♜", (chess.QUEEN, False): "♛", (chess.KING, False): "♚",
}


@st.cache_resource
def get_piece_font(size):
    """
    Loads a font capable of rendering chess glyphs.
    Falls back to matplotlib's bundled DejaVuSans if no system font is found.
    """
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        try:
            import matplotlib
            import os
            path = os.path.join(
                os.path.dirname(matplotlib.__file__),
                "mpl-data", "fonts", "ttf", "DejaVuSans.ttf"
            )
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()


def render_board_image(board, size=480, selected_square=None, last_move=None):
    """
    Draws the board as a plain PNG (no SVG/cairosvg dependency needed).
    Row 0 = rank 8 (top), col 0 = file a (left) — standard orientation.
    """
    sq = size // 8
    img = Image.new("RGB", (size, size), DARK_SQ)
    draw = ImageDraw.Draw(img)
    font = get_piece_font(int(sq * 0.68))

    for row in range(8):
        for col in range(8):
            rank = 7 - row
            file = col
            square = chess.square(file, rank)

            x0, y0 = col * sq, row * sq
            color = LIGHT_SQ if (file + rank) % 2 == 0 else DARK_SQ

            if last_move and square in (last_move.from_square, last_move.to_square):
                color = LASTMOVE_SQ
            if selected_square == square:
                color = HILITE_SQ

            draw.rectangle([x0, y0, x0 + sq, y0 + sq], fill=color)

            piece = board.piece_at(square)
            if piece:
                glyph = PIECE_GLYPH[(piece.piece_type, piece.color)]
                bbox = draw.textbbox((0, 0), glyph, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                tx = x0 + (sq - tw) / 2 - bbox[0]
                ty = y0 + (sq - th) / 2 - bbox[1]

                if piece.color:  # white piece
                    draw.text((tx, ty), glyph, font=font, fill="white",
                               stroke_width=2, stroke_fill="black")
                else:  # black piece
                    draw.text((tx, ty), glyph, font=font, fill="black",
                               stroke_width=1, stroke_fill="white")

    return img


def render_clickable_board():
    """
    Single clickable chess board — no invisible second board.
    Uses streamlit_image_coordinates to get real pixel clicks directly on
    the rendered board image, then maps those pixels to a square name.
    """
    board = chess.Board(st.session_state.play_fen)

    selected_square = None
    if st.session_state.play_selected_square:
        try:
            selected_square = chess.parse_square(st.session_state.play_selected_square)
        except Exception:
            selected_square = None

    last_move = get_last_move_for_svg()

    size = 480
    sq_size = size // 8

    img = render_board_image(
        board,
        size=size,
        selected_square=selected_square,
        last_move=last_move,
    )

    st.caption("Click a square to select it, then click the destination square.")

    coords = streamlit_image_coordinates(
        img,
        key="play_board_click",
    )

    if coords is not None:
        click_id = (coords["x"], coords["y"])

        if st.session_state.get("last_board_click") != click_id:
            st.session_state.last_board_click = click_id

            col = min(coords["x"] // sq_size, 7)
            row = min(coords["y"] // sq_size, 7)
            file_idx = int(col)
            rank = 7 - int(row)

            files = "abcdefgh"
            square_name = f"{files[file_idx]}{rank + 1}"

            handle_square_click(square_name)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────

if "user_id" not in st.session_state:
    st.session_state.user_id = "default_user"
if "game_id" not in st.session_state:
    st.session_state.game_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "puzzles" not in st.session_state:
    st.session_state.puzzles = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chatbot_opening" not in st.session_state:
    st.session_state.chatbot_opening = None
if "current_move_idx" not in st.session_state:
    st.session_state.current_move_idx = 0

if "play_fen" not in st.session_state:
    st.session_state.play_fen = chess.Board().fen()
if "play_history" not in st.session_state:
    st.session_state.play_history = []
if "play_difficulty" not in st.session_state:
    st.session_state.play_difficulty = "Easy"
if "play_engine_type" not in st.session_state:
    st.session_state.play_engine_type = "Stockfish"
if "play_selected_square" not in st.session_state:
    st.session_state.play_selected_square = None
if "play_status_message" not in st.session_state:
    st.session_state.play_status_message = ""
if "play_status_type" not in st.session_state:
    st.session_state.play_status_type = "info"
if "play_captured_by_you" not in st.session_state:
    st.session_state.play_captured_by_you = []
if "play_captured_by_engine" not in st.session_state:
    st.session_state.play_captured_by_engine = []
if "play_promotion_piece" not in st.session_state:
    st.session_state.play_promotion_piece = "Queen"
if "last_board_click" not in st.session_state:
    st.session_state.last_board_click = None

if "chat_context_source" not in st.session_state:
    st.session_state.chat_context_source = None
if "chat_play_context" not in st.session_state:
    st.session_state.chat_play_context = None
if "chat_thinking" not in st.session_state:
    st.session_state.chat_thinking = False


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("♟ ChessRL")
    st.caption("RL-Powered Chess Coach")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "📤 Analyze Game",
            "♟ Play Engine",
            "🧩 Puzzles",
            "💬 Coach Chat",
            "📖 Rulebook",
            "🎓 Rules Chat",
            "📊 Dashboard",
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.session_state.user_id = st.text_input(
        "User ID",
        value=st.session_state.user_id
    )
    st.caption(f"Logged in as: `{st.session_state.user_id}`")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — ANALYZE GAME
# ════════════════════════════════════════════════════════════════

if page == "📤 Analyze Game":
    st.title("📤 Analyze Your Game")
    st.caption(
        "Upload a PGN to get move-by-move analysis, mistake classification, "
        "and coaching feedback."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Upload PGN")
        pgn_input = st.text_area(
            "Paste your PGN here",
            height=200,
            placeholder='[Event "My Game"]\n1. e4 e5 2. Nf3 Nc6 3. Bc4 Nf6 4. Qxf7',
            help="You can copy PGN from Chess.com or Lichess game review page"
        )

        uploaded_file = st.file_uploader("Or upload a .pgn file", type=["pgn"])
        if uploaded_file:
            pgn_input = uploaded_file.read().decode("utf-8")
            st.success("PGN file loaded.")

        analyze_btn = st.button(
            "🔍 Analyze Game",
            type="primary",
            use_container_width=True
        )

    if analyze_btn and pgn_input.strip():
        with st.spinner("Analyzing your game... (this takes 10–30s depending on game length)"):
            try:
                resp = requests.post(
                    f"{API_URL}/upload_pgn",
                    json={
                        "pgn": pgn_input,
                        "user_id": st.session_state.user_id
                    },
                    timeout=120
                )

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.analysis = data["analysis"]
                    st.session_state.game_id = data["game_id"]
                    st.session_state.puzzles = data.get("puzzles", [])
                    st.session_state.chatbot_opening = data.get("chatbot_opening")
                    st.session_state.chat_history = []
                    st.session_state.current_move_idx = 0
                    st.session_state.chat_context_source = "pgn_analysis"

                    st.success("✅ Analysis complete!")
                    st.rerun()
                else:
                    st.error(
                        f"Analysis failed: {resp.json().get('detail', 'Unknown error')}"
                    )

            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Backend not running. Start it with: "
                    "`python -m uvicorn backend.main:app --reload --port 8000`"
                )

    elif analyze_btn:
        st.warning("Please paste a PGN first.")

    if st.session_state.analysis:
        analysis = st.session_state.analysis
        summary = analysis["summary"]
        moves = analysis["moves"]

        st.divider()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Blunders", summary["blunders"])
        m2.metric("🟠 Mistakes", summary["mistakes"])
        m3.metric("🟡 Inaccuracies", summary["inaccuracies"])
        m4.metric("📉 Avg CP Loss", f"{summary['avg_cp_loss']:.1f}")

        st.divider()

        col_board, col_moves = st.columns([1, 1])

        with col_board:
            st.subheader("Board Position")

            nav1, nav2, nav3, nav4 = st.columns(4)

            with nav1:
                if st.button("⏮ Start"):
                    st.session_state.current_move_idx = 0

            with nav2:
                if (
                    st.button("◀ Prev")
                    and st.session_state.current_move_idx > 0
                ):
                    st.session_state.current_move_idx -= 1

            with nav3:
                if (
                    st.button("Next ▶")
                    and st.session_state.current_move_idx < len(moves) - 1
                ):
                    st.session_state.current_move_idx += 1

            with nav4:
                if st.button("End ⏭"):
                    st.session_state.current_move_idx = len(moves) - 1

            current_idx = st.session_state.current_move_idx

            if current_idx < len(moves):
                current_move = moves[current_idx]

                try:
                    board = chess.Board(current_move["fen_before"])
                    last_move = (
                        chess.Move.from_uci(current_move["move_uci"])
                        if current_move.get("move_uci")
                        else None
                    )

                    svg = chess.svg.board(
                        board,
                        lastmove=last_move,
                        size=380
                    )
                    b64 = base64.b64encode(svg.encode()).decode()

                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" width="380"/>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Board error: {e}")

            if current_idx < len(moves):
                m = moves[current_idx]

                color = {
                    "Blunder": "🔴",
                    "Mistake": "🟠",
                    "Inaccuracy": "🟡",
                    "Good": "🟢"
                }.get(m["classification"], "⚪")

                st.info(
                    f"**Move {m['move_number']}: {m['move']}** "
                    f"{color} {m['classification']}\n\n"
                    f"CP Loss: **{m['cp_loss']}** | "
                    f"Best: `{m['best_move']}`\n\n"
                    f"Type: `{m['mistake_type'] or 'n/a'}`"
                )

        with col_moves:
            st.subheader("Move List")

            for i, m in enumerate(moves):
                color_map = {
                    "Blunder": "🔴",
                    "Mistake": "🟠",
                    "Inaccuracy": "🟡",
                    "Good": "🟢",
                }
                icon = color_map.get(m["classification"], "⚪")

                if st.button(
                    f"{icon} {m['move_number']}. {m['move']} ({m['classification']})",
                    key=f"move_{i}",
                    use_container_width=True,
                ):
                    st.session_state.current_move_idx = i
                    st.rerun()

        st.divider()
        st.subheader("🎯 Weakness Summary")

        weakness_cols = st.columns(len(summary["mistake_counts"]))
        for col, (wtype, count) in zip(
            weakness_cols,
            summary["mistake_counts"].items()
        ):
            if count > 0:
                col.metric(wtype.replace("_", " ").title(), count)

        primary = summary["primary_weakness"].replace("_", " ").title()
        st.warning(
            f"**Primary weakness this game:** {primary} — "
            "see Puzzles tab for targeted training"
        )


# ════════════════════════════════════════════════════════════════
# PAGE 2 — PLAY ENGINE
# ════════════════════════════════════════════════════════════════

elif page == "♟ Play Engine":
    st.title("♟ Play Against Engine")
    st.caption(
        "Click a piece, then click the destination square. "
        "The engine will reply automatically."
    )

    col_board, col_side = st.columns([1.4, 1])

    with col_board:
        st.subheader("Chess Board")

        render_clickable_board()

        try:
            board = chess.Board(st.session_state.play_fen)

            st.caption(
                f"Current turn: **{'White' if board.turn == chess.WHITE else 'Black'}**"
            )

            if board.is_game_over():
                st.success(f"Game over. Result: {board.result()}")

        except Exception as e:
            st.error(f"Board error: {e}")

    with col_side:
        st.subheader("Captured Pieces")

        st.markdown("**Captured by You**")
        render_captured_piece_images(st.session_state.play_captured_by_you)

        st.markdown("**Captured by Engine**")
        render_captured_piece_images(st.session_state.play_captured_by_engine)

        st.divider()
        st.subheader("Controls")

        # Engine Selector Addition
        st.session_state.play_engine_type = st.selectbox(
            "Select Opponent",
            ["Stockfish", "Custom AI"],
            index=["Stockfish", "Custom AI"].index(st.session_state.play_engine_type)
        )

        difficulty = st.selectbox(
            "Select Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(
                st.session_state.play_difficulty
            )
        )

        st.session_state.play_difficulty = difficulty

        st.session_state.play_promotion_piece = st.selectbox(
            "Pawn promotion piece",
            ["Queen", "Rook", "Bishop", "Knight"],
            index=["Queen", "Rook", "Bishop", "Knight"].index(
                st.session_state.play_promotion_piece
            )
        )

        if st.session_state.play_selected_square:
            st.info(
                f"Selected square: `{st.session_state.play_selected_square}`"
            )
        else:
            st.info("No square selected yet.")

        if st.button("Clear Selection", use_container_width=True):
            st.session_state.play_selected_square = None
            st.session_state.play_status_type = "info"
            st.session_state.play_status_message = "Selection cleared."
            st.rerun()

        reset_btn = st.button(
            "🔄 New Game",
            use_container_width=True
        )

        if reset_btn:
            st.session_state.play_fen = chess.Board().fen()
            st.session_state.play_history = []
            st.session_state.chat_play_context = None
            st.session_state.play_selected_square = None
            st.session_state.play_captured_by_you = []
            st.session_state.play_captured_by_engine = []
            st.session_state.play_status_type = "success"
            st.session_state.play_status_message = "New game started."
            st.rerun()

        if st.session_state.play_status_message:
            if st.session_state.play_status_type == "success":
                st.success(st.session_state.play_status_message)
            elif st.session_state.play_status_type == "error":
                st.error(st.session_state.play_status_message)
            else:
                st.info(st.session_state.play_status_message)

    st.divider()
    st.subheader("Move History")

    if not st.session_state.play_history:
        st.info("No moves played yet.")
    else:
        for i, item in enumerate(st.session_state.play_history, start=1):
            engine_text = (
                item["engine_move"]
                if item["engine_move"] is not None
                else "Game Over"
            )

            user_capture = piece_data_to_unicode(item.get("user_captured"))
            engine_capture = piece_data_to_unicode(item.get("engine_captured"))

            capture_text = ""

            if user_capture:
                capture_text += f" | You captured: {user_capture}"

            if engine_capture:
                capture_text += f" | Engine captured: {engine_capture}"

            st.write(
                f"**{i}.** You: `{item['user_move']}` | "
                f"Engine: `{engine_text}`"
                f"{capture_text}"
            )

        st.divider()
        st.subheader("💬 Send this game to Coach Chat")

        st.caption(
            "This sends the Play Engine game history as chatbot context. "
            "The backend chatbot can later use this to answer questions about this played game."
        )

        if st.button(
            "Use this Play Engine game in Coach Chat",
            use_container_width=True
        ):
            st.session_state.chat_context_source = "play_engine"
            st.session_state.chat_play_context = build_play_engine_context()
            st.session_state.chat_history = []
            st.session_state.chatbot_opening = (
                "I loaded your Play Engine game. "
                "You can ask me about your moves, the engine replies, or the current position."
            )
            st.success("Play Engine context sent to Coach Chat.")
            st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE 3 — PUZZLES
# ════════════════════════════════════════════════════════════════

elif page == "🧩 Puzzles":
    st.title("🧩 Recommended Puzzles")

    try:
        resp = requests.get(
            f"{API_URL}/profile/{st.session_state.user_id}",
            timeout=120
        )
        if resp.status_code == 200:
            games_played = resp.json()["games_played"]
        else:
            games_played = 0
    except Exception:
        games_played = 0

    if games_played < 3:
        st.info(
            f"🔒 Puzzle recommendations unlock after 3 games analyzed. "
            f"You've analyzed {games_played}/3 games. Keep going!"
        )
        st.progress(games_played / 3)
        st.stop()

    if not st.session_state.puzzles:
        try:
            resp = requests.get(
                f"{API_URL}/puzzles/{st.session_state.user_id}",
                timeout=120
            )

            if resp.status_code == 200:
                data = resp.json()
                st.session_state.puzzles = data.get("puzzles", [])

        except Exception:
            pass

    if not st.session_state.puzzles:
        st.info("Analyze a game first to get personalized puzzle recommendations.")
        st.stop()

    puzzles = st.session_state.puzzles
    weakness = (
        puzzles[0].get("theme", "").replace("_", " ").title()
        if puzzles
        else ""
    )

    st.caption(f"📌 These puzzles target your weakness: **{weakness}**")
    st.divider()

    for i, puzzle in enumerate(puzzles):
        with st.expander(
            f"Puzzle {i+1} | Rating: {puzzle['rating']} | Theme: {puzzle['theme']}",
            expanded=(i == 0)
        ):
            p_col1, p_col2 = st.columns([1, 1])

            with p_col1:
                try:
                    board = chess.Board(puzzle["fen"])

                    if puzzle.get("first_move"):
                        board.push_uci(puzzle["first_move"])

                    svg = chess.svg.board(board, size=320)
                    b64 = base64.b64encode(svg.encode()).decode()

                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" width="320"/>',
                        unsafe_allow_html=True
                    )

                except Exception as e:
                    st.error(f"Board error: {e}")

            with p_col2:
                st.markdown(f"**Rating:** {puzzle['rating']}")
                st.markdown(f"**Themes:** `{puzzle['themes_all']}`")
                st.markdown("**Your task:** Find the best move for the side to move.")

                if st.button(f"👁 Reveal Solution", key=f"solution_{i}"):
                    solution_moves = " → ".join(puzzle.get("solution", []))
                    st.success(f"Solution: `{solution_moves}`")

                if puzzle.get("url"):
                    st.markdown(f"[Play on Lichess ↗]({puzzle['url']})")

    st.divider()

    if st.button("🔄 Get New Puzzles", use_container_width=True):
        try:
            resp = requests.get(
                f"{API_URL}/puzzles/{st.session_state.user_id}",
                timeout=120
            )

            if resp.status_code == 200:
                st.session_state.puzzles = resp.json().get("puzzles", [])
                st.rerun()

        except Exception:
            st.error("Backend not reachable.")


# ════════════════════════════════════════════════════════════════
# PAGE 4 — COACH CHAT
# ════════════════════════════════════════════════════════════════

elif page == "💬 Coach Chat":
    st.title("💬 Ask Your Coach")
    st.caption(
        "Chat with the coach using either PGN analysis context or Play Engine context."
    )

    has_pgn_context = (
        st.session_state.game_id is not None
        and st.session_state.analysis is not None
    )
    has_play_context = (
        st.session_state.chat_play_context is not None
        or len(st.session_state.play_history) > 0
    )

    if not has_pgn_context and not has_play_context:
        st.info(
            "No chat context found yet. Analyze a PGN game or play against the engine first."
        )
        st.stop()

    if st.session_state.chat_context_source is None:
        if has_pgn_context:
            st.session_state.chat_context_source = "pgn_analysis"
        elif has_play_context:
            st.session_state.chat_context_source = "play_engine"

    st.subheader("Current Chat Context")

    context_cols = st.columns(2)

    with context_cols[0]:
        if has_pgn_context:
            if st.button("Use PGN Analysis Context", use_container_width=True):
                st.session_state.chat_context_source = "pgn_analysis"
                st.session_state.chat_history = []
                st.session_state.chatbot_opening = (
                    "I loaded your analyzed PGN game. "
                    "Ask me about any move, mistake, blunder, or weakness."
                )
                st.rerun()
        else:
            st.button(
                "Use PGN Analysis Context",
                disabled=True,
                use_container_width=True
            )

    with context_cols[1]:
        if has_play_context:
            if st.button("Use Play Engine Context", use_container_width=True):
                st.session_state.chat_context_source = "play_engine"
                st.session_state.chat_play_context = build_play_engine_context()
                st.session_state.chat_history = []
                st.session_state.chatbot_opening = (
                    "I loaded your Play Engine game. "
                    "Ask me about your moves, engine replies, or current position."
                )
                st.rerun()
        else:
            st.button(
                "Use Play Engine Context",
                disabled=True,
                use_container_width=True
            )

    if st.session_state.chat_context_source == "pgn_analysis":
        st.success("Active context: PGN Analysis")

        if st.session_state.analysis:
            summary = st.session_state.analysis.get("summary", {})
            st.write(
                f"Blunders: `{summary.get('blunders', 0)}` | "
                f"Mistakes: `{summary.get('mistakes', 0)}` | "
                f"Primary weakness: `{summary.get('primary_weakness', 'unknown')}`"
            )

    elif st.session_state.chat_context_source == "play_engine":
        st.success("Active context: Play Engine Game")
        st.session_state.chat_play_context = build_play_engine_context()

        st.write(
            f"Moves played: `{len(st.session_state.play_history)}` | "
            f"Difficulty: `{st.session_state.play_difficulty}`"
        )

    st.divider()

    if st.session_state.chatbot_opening:
        with st.chat_message("assistant"):
            st.write(st.session_state.chatbot_opening)
    else:
        with st.chat_message("assistant"):
            st.write(
                "Hi, I am your chess coach. Ask me about your moves, mistakes, blunders, or the current game."
            )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.chat_thinking:
        st.chat_input("Coach is thinking... please wait", disabled=True)

    else:
        if question := st.chat_input("Ask about any move... e.g. 'Why was move 15 bad?'"):
            st.session_state.chat_thinking = True
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.write(question)

            payload = {
                "question": question,
                "game_id": st.session_state.game_id,
                "user_id": st.session_state.user_id,
                "history": st.session_state.chat_history[:-1],
                "context_source": st.session_state.chat_context_source,
                "play_context": None,
            }

            if st.session_state.chat_context_source == "play_engine":
                st.session_state.chat_play_context = build_play_engine_context()
                payload["play_context"] = st.session_state.chat_play_context

            with st.chat_message("assistant"):
                st.markdown("⏳ *Coach is thinking...*")

                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=120
                    )

                    if resp.status_code == 200:
                        answer = resp.json().get(
                            "response",
                            "Sorry, something went wrong."
                        )
                    else:
                        answer = "Chatbot backend returned an error."

                except requests.exceptions.ConnectionError:
                    answer = "Backend not reachable. Make sure FastAPI is running."

                except Exception as e:
                    answer = f"Chatbot error: {e}"

                st.write(answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })
            st.session_state.chat_thinking = False
            st.rerun()


# ════════════════════════════════════════════════════════════════
# PAGE 5 — RULEBOOK
# ════════════════════════════════════════════════════════════════

elif page == "📖 Rulebook":
    st.title("📖 Chess Rulebook")
    st.caption(
        "Learn chess concepts, tactics, and principles. Smart links to your game weaknesses."
    )

    search_query = st.text_input(
        "🔍 Search rulebook",
        placeholder="e.g. fork, castling, pin..."
    )

    try:
        resp = requests.get(
            f"{API_URL}/rulebook/relevant/{st.session_state.user_id}",
            timeout=10
        )

        if resp.status_code == 200:
            data = resp.json()
            weakness = data.get("weakness", "general")
            relevant = data.get("entries", [])

            if relevant and not search_query:
                st.success(
                    f"📌 Based on your weakness "
                    f"(**{weakness.replace('_', ' ')}**), "
                    "we recommend these entries:"
                )

                for entry in relevant:
                    with st.expander(
                        f"⭐ {entry['title']} — {entry['category']}"
                    ):
                        st.markdown(f"**{entry['description']}**")

                        st.markdown("**Key Principles:**")
                        for p in entry["key_principles"]:
                            st.markdown(f"- {p}")

                        st.markdown("**Common Mistakes:**")
                        for m in entry["common_mistakes"]:
                            st.markdown(f"- ⚠️ {m}")

                        if entry.get("example_fen"):
                            try:
                                board = chess.Board(entry["example_fen"])
                                svg = chess.svg.board(board, size=300)
                                b64 = base64.b64encode(svg.encode()).decode()

                                st.markdown(
                                    f'<img src="data:image/svg+xml;base64,{b64}" width="300"/>',
                                    unsafe_allow_html=True
                                )

                            except Exception:
                                pass

                st.divider()

    except Exception:
        pass

    if search_query:
        try:
            resp = requests.get(
                f"{API_URL}/rulebook/search",
                params={"q": search_query},
                timeout=10
            )

            if resp.status_code == 200:
                results = resp.json().get("results", [])

                if results:
                    st.subheader(f"Search results for '{search_query}'")
                else:
                    st.warning(f"No results found for '{search_query}'")

            else:
                results = []

        except Exception:
            results = []
            st.error("Backend not reachable.")

    else:
        try:
            resp = requests.get(f"{API_URL}/rulebook", timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                results = data.get("entries", [])
            else:
                results = []

        except Exception:
            results = []

    if results and not search_query:
        categories = list(set(e["category"] for e in results))

        for category in sorted(categories):
            st.subheader(f"📚 {category}")

            cat_entries = [
                e for e in results
                if e["category"] == category
            ]

            for entry in cat_entries:
                with st.expander(f"{entry['title']}"):
                    st.markdown(entry["description"])

                    st.markdown("**Key Principles:**")
                    for p in entry["key_principles"]:
                        st.markdown(f"- {p}")

                    st.markdown("**Common Mistakes:**")
                    for m in entry["common_mistakes"]:
                        st.markdown(f"- ⚠️ {m}")

                    if entry.get("example_fen"):
                        try:
                            board = chess.Board(entry["example_fen"])
                            svg = chess.svg.board(board, size=300)
                            b64 = base64.b64encode(svg.encode()).decode()

                            st.markdown(
                                f'<img src="data:image/svg+xml;base64,{b64}" width="300"/>',
                                unsafe_allow_html=True
                            )

                        except Exception:
                            pass

    elif results and search_query:
        for entry in results:
            with st.expander(f"{entry['title']} — {entry['category']}"):
                st.markdown(entry["description"])

                st.markdown("**Key Principles:**")
                for p in entry["key_principles"]:
                    st.markdown(f"- {p}")

                st.markdown("**Common Mistakes:**")
                for m in entry["common_mistakes"]:
                    st.markdown(f"- ⚠️ {m}")


# ════════════════════════════════════════════════════════════════
# PAGE 6 — RULES CHAT
# ════════════════════════════════════════════════════════════════

elif page == "🎓 Rules Chat":
    st.title("🎓 Chess Rules Assistant")
    st.caption("Ask me anything about chess rules, tactics, and concepts.")

    if "rules_chat_history" not in st.session_state:
        st.session_state.rules_chat_history = []
    if "rules_thinking" not in st.session_state:
        st.session_state.rules_thinking = False

    if not st.session_state.rules_thinking:
        try:
            resp = requests.get(
                f"{API_URL}/rulebook/relevant/{st.session_state.user_id}",
                timeout=10
            )

            if resp.status_code == 200:
                data = resp.json()
                weakness = data.get("weakness", "general")
                relevant = data.get("entries", [])

                if relevant and weakness != "general":
                    st.info(
                        f"💡 Based on your weakness "
                        f"(**{weakness.replace('_', ' ')}**), "
                        f"try asking about: "
                        f"{', '.join(e['title'] for e in relevant[:3])}"
                    )

        except Exception:
            pass

    st.markdown("**Try asking:**")

    suggested = [
        "What is a pin?",
        "Explain forks",
        "How does castling work?",
        "What is a hanging piece?",
        "How do I improve my king safety?",
        "What are good opening principles?"
    ]

    cols = st.columns(3)

    for i, suggestion in enumerate(suggested):
        if cols[i % 3].button(
            suggestion,
            key=f"suggest_{i}",
            use_container_width=True
        ):
            st.session_state.rules_chat_history.append({
                "role": "user",
                "content": suggestion
            })
            st.session_state.rules_thinking = True
            st.rerun()

    st.divider()

    if not st.session_state.rules_chat_history:
        with st.chat_message("assistant"):
            st.write(
                "Hi! I'm your chess rules assistant. Ask me anything about chess — "
                "tactics, rules, openings, endgames, or how to improve specific aspects of your game!"
            )

    for msg in st.session_state.rules_chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.rules_thinking:
        question = st.session_state.rules_chat_history[-1]["content"]

        with st.chat_message("assistant"):
            st.markdown("⏳ *Looking up rules...*")

            try:
                rulebook_content = ""

                search_resp = requests.get(
                    f"{API_URL}/rulebook/search",
                    params={"q": question},
                    timeout=10
                )

                results = []

                if search_resp.status_code == 200:
                    results = search_resp.json().get("results", [])

                if results:
                    entry = results[0]
                    principles = "\n".join(
                        f"- {p}"
                        for p in entry.get("key_principles", [])[:3]
                    )

                    rulebook_content = (
                        f"{entry['title']}: {entry['description']}\n"
                        f"Key principles:\n{principles}"
                    )

            except Exception:
                rulebook_content = ""

            try:
                resp = requests.post(
                    f"{API_URL}/chat/rules",
                    json={
                        "question": question,
                        "rulebook_content": rulebook_content,
                        "history": st.session_state.rules_chat_history[:-1],
                    },
                    timeout=120,
                )

                if resp.status_code == 200:
                    answer = resp.json().get(
                        "response",
                        "Sorry, something went wrong."
                    )
                else:
                    answer = "Sorry, try again."

            except requests.exceptions.ConnectionError:
                answer = "Backend not reachable."

            except Exception as e:
                answer = f"Error: {e}"

            st.session_state.rules_chat_history.append({
                "role": "assistant",
                "content": answer
            })
            st.session_state.rules_thinking = False
            st.rerun()

    if not st.session_state.rules_thinking:
        if question := st.chat_input("Ask about any chess rule or concept..."):
            st.session_state.rules_chat_history.append({
                "role": "user",
                "content": question
            })
            st.session_state.rules_thinking = True
            st.rerun()
    else:
        st.chat_input("Looking up rules... please wait", disabled=True)


# ════════════════════════════════════════════════════════════════
# PAGE 7 — DASHBOARD
# ════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.title("📊 Your Progress")

    try:
        resp = requests.get(
            f"{API_URL}/profile/{st.session_state.user_id}",
            timeout=120
        )

        if resp.status_code == 200:
            profile = resp.json()

            c1, c2, c3 = st.columns(3)
            c1.metric("Games Analyzed", profile["games_played"])
            c2.metric("Estimated Elo", profile["est_elo"])
            c3.metric("Avg CP Loss", f"{profile['avg_cp_loss']:.1f}")

            st.divider()
            st.subheader("Weakness Profile")

            weakness_data = profile.get("weakness_profile", {})

            if weakness_data:
                import pandas as pd

                df = pd.DataFrame(
                    list(weakness_data.items()),
                    columns=["Weakness", "Count"]
                ).sort_values("Count", ascending=False)

                st.bar_chart(df.set_index("Weakness"))

            else:
                st.info("Analyze more games to see your weakness profile.")

        elif resp.status_code == 404:
            st.info("No games analyzed yet. Go to Analyze Game to get started.")

        else:
            st.error("Could not load profile.")

    except Exception:
        st.error("Backend not reachable.")
