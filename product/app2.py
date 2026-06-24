"""
app.py — Streamlit Frontend
-----------------------------
Covers Pipeline 2A (PGN upload + analysis display),
Pipeline 2B (Play Against Engine),
Pipeline 3 (puzzle recommendations),
and Pipeline 4 frontend wiring (Coach Chat UI + context passing).

Run with: streamlit run product/app.py
Make sure backend is running: python -m uvicorn backend.main:app --reload --port 8000
"""

import streamlit as st
import requests
import chess
import chess.pgn
import chess.svg
import io
import base64

API_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChessRL — Your Personal Chess Coach",
    page_icon="♟",
    layout="wide",
)

# ── Helper function for Pipeline 2B chat context ─────────────
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
    }


# ── Session state init ───────────────────────────────────────
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

# ── Pipeline 2B session state ────────────────────────────────
if "play_fen" not in st.session_state:
    st.session_state.play_fen = chess.Board().fen()
if "play_history" not in st.session_state:
    st.session_state.play_history = []
if "play_difficulty" not in st.session_state:
    st.session_state.play_difficulty = "Easy"

# ── Pipeline 4 frontend context state ────────────────────────
if "chat_context_source" not in st.session_state:
    st.session_state.chat_context_source = None

if "chat_play_context" not in st.session_state:
    st.session_state.chat_play_context = None


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("♟ ChessRL")
    st.caption("RL-Powered Chess Coach")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📤 Analyze Game", "♟ Play Engine", "🧩 Puzzles", "💬 Coach Chat", "📊 Dashboard"],
        label_visibility="collapsed"
    )

    st.divider()
    st.session_state.user_id = st.text_input("User ID", value=st.session_state.user_id)
    st.caption(f"Logged in as: `{st.session_state.user_id}`")


# ════════════════════════════════════════════════════════════════
# PAGE 1 — PIPELINE 2A: PGN UPLOAD + ANALYSIS
# ════════════════════════════════════════════════════════════════

if page == "📤 Analyze Game":
    st.title("📤 Analyze Your Game")
    st.caption("Upload a PGN to get move-by-move analysis, mistake classification, and coaching feedback.")

    col1, col2 = st.columns([1, 1])

    with col1:
        # ── PGN Input ────────────────────────────────────────
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

        analyze_btn = st.button("🔍 Analyze Game", type="primary", use_container_width=True)

    # ── Run Analysis ─────────────────────────────────────────
    if analyze_btn and pgn_input.strip():
        with st.spinner("Analyzing your game... (this takes 10–30s depending on game length)"):
            try:
                resp = requests.post(f"{API_URL}/upload_pgn", json={
                    "pgn": pgn_input,
                    "user_id": st.session_state.user_id
                }, timeout=120)

                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state.analysis = data["analysis"]
                    st.session_state.game_id = data["game_id"]
                    st.session_state.puzzles = data.get("puzzles", [])
                    st.session_state.chatbot_opening = data.get("chatbot_opening")
                    st.session_state.chat_history = []
                    st.session_state.current_move_idx = 0

                    # Pipeline 4 context source becomes PGN analysis
                    st.session_state.chat_context_source = "pgn_analysis"

                    st.success("✅ Analysis complete!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {resp.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Start it with: `python -m uvicorn backend.main:app --reload --port 8000`")

    elif analyze_btn:
        st.warning("Please paste a PGN first.")

    # ── Display Analysis Results ─────────────────────────────
    if st.session_state.analysis:
        analysis = st.session_state.analysis
        summary = analysis["summary"]
        moves = analysis["moves"]

        st.divider()

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Blunders", summary["blunders"])
        m2.metric("🟠 Mistakes", summary["mistakes"])
        m3.metric("🟡 Inaccuracies", summary["inaccuracies"])
        m4.metric("📉 Avg CP Loss", f"{summary['avg_cp_loss']:.1f}")

        st.divider()

        col_board, col_moves = st.columns([1, 1])

        with col_board:
            st.subheader("Board Position")

            # Navigation
            nav1, nav2, nav3, nav4 = st.columns(4)
            with nav1:
                if st.button("⏮ Start"):
                    st.session_state.current_move_idx = 0
            with nav2:
                if st.button("◀ Prev") and st.session_state.current_move_idx > 0:
                    st.session_state.current_move_idx -= 1
            with nav3:
                if st.button("Next ▶") and st.session_state.current_move_idx < len(moves) - 1:
                    st.session_state.current_move_idx += 1
            with nav4:
                if st.button("End ⏭"):
                    st.session_state.current_move_idx = len(moves) - 1

            # Render board as SVG
            current_idx = st.session_state.current_move_idx
            if current_idx < len(moves):
                current_move = moves[current_idx]
                try:
                    board = chess.Board(current_move["fen_before"])
                    last_move = chess.Move.from_uci(current_move["move_uci"]) if current_move.get("move_uci") else None
                    svg = chess.svg.board(board, lastmove=last_move, size=380)
                    b64 = base64.b64encode(svg.encode()).decode()
                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" width="380"/>',
                        unsafe_allow_html=True
                    )
                except Exception as e:
                    st.error(f"Board error: {e}")

            # Current move info box
            if current_idx < len(moves):
                m = moves[current_idx]
                color = {
                    "Blunder": "🔴",
                    "Mistake": "🟠",
                    "Inaccuracy": "🟡",
                    "Good": "🟢"
                }.get(m["classification"], "⚪")

                st.info(
                    f"**Move {m['move_number']}: {m['move']}** {color} {m['classification']}\n\n"
                    f"CP Loss: **{m['cp_loss']}** | Best: `{m['best_move']}`\n\n"
                    f"Type: `{m['mistake_type'] or 'n/a'}`"
                )

        with col_moves:
            st.subheader("Move List")

            # Color-coded move list
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

        # Weakness summary
        st.divider()
        st.subheader("🎯 Weakness Summary")
        weakness_cols = st.columns(len(summary["mistake_counts"]))
        for col, (wtype, count) in zip(weakness_cols, summary["mistake_counts"].items()):
            if count > 0:
                col.metric(wtype.replace("_", " ").title(), count)

        primary = summary["primary_weakness"].replace("_", " ").title()
        st.warning(f"**Primary weakness this game:** {primary} — see Puzzles tab for targeted training")


# ════════════════════════════════════════════════════════════════
# PAGE 2B — USER PLAYS AGAINST ENGINE
# ════════════════════════════════════════════════════════════════

elif page == "♟ Play Engine":
    st.title("♟ Play Against Engine")
    st.caption("Play a live game against the engine. Select difficulty, enter your move, and the engine will reply.")

    col_board, col_controls = st.columns([1, 1])

    with col_board:
        st.subheader("Chess Board")

        try:
            board = chess.Board(st.session_state.play_fen)

            svg = chess.svg.board(board, size=420)
            b64 = base64.b64encode(svg.encode()).decode()

            st.markdown(
                f'<img src="data:image/svg+xml;base64,{b64}" width="420"/>',
                unsafe_allow_html=True
            )

            st.caption(f"Current turn: **{'White' if board.turn == chess.WHITE else 'Black'}**")

            if board.is_game_over():
                st.success(f"Game over. Result: {board.result()}")

        except Exception as e:
            st.error(f"Board error: {e}")

    with col_controls:
        st.subheader("Controls")

        difficulty = st.selectbox(
            "Select Difficulty",
            ["Easy", "Medium", "Hard"],
            index=["Easy", "Medium", "Hard"].index(st.session_state.play_difficulty)
        )

        st.session_state.play_difficulty = difficulty

        st.markdown("Enter your move in **UCI format**.")
        st.caption("Examples: `e2e4`, `g1f3`, `e7e8q`")

        user_move = st.text_input(
            "Your move",
            placeholder="e2e4"
        )

        play_btn = st.button(
            "▶ Play Move",
            type="primary",
            use_container_width=True
        )

        reset_btn = st.button(
            "🔄 New Game",
            use_container_width=True
        )

        if reset_btn:
            st.session_state.play_fen = chess.Board().fen()
            st.session_state.play_history = []
            st.session_state.chat_play_context = None
            st.rerun()

        if play_btn:
            if not user_move.strip():
                st.warning("Please enter a move first.")
            else:
                try:
                    resp = requests.post(
                        f"{API_URL}/engine_move",
                        json={
                            "fen": st.session_state.play_fen,
                            "move": user_move.strip(),
                            "difficulty": difficulty.lower()
                        },
                        timeout=120
                    )

                    if resp.status_code == 200:
                        data = resp.json()

                        if data.get("success"):
                            engine_move = data.get("engine_move")
                            st.session_state.play_fen = data.get("fen")

                            st.session_state.play_history.append({
                                "user_move": user_move.strip(),
                                "engine_move": engine_move
                            })

                            # Update Play Engine context for chatbot
                            st.session_state.chat_play_context = build_play_engine_context()

                            if engine_move is None:
                                st.success(f"You played `{user_move.strip()}`. Game ended.")
                            else:
                                st.success(f"You played `{user_move.strip()}`. Engine replied `{engine_move}`.")

                            st.rerun()

                        else:
                            st.error(data.get("message", "Illegal move."))

                    else:
                        st.error("Backend returned an error.")

                except requests.exceptions.ConnectionError:
                    st.error("Backend not running. Start it with: `python -m uvicorn backend.main:app --reload --port 8000`")

                except Exception as e:
                    st.error(f"Error: {e}")

    st.divider()
    st.subheader("Move History")

    if not st.session_state.play_history:
        st.info("No moves played yet.")
    else:
        for i, item in enumerate(st.session_state.play_history, start=1):
            engine_text = item["engine_move"] if item["engine_move"] is not None else "Game Over"
            st.write(
                f"**{i}.** You: `{item['user_move']}`  |  Engine: `{engine_text}`"
            )

        st.divider()
        st.subheader("💬 Send this game to Coach Chat")

        st.caption(
            "This sends the Play Engine game history as chatbot context. "
            "The backend chatbot can later use this to answer questions about this played game."
        )

        if st.button("Use this Play Engine game in Coach Chat", use_container_width=True):
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
# PAGE 2 — PIPELINE 3: PUZZLE RECOMMENDATIONS
# ════════════════════════════════════════════════════════════════

elif page == "🧩 Puzzles":
    st.title("🧩 Recommended Puzzles")

    if not st.session_state.puzzles:
        # Try to fetch from API
        try:
            resp = requests.get(
                f"{API_URL}/puzzles/{st.session_state.user_id}",
                timeout=10
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
    weakness = puzzles[0].get("theme", "").replace("_", " ").title() if puzzles else ""

    st.caption(f"📌 These puzzles target your weakness: **{weakness}**")
    st.divider()

    for i, puzzle in enumerate(puzzles):
        with st.expander(f"Puzzle {i+1}  |  Rating: {puzzle['rating']}  |  Theme: {puzzle['theme']}", expanded=(i == 0)):
            p_col1, p_col2 = st.columns([1, 1])

            with p_col1:
                # Render puzzle position
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

                # Solution reveal
                if st.button(f"👁 Reveal Solution", key=f"solution_{i}"):
                    solution_moves = " → ".join(puzzle.get("solution", []))
                    st.success(f"Solution: `{solution_moves}`")

                if puzzle.get("url"):
                    st.markdown(f"[Play on Lichess ↗]({puzzle['url']})")

    # Refresh puzzles button
    st.divider()
    if st.button("🔄 Get New Puzzles", use_container_width=True):
        try:
            resp = requests.get(
                f"{API_URL}/puzzles/{st.session_state.user_id}",
                timeout=10
            )
            if resp.status_code == 200:
                st.session_state.puzzles = resp.json().get("puzzles", [])
                st.rerun()
        except Exception:
            st.error("Backend not reachable.")


# ════════════════════════════════════════════════════════════════
# PAGE 3 — PIPELINE 4: CHATBOT FRONTEND
# ════════════════════════════════════════════════════════════════

elif page == "💬 Coach Chat":
    st.title("💬 Ask Your Coach")
    st.caption("Chat with the coach using either PGN analysis context or Play Engine context.")

    has_pgn_context = st.session_state.game_id is not None and st.session_state.analysis is not None
    has_play_context = st.session_state.chat_play_context is not None or len(st.session_state.play_history) > 0

    if not has_pgn_context and not has_play_context:
        st.info(
            "No chat context found yet. Analyze a PGN game or play against the engine first."
        )
        st.stop()

    # Auto-select context if none selected
    if st.session_state.chat_context_source is None:
        if has_pgn_context:
            st.session_state.chat_context_source = "pgn_analysis"
        elif has_play_context:
            st.session_state.chat_context_source = "play_engine"

    # Context switcher
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
            st.button("Use PGN Analysis Context", disabled=True, use_container_width=True)

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
            st.button("Use Play Engine Context", disabled=True, use_container_width=True)

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

    # Opening message
    if st.session_state.chatbot_opening:
        with st.chat_message("assistant"):
            st.write(st.session_state.chatbot_opening)
    else:
        with st.chat_message("assistant"):
            st.write(
                "Hi, I am your chess coach. Ask me about your moves, mistakes, blunders, or the current game."
            )

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if question := st.chat_input("Ask about any move... e.g. 'Why was move 15 bad?'"):
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
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

                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json=payload,
                        timeout=60
                    )

                    if resp.status_code == 200:
                        answer = resp.json().get("response", "Sorry, something went wrong.")
                    else:
                        answer = (
                            "Chatbot backend is not ready yet, or it returned an error. "
                            "Frontend context wiring is working and the question/context was sent."
                        )

                except requests.exceptions.ConnectionError:
                    answer = (
                        "Chatbot backend is not reachable. Make sure FastAPI is running. "
                        "Frontend chat UI and context wiring are ready."
                    )

                except Exception as e:
                    answer = f"Chatbot error: {e}"

                st.write(answer)

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })


# ════════════════════════════════════════════════════════════════
# PAGE 4 — DASHBOARD
# ════════════════════════════════════════════════════════════════

elif page == "📊 Dashboard":
    st.title("📊 Your Progress")

    try:
        resp = requests.get(f"{API_URL}/profile/{st.session_state.user_id}", timeout=10)
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