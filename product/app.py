"""
app.py — Streamlit Frontend
-----------------------------
Covers Pipeline 2A (PGN upload + analysis display)
and Pipeline 3 (puzzle recommendations).

Run with: streamlit run product/app.py
Make sure backend is running: uvicorn backend.main:app --reload --port 8000
"""

import streamlit as st
import requests
import chess
import chess.pgn
import io

API_URL = "http://localhost:8000"

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="ChessRL — Your Personal Chess Coach",
    page_icon="♟",
    layout="wide",
)

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


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.title("♟ ChessRL")
    st.caption("RL-Powered Chess Coach")
    st.divider()

    page = st.radio(
        "Navigate",
        ["📤 Analyze Game", "🧩 Puzzles", "💬 Coach Chat", "📊 Dashboard"],
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
                    st.success("✅ Analysis complete!")
                    st.rerun()
                else:
                    st.error(f"Analysis failed: {resp.json().get('detail', 'Unknown error')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Backend not running. Start it with: `uvicorn backend.main:app --reload`")

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
                    import base64
                    b64 = base64.b64encode(svg.encode()).decode()
                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" width="380"/>',
                        unsafe_allow_html=True)
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
                    import base64
                    b64 = base64.b64encode(svg.encode()).decode()
                    st.markdown(
                        f'<img src="data:image/svg+xml;base64,{b64}" width="320"/>',
                        unsafe_allow_html=True)
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
# PAGE 3 — PIPELINE 4: CHATBOT
# ════════════════════════════════════════════════════════════════

elif page == "💬 Coach Chat":
    st.title("💬 Ask Your Coach")

    if not st.session_state.game_id:
        st.info("Analyze a game first, then come back here to ask questions about it.")
        st.stop()

    # Opening message
    if st.session_state.chatbot_opening:
        with st.chat_message("assistant"):
            st.write(st.session_state.chatbot_opening)

    # Chat history
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Input
    if question := st.chat_input("Ask about any move...  e.g. 'Why was move 15 bad?'"):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(f"{API_URL}/chat", json={
                        "question": question,
                        "game_id": st.session_state.game_id,
                        "user_id": st.session_state.user_id,
                        "history": st.session_state.chat_history[:-1],
                    }, timeout=60)
                    answer = resp.json().get("response", "Sorry, something went wrong.")
                except Exception:
                    answer = "Backend not reachable. Make sure uvicorn is running."
                st.write(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})


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
