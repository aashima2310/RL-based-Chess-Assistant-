import asyncio
import os
import sys
import time
from typing import Optional

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(__file__))

from database import (
    init_db,
    save_game_analysis,
    get_player_profile,
    get_game_analysis,
    get_connection,
    execute_query,
    get_recent_mistakes,
    get_improvement_trend,
    get_common_patterns,
)
from analyzer import analyze_pgn
from puzzle import get_puzzles, preload_puzzles
from chatbot import get_opening_message, answer_question
from game_manager import GameManager
from rulebook import (
    get_all_entries,
    get_entry,
    get_categories,
    get_relevant_entries,
    search_rulebook,
)

app = FastAPI(
    title="ChessRL API",
    version="2.0.0",
    description="API layer for the ChessRL frontend. Existing chess/AI modules remain unchanged.",
)

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in cors_origins],
    allow_methods=["*"],
    allow_headers=["*"],
)


from auth_games import router as auth_router
app.include_router(auth_router)


@app.on_event("startup")
def startup():
    init_db()
    preload_puzzles()
    print("ChessRL API ready.")


# -----------------------------
# Request models
# -----------------------------

class PGNRequest(BaseModel):
    pgn: str
    user_id: str = "default_user"
    personality: str = "friendly"


class EngineMoveRequest(BaseModel):
    fen: str
    move: str
    difficulty: str = "easy"
    engine_type: str = "custom"


class RulesChatRequest(BaseModel):
    question: str
    rulebook_content: str = ""
    history: list = Field(default_factory=list)


class ChatRequest(BaseModel):
    question: str
    game_id: Optional[int] = None
    user_id: str = "default_user"
    history: list = Field(default_factory=list)
    context_source: str = "pgn_analysis"
    play_context: Optional[dict] = None
    personality: str = "encouraging"


class NewGameRequest(BaseModel):
    difficulty: str = "easy"
    engine_type: str = "custom"
    fen: Optional[str] = None


# -----------------------------
# Health / metadata
# -----------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "chessrl-api", "version": "2.0.0"}


@app.get("/api/health")
def api_health():
    return health()


@app.get("/api/config")
def config():
    return {
        "difficulties": ["easy", "medium", "hard"],
        "engine_types": ["custom", "stockfish"],
        "features": [
            "play",
            "pgn_analysis",
            "puzzles",
            "coach_chat",
            "rulebook",
            "profile",
            "history",
        ],
    }


# -----------------------------
# Game / AI play
# Existing /engine_move contract is preserved.
# -----------------------------

@app.post("/engine_move")
def engine_move(req: EngineMoveRequest):
    try:
        game = GameManager(
            difficulty=req.difficulty,
            fen=req.fen,
            engine_type=req.engine_type,
        )

        user_result = game.play_user_move(req.move)

        if not user_result["success"]:
            return {
                "success": False,
                "message": user_result["message"],
                "engine_move": None,
                "fen": req.fen,
                "game_over": False,
                "result": None,
            }

        # Do not ask the engine for another move if the human just ended the game.
        if game.is_game_over():
            return {
                "success": True,
                "message": "Game ended.",
                "engine_move": None,
                "fen": game.get_fen(),
                "game_over": True,
                "result": game.get_result(),
            }

        engine_reply = game.play_engine_move()

        return {
            "success": True,
            "message": "Move played successfully",
            "engine_move": engine_reply,
            "fen": game.get_fen(),
            "game_over": game.is_game_over(),
            "result": game.get_result(),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Engine error: {exc}")


@app.post("/api/game/move")
def api_engine_move(req: EngineMoveRequest):
    return engine_move(req)


@app.post("/api/game/new")
def new_game(req: NewGameRequest):
    # The frontend can start from a supplied FEN for analysis/replay.
    # The first AI move is intentionally not made here; the UI decides who moves first.
    return {
        "success": True,
        "fen": req.fen,
        "difficulty": req.difficulty,
        "engine_type": req.engine_type,
        "message": "Game ready",
    }


# -----------------------------
# PGN analysis
# -----------------------------

@app.post("/upload_pgn")
def upload_pgn(req: PGNRequest):
    if not req.pgn.strip():
        raise HTTPException(status_code=400, detail="PGN is empty")

    try:
        t0 = time.time()
        analysis = analyze_pgn(req.pgn)
        print(f"[TIMING] analyze_pgn: {time.time() - t0:.1f}s")

        if "error" in analysis:
            raise HTTPException(status_code=500, detail=analysis["error"])

        t1 = time.time()
        game_id = save_game_analysis(req.user_id, req.pgn, analysis)
        profile = get_player_profile(req.user_id)
        print(f"[TIMING] db save+fetch: {time.time() - t1:.1f}s")

        games_played = profile["games_played"] if profile else 1
        est_elo = profile["est_elo"] if profile else 1000

        puzzles = (
            get_puzzles(profile["primary_weakness"], est_elo)
            if profile and games_played >= 3
            else []
        )

        chatbot_opening = get_opening_message(
            analysis,
            profile or {},
            personality=req.personality,
        )

        return {
            "game_id": game_id,
            "analysis": analysis,
            "player_profile": profile,
            "puzzles": puzzles,
            "puzzles_unlocked": games_played >= 3,
            "games_until_puzzles": max(0, 3 - games_played),
            "chatbot_opening": chatbot_opening,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")


@app.post("/api/analysis")
def api_upload_pgn(req: PGNRequest):
    return upload_pgn(req)


@app.get("/analysis/{game_id}")
def get_analysis(game_id: int):
    analysis = get_game_analysis(game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game not found")
    return analysis


@app.get("/api/analysis/{game_id}")
def api_get_analysis(game_id: int):
    return get_analysis(game_id)


# -----------------------------
# Puzzles
# -----------------------------

@app.get("/puzzles/{user_id}")
def get_user_puzzles(user_id: str, n: int = 5):
    profile = get_player_profile(user_id)

    if not profile:
        return {
            "weakness": "general",
            "user_elo": 1000,
            "puzzles": get_puzzles("general", 1000, n),
        }

    return {
        "weakness": profile["primary_weakness"],
        "user_elo": profile["est_elo"],
        "puzzles": get_puzzles(
            profile["primary_weakness"],
            profile["est_elo"],
            n,
        ),
    }


@app.get("/api/puzzles/{user_id}")
def api_get_user_puzzles(user_id: str, n: int = 5):
    return get_user_puzzles(user_id, n)


# -----------------------------
# Profile / history
# -----------------------------

@app.get("/profile/{user_id}")
def profile(user_id: str):
    p = get_player_profile(user_id)
    if not p:
        return {
            "user_id": user_id,
            "games_played": 0,
            "est_elo": 800,
            "avg_cp_loss": 0,
            "weakness_profile": {},
            "primary_weakness": "general",
            "recent_games": [],
        }
    return p


@app.get("/api/profile/{user_id}")
def api_profile(user_id: str):
    return profile(user_id)


@app.get("/profile/{user_id}/history")
def profile_history(user_id: str):
    conn = get_connection()
    c = conn.cursor()

    execute_query(c, """
        SELECT played_at, blunders, mistakes, inaccuracies,
               avg_cp_loss, primary_weakness,
               CASE
                   WHEN avg_cp_loss < 20  THEN 2200
                   WHEN avg_cp_loss < 40  THEN 1800
                   WHEN avg_cp_loss < 60  THEN 1500
                   WHEN avg_cp_loss < 80  THEN 1200
                   WHEN avg_cp_loss < 120 THEN 1000
                   ELSE 800
               END as est_elo
        FROM games
        WHERE user_id = ?
        ORDER BY played_at ASC
    """, (user_id,))

    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return {"history": rows}


@app.get("/api/profile/{user_id}/history")
def api_profile_history(user_id: str):
    return profile_history(user_id)


@app.get("/api/profile/{user_id}/insights")
def profile_insights(user_id: str):
    return {
        "recent_mistakes": get_recent_mistakes(user_id, 10),
        "improvement_trend": get_improvement_trend(user_id, 10),
        "common_patterns": get_common_patterns(user_id),
    }


# -----------------------------
# Coach / chatbot
# -----------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    if req.context_source == "play_engine":
        if not req.play_context:
            raise HTTPException(
                status_code=400,
                detail="play_context required for play_engine context",
            )

        response = answer_question(
            question=req.question,
            game_analysis={},
            player_profile={},
            chat_history=req.history,
            play_context=req.play_context,
            user_id=req.user_id,
            personality=req.personality,
        )
        return {"response": response}

    if req.game_id is None:
        raise HTTPException(
            status_code=400,
            detail="game_id is required for pgn_analysis context",
        )

    analysis = get_game_analysis(req.game_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail="Game not found. Analyze a game first.",
        )

    profile = get_player_profile(req.user_id) or {}

    response = answer_question(
        question=req.question,
        game_analysis=analysis,
        player_profile=profile,
        chat_history=req.history,
        play_context=None,
        user_id=req.user_id,
        personality=req.personality,
    )

    return {"response": response}


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    return chat(req)


@app.get("/chat/opening/{game_id}")
def chat_opening(game_id: int, user_id: str = "default_user"):
    analysis = get_game_analysis(game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game not found")

    profile = get_player_profile(user_id) or {}
    message = get_opening_message(analysis, profile)
    return {"opening_message": message}


@app.post("/chat/rules")
def chat_rules(req: RulesChatRequest):
    from chatbot import call_ollama_fast

    rulebook_section = req.rulebook_content.strip()

    if not rulebook_section:
        results = search_rulebook(req.question)
        if results:
            entry = results[0]
            principles = "\n".join(
                f"- {p}" for p in entry["key_principles"][:3]
            )
            rulebook_section = (
                f"RULEBOOK: {entry['title']}\n"
                f"{entry['description']}\n\n"
                f"Key principles:\n{principles}"
            )

    history_text = ""
    for msg in req.history[-6:]:
        role = "User" if msg["role"] == "user" else "Coach"
        history_text += f"{role}: {msg['content']}\n"

    ref = f"\nREFERENCE:\n{rulebook_section}\n" if rulebook_section else ""

    prompt = f"""You are a chess teacher. Be clear and encouraging. Max 120 words. No ASCII boards.
{ref}
{history_text}
User: {req.question}
Teacher:"""

    return {"response": call_ollama_fast(prompt)}


@app.post("/api/chat/rules")
def api_chat_rules(req: RulesChatRequest):
    return chat_rules(req)


# -----------------------------
# Rulebook
# -----------------------------

@app.get("/rulebook")
def rulebook_all():
    return {
        "entries": get_all_entries(),
        "categories": get_categories(),
    }


@app.get("/api/rulebook")
def api_rulebook_all():
    return rulebook_all()


@app.get("/rulebook/search")
def rulebook_search(q: str):
    return {"results": search_rulebook(q)}


@app.get("/api/rulebook/search")
def api_rulebook_search(q: str):
    return rulebook_search(q)


@app.get("/rulebook/relevant/{user_id}")
def rulebook_relevant(user_id: str):
    profile = get_player_profile(user_id)
    weakness = profile["primary_weakness"] if profile else "general"
    return {
        "weakness": weakness,
        "entries": get_relevant_entries(weakness),
    }


@app.get("/api/rulebook/relevant/{user_id}")
def api_rulebook_relevant(user_id: str):
    return rulebook_relevant(user_id)


@app.get("/rulebook/{entry_id}")
def rulebook_entry(entry_id: str):
    entry = get_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Entry not found")
    return entry


@app.get("/api/rulebook/{entry_id}")
def api_rulebook_entry(entry_id: str):
    return rulebook_entry(entry_id)
