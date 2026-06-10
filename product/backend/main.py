from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
sys.path.append(os.path.dirname(__file__))

from analyzer import analyze_pgn
from database import init_db, save_game_analysis, get_player_profile, get_game_analysis
from puzzle import get_puzzles

app = FastAPI(title="ChessRL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    init_db()
    print("ChessRL API ready.")

class PGNRequest(BaseModel):
    pgn: str
    user_id: str = "default_user"

@app.post("/upload_pgn")
def upload_pgn(req: PGNRequest):
    if not req.pgn.strip():
        raise HTTPException(status_code=400, detail="PGN is empty")
    analysis = analyze_pgn(req.pgn)
    if "error" in analysis:
        raise HTTPException(status_code=500, detail=analysis["error"])
    game_id = save_game_analysis(req.user_id, req.pgn, analysis)
    profile = get_player_profile(req.user_id)
    weakness = analysis["summary"]["primary_weakness"]
    est_elo = profile["est_elo"] if profile else 1000
    puzzles = get_puzzles(weakness, est_elo)
    return {
        "game_id": game_id,
        "analysis": analysis,
        "player_profile": profile,
        "puzzles": puzzles,
    }

@app.get("/analysis/{game_id}")
def get_analysis(game_id: int):
    analysis = get_game_analysis(game_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Game not found")
    return analysis

@app.get("/puzzles/{user_id}")
def get_user_puzzles(user_id: str, n: int = 5):
    profile = get_player_profile(user_id)
    if not profile:
        return {"weakness": "general", "user_elo": 1000, "puzzles": get_puzzles("general", 1000, n)}
    return {
        "weakness": profile["primary_weakness"],
        "user_elo": profile["est_elo"],
        "puzzles": get_puzzles(profile["primary_weakness"], profile["est_elo"], n),
    }

@app.get("/profile/{user_id}")
def profile(user_id: str):
    p = get_player_profile(user_id)
    if not p:
        raise HTTPException(status_code=404, detail="No games analyzed yet")
    return p

@app.get("/health")
def health():
    return {"status": "ok"}