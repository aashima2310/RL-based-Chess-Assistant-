import sqlite3
import json
from datetime import datetime

import os
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chessrl.db")

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS players (
            user_id      TEXT PRIMARY KEY,
            games_played INTEGER DEFAULT 0,
            avg_cp_loss  REAL DEFAULT 0,
            est_elo      INTEGER DEFAULT 800,
            created_at   TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            played_at     TEXT,
            pgn           TEXT,
            blunders      INTEGER,
            mistakes      INTEGER,
            inaccuracies  INTEGER,
            avg_cp_loss   REAL,
            primary_weakness TEXT,
            analysis_json TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS mistakes (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        TEXT,
            game_id        INTEGER,
            move           TEXT,
            mistake_type   TEXT,
            classification TEXT,
            cp_loss        INTEGER
        )
    """)
    conn.commit()
    conn.close()

def cp_loss_to_elo(avg_cp_loss):
    if avg_cp_loss < 20:  return 2200
    if avg_cp_loss < 40:  return 1800
    if avg_cp_loss < 60:  return 1500
    if avg_cp_loss < 80:  return 1200
    if avg_cp_loss < 120: return 1000
    return 800

def save_game_analysis(user_id, pgn, analysis):
    conn = get_connection()
    c = conn.cursor()
    summary = analysis["summary"]
    c.execute("""
        INSERT INTO players (user_id, games_played, created_at)
        VALUES (?, 1, ?)
        ON CONFLICT(user_id) DO UPDATE SET games_played = games_played + 1
    """, (user_id, datetime.now().isoformat()))
    c.execute("""
        INSERT INTO games
            (user_id, played_at, pgn, blunders, mistakes, inaccuracies,
             avg_cp_loss, primary_weakness, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        user_id, datetime.now().isoformat(), pgn,
        summary["blunders"], summary["mistakes"], summary["inaccuracies"],
        summary["avg_cp_loss"], summary["primary_weakness"],
        json.dumps(analysis),
    ))
    game_id = c.lastrowid
    for move in analysis["moves"]:
        if move["mistake_type"]:
            c.execute("""
                INSERT INTO mistakes
                    (user_id, game_id, move, mistake_type, classification, cp_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, game_id, move["move"], move["mistake_type"],
                  move["classification"], move["cp_loss"]))
    est_elo = cp_loss_to_elo(summary["avg_cp_loss"])
    c.execute("""
        UPDATE players SET avg_cp_loss = ?, est_elo = ? WHERE user_id = ?
    """, (summary["avg_cp_loss"], est_elo, user_id))
    conn.commit()
    conn.close()
    return game_id

def get_primary_weakness(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT mistake_type, COUNT(*) as cnt
        FROM mistakes WHERE user_id = ?
        GROUP BY mistake_type ORDER BY cnt DESC LIMIT 1
    """, (user_id,))
    row = c.fetchone()
    conn.close()
    return row["mistake_type"] if row else "general"

def get_player_profile(user_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM players WHERE user_id = ?", (user_id,))
    player = c.fetchone()
    if not player:
        conn.close()
        return None
    c.execute("""
        SELECT mistake_type, COUNT(*) as cnt
        FROM mistakes WHERE user_id = ?
        GROUP BY mistake_type ORDER BY cnt DESC
    """, (user_id,))
    weaknesses = {row["mistake_type"]: row["cnt"] for row in c.fetchall()}
    c.execute("""
        SELECT played_at, blunders, avg_cp_loss, primary_weakness
        FROM games WHERE user_id = ?
        ORDER BY played_at DESC LIMIT 5
    """, (user_id,))
    recent_games = [dict(row) for row in c.fetchall()]
    conn.close()
    return {
        "user_id": user_id,
        "games_played": player["games_played"],
        "est_elo": player["est_elo"],
        "avg_cp_loss": player["avg_cp_loss"],
        "weakness_profile": weaknesses,
        "primary_weakness": get_primary_weakness(user_id),
        "recent_games": recent_games,
    }

def get_game_analysis(game_id):
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT analysis_json FROM games WHERE id = ?", (game_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row["analysis_json"]) if row else None