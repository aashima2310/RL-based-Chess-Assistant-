import os
import json
from datetime import datetime

DATABASE_URL = os.getenv("DATABASE_URL")
IS_POSTGRES = DATABASE_URL and DATABASE_URL.startswith("postgres")

if IS_POSTGRES:
    import psycopg2
    from psycopg2.extras import RealDictCursor
else:
    import sqlite3
    DB_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(DB_DIR, exist_ok=True)
    DB_PATH = os.path.join(DB_DIR, "chessrl.db")

def get_connection():
    if IS_POSTGRES:
        return psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn

def execute_query(cursor, query, params=()):
    if IS_POSTGRES:
        query = query.replace("?", "%s")
    cursor.execute(query, params)

def init_db():
    conn = get_connection()
    c = conn.cursor()
    
    id_type = "SERIAL" if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"
    
    c.execute("""
        CREATE TABLE IF NOT EXISTS players (
            user_id      TEXT PRIMARY KEY,
            games_played INTEGER DEFAULT 0,
            avg_cp_loss  REAL DEFAULT 0,
            est_elo      INTEGER DEFAULT 800,
            created_at   TEXT
        )
    """)
    
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS games (
            id               {id_type if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
            user_id          TEXT,
            played_at        TEXT,
            pgn              TEXT,
            blunders         INTEGER,
            mistakes         INTEGER,
            inaccuracies     INTEGER,
            avg_cp_loss      REAL,
            primary_weakness TEXT,
            analysis_json    TEXT
        )
    """)
    
    c.execute(f"""
        CREATE TABLE IF NOT EXISTS mistakes (
            id             {id_type if IS_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"},
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
    
    execute_query(c, """
        INSERT INTO players (user_id, games_played, created_at)
        VALUES (?, 1, ?)
        ON CONFLICT(user_id) DO UPDATE SET games_played = players.games_played + 1
    """, (user_id, datetime.now().isoformat()))
    
    insert_game_query = """
        INSERT INTO games
            (user_id, played_at, pgn, blunders, mistakes, inaccuracies,
             avg_cp_loss, primary_weakness, analysis_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    if IS_POSTGRES:
        insert_game_query = insert_game_query.replace("?", "%s") + " RETURNING id"
        c.execute(insert_game_query, (
            user_id, datetime.now().isoformat(), pgn,
            summary["blunders"], summary["mistakes"], summary["inaccuracies"],
            summary["avg_cp_loss"], summary["primary_weakness"],
            json.dumps(analysis),
        ))
        game_id = c.fetchone()["id"]
    else:
        c.execute(insert_game_query, (
            user_id, datetime.now().isoformat(), pgn,
            summary["blunders"], summary["mistakes"], summary["inaccuracies"],
            summary["avg_cp_loss"], summary["primary_weakness"],
            json.dumps(analysis),
        ))
        game_id = c.lastrowid

    for move in analysis["moves"]:
        if move["mistake_type"]:
            execute_query(c, """
                INSERT INTO mistakes
                    (user_id, game_id, move, mistake_type, classification, cp_loss)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, game_id, move["move"], move["mistake_type"],
                  move["classification"], move["cp_loss"]))
            
    est_elo = cp_loss_to_elo(summary["avg_cp_loss"])
    execute_query(c, """
        UPDATE players SET avg_cp_loss = ?, est_elo = ? WHERE user_id = ?
    """, (summary["avg_cp_loss"], est_elo, user_id))
    
    conn.commit()
    conn.close()
    return game_id

def get_primary_weakness(user_id):
    conn = get_connection()
    c = conn.cursor()
    execute_query(c, """
        SELECT mistake_type, COUNT(*) as cnt
        FROM mistakes WHERE user_id = ?
        AND mistake_type != 'general'
        GROUP BY mistake_type ORDER BY cnt DESC LIMIT 1
    """, (user_id,))
    row = c.fetchone()
    
    if not row:
        execute_query(c, """
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
    execute_query(c, "SELECT * FROM players WHERE user_id = ?", (user_id,))
    player = c.fetchone()
    if not player:
        conn.close()
        return None
        
    execute_query(c, """
        SELECT mistake_type, COUNT(*) as cnt
        FROM mistakes WHERE user_id = ?
        GROUP BY mistake_type ORDER BY cnt DESC
    """, (user_id,))
    weaknesses = {row["mistake_type"]: row["cnt"] for row in c.fetchall()}
    
    execute_query(c, """
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
    execute_query(c, "SELECT analysis_json FROM games WHERE id = ?", (game_id,))
    row = c.fetchone()
    conn.close()
    return json.loads(row["analysis_json"]) if row else None

def get_recent_mistakes(user_id: str, limit: int = 10) -> list:
    conn = get_connection()
    c = conn.cursor()
    execute_query(c, """
        SELECT m.move, m.mistake_type, m.classification, m.cp_loss,
               m.game_id, g.played_at
        FROM mistakes m
        JOIN games g ON m.game_id = g.id
        WHERE m.user_id = ?
        ORDER BY g.played_at DESC, m.id DESC
        LIMIT ?
    """, (user_id, limit))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_improvement_trend(user_id: str, last_n: int = 5) -> list:
    conn = get_connection()
    c = conn.cursor()
    execute_query(c, """
        SELECT played_at, blunders, mistakes, avg_cp_loss, primary_weakness
        FROM games WHERE user_id = ?
        ORDER BY played_at DESC LIMIT ?
    """, (user_id, last_n))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return list(reversed(rows))

def get_common_patterns(user_id: str) -> dict:
    conn = get_connection()
    c = conn.cursor()
    execute_query(c, """
        SELECT mistake_type, COUNT(*) as cnt
        FROM mistakes WHERE user_id = ?
        GROUP BY mistake_type ORDER BY cnt DESC
    """, (user_id,))
    rows = c.fetchall()
    conn.close()
    return {row["mistake_type"]: row["cnt"] for row in rows}
