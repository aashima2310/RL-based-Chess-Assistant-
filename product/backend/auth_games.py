"""Login, saved games, and database check endpoint."""
import os
import re
import hmac
import json
import time
import base64
import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

from database import get_connection, execute_query, IS_POSTGRES

router = APIRouter()

SECRET = os.getenv("AUTH_SECRET", "dev-secret-change-me").encode()
TOKEN_TTL = 60 * 60 * 24 * 30  # 30 days

_ready = False


def _ensure():
    global _ready

    if _ready:
        return

    conn = get_connection()
    c = conn.cursor()

    pk = (
        "SERIAL PRIMARY KEY"
        if IS_POSTGRES
        else "INTEGER PRIMARY KEY AUTOINCREMENT"
    )

    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS users (
            id {pk},
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT
        )
        """
    )

    c.execute(
        f"""
        CREATE TABLE IF NOT EXISTS played_games (
            id {pk},
            user_id TEXT,
            played_at TEXT,
            pgn TEXT,
            result TEXT,
            difficulty TEXT,
            engine_type TEXT,
            side TEXT,
            num_moves INTEGER
        )
        """
    )

    conn.commit()
    conn.close()

    _ready = True


def _hash(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)

    h = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        salt,
        200_000,
    )

    return salt.hex() + "$" + h.hex()


def _verify(password: str, stored: str) -> bool:
    try:
        salt_hex, _ = stored.split("$")

        return hmac.compare_digest(
            _hash(password, bytes.fromhex(salt_hex)),
            stored,
        )

    except Exception:
        return False


def _make_token(username: str) -> str:
    body = base64.urlsafe_b64encode(
        json.dumps(
            {
                "u": username,
                "exp": int(time.time()) + TOKEN_TTL,
            }
        ).encode()
    ).decode()

    sig = hmac.new(
        SECRET,
        body.encode(),
        hashlib.sha256,
    ).hexdigest()

    return f"{body}.{sig}"


def _user_from(authorization: Optional[str]) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Please log in",
        )

    token = authorization[7:]

    try:
        body, sig = token.split(".")

        good = hmac.new(
            SECRET,
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig, good):
            raise ValueError

        data = json.loads(
            base64.urlsafe_b64decode(body.encode())
        )

        if data["exp"] < time.time():
            raise ValueError

        return data["u"]

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Session expired, please log in again",
        )


class Cred(BaseModel):
    username: str
    password: str


class SaveGame(BaseModel):
    pgn: str
    result: str = ""
    difficulty: str = ""
    engine_type: str = ""
    side: str = "white"
    num_moves: int = 0


def _clean(cred: Cred) -> str:
    u = cred.username.strip().lower()

    if not re.fullmatch(r"[a-z0-9_.-]{3,30}", u):
        raise HTTPException(
            status_code=400,
            detail="Username: 3-30 letters, numbers, _ . -",
        )

    if len(cred.password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters",
        )

    return u


@router.post("/auth/register")
def register(cred: Cred):
    _ensure()

    u = _clean(cred)

    conn = get_connection()
    c = conn.cursor()

    execute_query(
        c,
        "SELECT id FROM users WHERE username = ?",
        (u,),
    )

    if c.fetchone():
        conn.close()

        raise HTTPException(
            status_code=409,
            detail="Username already taken",
        )

    execute_query(
        c,
        """
        INSERT INTO users
            (username, password_hash, created_at)
        VALUES (?, ?, ?)
        """,
        (
            u,
            _hash(cred.password),
            datetime.now().isoformat(),
        ),
    )

    conn.commit()
    conn.close()

    return {
        "username": u,
        "token": _make_token(u),
    }


@router.post("/auth/login")
def login(cred: Cred):
    _ensure()

    u = cred.username.strip().lower()

    conn = get_connection()
    c = conn.cursor()

    execute_query(
        c,
        "SELECT password_hash FROM users WHERE username = ?",
        (u,),
    )

    row = c.fetchone()

    conn.close()

    if not row or not _verify(
        cred.password,
        row["password_hash"],
    ):
        raise HTTPException(
            status_code=401,
            detail="Wrong username or password",
        )

    return {
        "username": u,
        "token": _make_token(u),
    }


@router.post("/games")
def save_played_game(
    g: SaveGame,
    authorization: Optional[str] = Header(None),
):
    """
    Save a completed game played against the engine.

    Returns the actual database game_id so the frontend
    can immediately identify the saved game.
    """

    _ensure()

    u = _user_from(authorization)

    conn = get_connection()
    c = conn.cursor()

    played_at = datetime.now().isoformat()

    insert_query = """
        INSERT INTO played_games
            (
                user_id,
                played_at,
                pgn,
                result,
                difficulty,
                engine_type,
                side,
                num_moves
            )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """

    if IS_POSTGRES:
        # PostgreSQL needs RETURNING to obtain the inserted ID.
        insert_query = insert_query.replace("?", "%s") + " RETURNING id"

        c.execute(
            insert_query,
            (
                u,
                played_at,
                g.pgn,
                g.result,
                g.difficulty,
                g.engine_type,
                g.side,
                g.num_moves,
            ),
        )

        row = c.fetchone()
        game_id = row["id"]

    else:
        # SQLite exposes the inserted ID through lastrowid.
        c.execute(
            insert_query,
            (
                u,
                played_at,
                g.pgn,
                g.result,
                g.difficulty,
                g.engine_type,
                g.side,
                g.num_moves,
            ),
        )

        game_id = c.lastrowid

    conn.commit()
    conn.close()

    return {
        "saved": True,
        "game_id": game_id,
    }


@router.get("/games")
def list_played_games(
    authorization: Optional[str] = Header(None),
):
    """Return the logged-in user's engine games."""

    _ensure()

    u = _user_from(authorization)

    conn = get_connection()
    c = conn.cursor()

    execute_query(
        c,
        """
        SELECT
            id,
            played_at,
            pgn,
            result,
            difficulty,
            engine_type,
            side,
            num_moves
        FROM played_games
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 50
        """,
        (u,),
    )

    rows = [dict(r) for r in c.fetchall()]

    conn.close()

    return {
        "games": rows,
    }


@router.get("/api/dbcheck")
def dbcheck():
    try:
        _ensure()

        conn = get_connection()
        c = conn.cursor()

        c.execute(
            "SELECT COUNT(*) AS n FROM users"
        )

        n = c.fetchone()["n"]

        conn.close()

        return {
            "database": (
                "postgres"
                if IS_POSTGRES
                else "sqlite (temporary, wiped on restart)"
            ),
            "ok": True,
            "users": n,
        }

    except Exception as e:
        return {
            "database": (
                "postgres"
                if IS_POSTGRES
                else "sqlite"
            ),
            "ok": False,
            "error": str(e),
        }
 
