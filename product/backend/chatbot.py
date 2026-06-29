import re
import chess
import requests
from typing import Optional

from rulebook import search_rulebook

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
MAX_RETRIES = 3

INTENT_MOVE    = "move_question"
INTENT_HISTORY = "history_question"
INTENT_PATTERN = "pattern_question"
INTENT_CONCEPT = "concept_question"
INTENT_GENERAL = "general_question"

_MOVE_KEYWORDS    = ["move", "why", "what should", "better", "blunder", "mistake",
                     "played", "instead", "best move", "inaccuracy", "cp loss"]
_HISTORY_KEYWORDS = ["improving", "improve", "progress", "trend", "getting better",
                     "previous game", "last game", "past", "history", "before"]
_PATTERN_KEYWORDS = ["always", "keep", "repeat", "pattern", "weakness", "weak",
                     "common", "often", "habit", "same mistake", "what do i"]
_CONCEPT_KEYWORDS = ["what is", "what's", "explain", "how does", "tell me about",
                     "define", "meaning", "how to", "teach", "describe"]
_CHESS_TERMS      = ["pin", "fork", "skewer", "hanging", "castle", "castling",
                     "en passant", "check", "checkmate", "stalemate", "opposition",
                     "endgame", "opening", "development", "center", "pawn", "tactic",
                     "fianchetto", "zugzwang", "tempo", "gambit", "sacrifice"]


def detect_intent(question: str) -> str:
    q = question.lower()
    if any(kw in q for kw in _CONCEPT_KEYWORDS) and any(t in q for t in _CHESS_TERMS):
        return INTENT_CONCEPT
    if any(kw in q for kw in _HISTORY_KEYWORDS):
        return INTENT_HISTORY
    if any(kw in q for kw in _PATTERN_KEYWORDS):
        return INTENT_PATTERN
    if any(kw in q for kw in _MOVE_KEYWORDS):
        return INTENT_MOVE
    return INTENT_GENERAL


def _call_ollama(prompt: str, max_tokens: int = 300) -> str:
    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": max_tokens},
        }, timeout=90)
        return resp.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"ERROR: {e}"


def call_ollama_fast(prompt: str) -> str:
    return _call_ollama(prompt, max_tokens=180)


def _extract_piece_moves(text: str) -> list:
    pattern = r'\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)\b'
    return re.findall(pattern, text)


def _validate_no_hallucination(response: str, allowed_moves: set) -> bool:
    if not allowed_moves:
        return True
    normalized_allowed = {m.rstrip('+#') for m in allowed_moves}
    for move in _extract_piece_moves(response):
        if move.rstrip('+#') not in normalized_allowed:
            return False
    return True


def _build_allowed_moves(moves_data: list) -> set:
    allowed = set()
    for m in moves_data:
        if m.get("move"):
            allowed.add(m["move"])
        if m.get("best_move"):
            allowed.add(m["best_move"])
        for alt in m.get("alternatives", []):
            allowed.add(alt)
        if m.get("move_uci"):
            allowed.add(m["move_uci"])
    return allowed


_SYSTEM_HEADER = """You are ChessRL, a personalized chess coach.
Use ONLY the verified information provided in the sections below.

Rules you MUST follow:
  - Never invent statistics, percentages, or trends not present in the data.
  - Never invent previous games or moves not listed here.
  - Never infer improvement unless you have at least 2 data points to compare.
  - If the information needed is missing, say: "I don't have enough data to answer that accurately."
  - Keep answers under 150 words. No ASCII boards.
"""


def _section(title: str, content: str) -> str:
    return f"\n=== {title} ===\n{content}\n{'=' * (len(title) + 8)}\n"


def _build_game_section(game_analysis: dict) -> str:
    summary = game_analysis.get("summary", {})
    moves = game_analysis.get("moves", [])
    bad_moves = [m for m in moves if m["classification"] in ("Blunder", "Mistake")]

    lines = [
        f"Total moves: {summary.get('total_moves', '?')}",
        f"Blunders: {summary.get('blunders', 0)}",
        f"Mistakes: {summary.get('mistakes', 0)}",
        f"Inaccuracies: {summary.get('inaccuracies', 0)}",
        f"Avg centipawn loss: {summary.get('avg_cp_loss', 0):.1f}",
        f"Primary weakness this game: {summary.get('primary_weakness', 'general')}",
        "",
        "Key mistakes:",
    ]
    if bad_moves:
        for m in bad_moves[:6]:
            alts = ", ".join(m.get("alternatives", [])[:3])
            lines.append(
                f"  Move {m['move_number']}: {m['move']} "
                f"({m['classification']}, {m['cp_loss']}cp) — "
                f"best: {m['best_move']}, alts: {alts}, type: {m.get('mistake_type', 'general')}"
            )
    else:
        lines.append("  None — well played!")
    return "\n".join(lines)


def _build_profile_section(player_profile: dict) -> str:
    if not player_profile:
        return "No profile data yet."
    return (
        f"Estimated Elo: {player_profile.get('est_elo', '?')}\n"
        f"Games analyzed: {player_profile.get('games_played', 0)}\n"
        f"Primary weakness (all-time): {player_profile.get('primary_weakness', 'general')}"
    )


def _build_recent_mistakes_section(recent_mistakes: list) -> str:
    if not recent_mistakes:
        return "No previous mistakes on record."
    lines = []
    for m in recent_mistakes[:10]:
        lines.append(
            f"  Game {m['game_id']} — Move {m['move']} "
            f"({m['classification']}, {m['cp_loss']}cp, type: {m['mistake_type']})"
        )
    return "\n".join(lines)


def _build_trend_section(trend: list) -> str:
    if not trend:
        return "No game history available."
    if len(trend) == 1:
        return f"Only 1 game on record. Avg CP loss: {trend[0]['avg_cp_loss']:.1f}. Need more games to show trend."
    lines = []
    for i, g in enumerate(trend, 1):
        lines.append(
            f"  Game {i}: avg CP loss {g['avg_cp_loss']:.1f}, "
            f"blunders: {g['blunders']}, weakness: {g['primary_weakness']}"
        )
    first_cp = trend[0]["avg_cp_loss"]
    last_cp = trend[-1]["avg_cp_loss"]
    direction = "improving" if last_cp < first_cp else "getting worse" if last_cp > first_cp else "stable"
    lines.append(f"\nTrend: {direction} ({first_cp:.1f} → {last_cp:.1f} avg CP loss)")
    return "\n".join(lines)


def _build_patterns_section(patterns: dict) -> str:
    if not patterns:
        return "No pattern data yet."
    lines = []
    for mistake_type, count in sorted(patterns.items(), key=lambda x: -x[1]):
        lines.append(f"  {mistake_type}: {count} occurrence(s)")
    return "\n".join(lines)


def _build_rulebook_section(question: str) -> str:
    results = search_rulebook(question.lower())
    if not results:
        return ""
    entry = results[0]
    principles = "\n".join(f"  - {p}" for p in entry["key_principles"][:3])
    mistakes = "\n".join(f"  - {m}" for m in entry["common_mistakes"][:2])
    return (
        f"{entry['title']}: {entry['description']}\n\n"
        f"Key principles:\n{principles}\n\n"
        f"Common mistakes:\n{mistakes}"
    )


def _build_play_engine_section(play_context: dict) -> str:
    moves = play_context.get("moves", [])
    difficulty = play_context.get("difficulty", "easy")
    game_over = play_context.get("game_over", False)
    result = play_context.get("result")

    lines = [
        f"Difficulty: {difficulty}",
        f"Status: {'Result: ' + result if game_over else 'Game still in progress'}",
        f"Total moves played: {len(moves)}",
        "",
        "Move history:",
    ]
    for i, m in enumerate(moves[:15], 1):
        engine_text = m.get("engine_move") or "game ended"
        lines.append(f"  {i}. You: {m['user_move']}  →  Engine: {engine_text}")
    return "\n".join(lines)


def _build_prompt_for_intent(intent, question, game_analysis, player_profile, chat_history, user_id=None):
    sections = [_SYSTEM_HEADER]

    if game_analysis.get("summary"):
        sections.append(_section("CURRENT GAME", _build_game_section(game_analysis)))

    if intent == INTENT_MOVE:
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))

    elif intent == INTENT_HISTORY:
        if user_id:
            from database import get_improvement_trend
            trend = get_improvement_trend(user_id, last_n=5)
            sections.append(_section("GAME HISTORY (last 5 games)", _build_trend_section(trend)))
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))

    elif intent == INTENT_PATTERN:
        if user_id:
            from database import get_recent_mistakes, get_common_patterns
            patterns = get_common_patterns(user_id)
            recent = get_recent_mistakes(user_id, limit=10)
            sections.append(_section("MISTAKE PATTERNS (all games)", _build_patterns_section(patterns)))
            sections.append(_section("RECENT MISTAKE EXAMPLES", _build_recent_mistakes_section(recent)))
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))

    elif intent == INTENT_CONCEPT:
        rulebook = _build_rulebook_section(question)
        if rulebook:
            sections.append(_section("CHESS KNOWLEDGE BASE", rulebook))
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))
        sections.append("\nAfter explaining the concept, briefly relate it to the player's game if relevant.")

    else:
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))

    history_text = _format_history(chat_history)
    if history_text:
        sections.append(_section("CONVERSATION HISTORY", history_text))

    sections.append(f"\nUser: {question}\nCoach:")
    return "\n".join(sections)


def get_opening_message(game_analysis: dict, player_profile: dict) -> str:
    summary = game_analysis.get("summary", {})
    blunders = summary.get("blunders", 0)
    mistakes = summary.get("mistakes", 0)
    weakness = summary.get("primary_weakness", "general")

    prompt = f"""{_SYSTEM_HEADER}
{_section("CURRENT GAME", _build_game_section(game_analysis))}
{_section("PLAYER PROFILE", _build_profile_section(player_profile))}

Write a short 3-4 sentence opening message:
1. Summarize: {blunders} blunder(s), {mistakes} mistake(s).
2. Name the primary weakness: {weakness}.
3. Invite them to ask about any specific move.
Only reference moves from the game data. Never invent statistics.
Coach:"""

    response = _call_ollama(prompt, max_tokens=200)
    if response.startswith("ERROR"):
        return (
            f"I analyzed your game. You had {blunders} blunder(s) and "
            f"{mistakes} mistake(s). Your main weakness this game was "
            f"{weakness}. Ask me about any specific move!"
        )
    return response


def answer_question(
    question: str,
    game_analysis: dict,
    player_profile: dict,
    chat_history: list,
    play_context: Optional[dict] = None,
    user_id: Optional[str] = None,
) -> str:
    if play_context is not None:
        play_section = _build_play_engine_section(play_context)
        history_text = _format_history(chat_history)
        prompt = (
            f"{_SYSTEM_HEADER}"
            f"{_section('LIVE GAME', play_section)}"
            + (f"{_section('CONVERSATION HISTORY', history_text)}" if history_text else "")
            + f"\nUser: {question}\nCoach:"
        )
        response = _call_ollama(prompt, max_tokens=200)
        return response if not response.startswith("ERROR") else _ollama_fallback(response)

    moves_data = game_analysis.get("moves", [])
    allowed_moves = _build_allowed_moves(moves_data)

    for qm in _extract_piece_moves(question):
        if allowed_moves and qm.rstrip('+#') not in {m.rstrip('+#') for m in allowed_moves}:
            return (
                f"{qm} was not played in this game. "
                "Would you like to ask about a move that was actually played?"
            )

    intent = detect_intent(question)

    for attempt in range(MAX_RETRIES):
        prompt = _build_prompt_for_intent(
            intent, question, game_analysis, player_profile, chat_history, user_id
        )
        response = _call_ollama(prompt, max_tokens=300)

        if response.startswith("ERROR"):
            return _ollama_fallback(response)

        if _validate_no_hallucination(response, allowed_moves):
            return response

        chat_history = chat_history + [{
            "role": "system",
            "content": "[Note: previous response mentioned moves not in the game. Be more careful.]"
        }]

    return (
        "I can see this was a tricky position. Based on the analysis data, "
        "there was a significant evaluation swing. Would you like to ask about "
        "a specific move from the game?"
    )


def _format_history(chat_history: list) -> str:
    if not chat_history:
        return ""
    lines = []
    for msg in chat_history[-8:]:
        if msg.get("role") == "system":
            continue
        role = "User" if msg["role"] == "user" else "Coach"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def _ollama_fallback(error_msg: str) -> str:
    if "not running" in error_msg:
        return "Ollama is not running. Please start it with `ollama serve` and try again."
    return "Something went wrong with the AI backend. Please try again."
