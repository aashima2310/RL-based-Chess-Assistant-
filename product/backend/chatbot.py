import re
import os
import chess
from typing import Optional
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")

load_dotenv(dotenv_path, encoding="utf-8-sig")

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
try:
    from groq import Groq
    groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
except Exception as e:
    print(f"[WARNING] Groq library error: {e}")
    groq_client = None

MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
MAX_RETRIES = 2

from rulebook import search_rulebook

PERSONALITIES = {
    "encouraging": {
        "label": "Encouraging",
        "tone_instruction": "Be warm, encouraging, and supportive. Celebrate what went well before addressing mistakes.",
        "greeting_suffix": "You're doing great — let's keep improving together!",
    },
    "direct": {
        "label": "Direct & Blunt",
        "tone_instruction": "Be blunt, direct, and no-nonsense. Skip pleasantries and get straight to the point.",
        "greeting_suffix": "Let's get to work.",
    },
    "witty": {
        "label": "Witty",
        "tone_instruction": "Be witty and a little sarcastic, while still being genuinely helpful and accurate.",
        "greeting_suffix": "Let's see what chaos we created this time.",
    },
    "grandmaster": {
        "label": "Grandmaster",
        "tone_instruction": "Speak like a seasoned grandmaster giving a post-game analysis — precise, calm, a bit formal.",
        "greeting_suffix": "Let's review the game.",
    },
}
DEFAULT_PERSONALITY = "encouraging"

def get_personality(key: str) -> dict:
    return PERSONALITIES.get(key, PERSONALITIES[DEFAULT_PERSONALITY])

INTENT_SUMMARY        = "summary"          # Python only
INTENT_MOVE_DETAILS   = "move_details"     # Python only
INTENT_BEST_MOVE      = "best_move"        # Python only
INTENT_IMPROVEMENT    = "improvement"      # Python only
INTENT_ALL_BEST_MOVES = "all_best_moves"  # Python only (Multi-move summary)
INTENT_EXPLANATION    = "explanation"      # RAG prompt
INTENT_CONCEPT        = "concept"          # RAG prompt
INTENT_TREND          = "trend"            # RAG prompt (history)
INTENT_PATTERN        = "pattern"          # RAG prompt (history)
INTENT_GENERAL        = "general"          # RAG prompt

_SUMMARY_KEYWORDS        = ["how did i do", "overview", "summary", "how was my game", "how did i play", "recap"]
_MOVE_DETAILS_KEYWORDS    = ["what was my mistake", "what happened", "what did i play", "what move was", "what was wrong with", "mistake on move"]
_BEST_MOVE_KEYWORDS      = ["what else could", "what other move", "what should i have", "best move", "better move", "alternative", "instead"]
_EXPLANATION_KEYWORDS    = ["why", "explain why", "how come", "what makes"]
_HISTORY_KEYWORDS        = ["progress", "trend", "getting better", "previous game", "last game", "past", "history", "before", "recent", "recently"]
_PATTERN_KEYWORDS        = ["always", "keep", "repeat", "pattern", "weakness", "weak", "common", "often", "habit", "same mistake"]
_CONCEPT_KEYWORDS        = ["what is", "what's", "explain", "how does", "tell me about", "define", "meaning", "how to", "teach", "describe"]
_CHESS_TERMS             = ["pin", "fork", "skewer", "hanging", "castle", "castling", "en passant", "check", "checkmate", "stalemate", "opposition", "endgame", "opening", "development", "center", "pawn", "tactic", "fianchetto", "zugzwang", "tempo", "gambit", "sacrifice"]
_CURRENT_GAME_KEYWORDS   = ["this game", "in this game", "this match", "current game", "today's game"]
_HISTORY_PATTERN_KEYWORDS= ["across games", "across my games", "all games", "keep making", "usually", "overall"]
_IMPROVEMENT_KEYWORDS    = ["how can i improve", "how do i improve", "how should i improve", "improve this game", "improve my play"]

_MOVE_NUMBER_PATTERN = re.compile(r'\bmove\s*#?(\d+)\b', re.IGNORECASE)

def detect_intent(question: str) -> str:
    q = question.lower()
    
    is_multi = any(k in q for k in ["all", "every", "each", "moves", "mistakes", "blunders", "wrong moves"])
    is_best_or_detail = any(k in q for k in _BEST_MOVE_KEYWORDS) or any(k in q for k in _MOVE_DETAILS_KEYWORDS)
    
    if is_multi and is_best_or_detail:
        return INTENT_ALL_BEST_MOVES
        
    if any(k in q for k in _CURRENT_GAME_KEYWORDS): return INTENT_MOVE_DETAILS
    if any(k in q for k in _SUMMARY_KEYWORDS): return INTENT_SUMMARY
    if any(k in q for k in _BEST_MOVE_KEYWORDS): return INTENT_BEST_MOVE
    if any(k in q for k in _MOVE_DETAILS_KEYWORDS): return INTENT_MOVE_DETAILS
    if any(k in q for k in _IMPROVEMENT_KEYWORDS): return INTENT_IMPROVEMENT
    if any(k in q for k in _HISTORY_PATTERN_KEYWORDS) or any(k in q for k in _PATTERN_KEYWORDS): return INTENT_PATTERN
    if any(k in q for k in _HISTORY_KEYWORDS): return INTENT_TREND
    if any(k in q for k in _CONCEPT_KEYWORDS) and any(t in q for t in _CHESS_TERMS): return INTENT_CONCEPT
    if any(k in q for k in _EXPLANATION_KEYWORDS): return INTENT_EXPLANATION
    
    return INTENT_GENERAL


def _extract_move_number(question: str, chat_history: list) -> Optional[int]:
    m = _MOVE_NUMBER_PATTERN.search(question)
    if m: return int(m.group(1))
    for msg in reversed(chat_history[-6:]):
        m = _MOVE_NUMBER_PATTERN.search(msg.get("content", ""))
        if m: return int(m.group(1))
    return None

def _extract_piece_moves(text: str) -> list:
    pattern = r'\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)\b'
    return re.findall(pattern, text)

def _build_allowed_moves(moves_data: list) -> set:
    allowed = set()
    for m in moves_data:
        if m.get("move"): allowed.add(m["move"])
        if m.get("best_move"): allowed.add(m["best_move"])
        for alt in m.get("alternatives", []): allowed.add(alt)
        if m.get("move_uci"): allowed.add(m["move_uci"])
    return allowed

def _validate_no_hallucination(response: str, allowed_moves: set) -> bool:
    if not allowed_moves: return True
    normalized_allowed = {m.rstrip('+#') for m in allowed_moves}
    for move in _extract_piece_moves(response):
        if move.rstrip('+#') not in normalized_allowed:
            return False
    return True

def _find_move(moves_data: list, move_number: Optional[int]) -> Optional[dict]:
    if move_number is not None:
        for m in moves_data:
            if m["move_number"] == move_number:
                return m
        return None
    bad = [m for m in moves_data if m["classification"] in ("Blunder", "Mistake", "Inaccuracy")]
    if bad:
        return max(bad, key=lambda m: m["cp_loss"])
    return None


def _answer_summary(game_analysis: dict) -> str:
    s = game_analysis.get("summary", {})
    weakness = (s.get("primary_weakness") or "general").replace("_", " ")
    return (
        f"You played {s.get('total_moves', 0)} moves with an average centipawn loss "
        f"of {s.get('avg_cp_loss', 0):.0f}. That breaks down to {s.get('blunders', 0)} "
        f"blunder(s), {s.get('mistakes', 0)} mistake(s), and {s.get('inaccuracies', 0)} "
        f"inaccuracy(ies). Your main weakness this game was {weakness}."
    )

def _answer_move_details(move: dict) -> str:
    if move is None:
        return "Excellent game! I couldn't find any mistakes, blunders, or inaccuracies worth discussing."
    if move["classification"] == "Good":
        return f"Move {move['move_number']} ({move['move']}) was actually a good move — no issue there."
    mtype = (move.get("mistake_type") or "general").replace("_", " ")
    return (
        f"Move {move['move_number']}: you played {move['move']}, which was a "
        f"{move['classification'].lower()} ({move['cp_loss']}cp lost). "
        f"The engine's best move there was {move['best_move']}. "
        f"This was classified as a {mtype} issue."
    )

def _answer_best_move(move: dict) -> str:
    if move is None:
        return "I couldn't find that move in this game. Could you double check the move number?"
    alts = [a for a in move.get("alternatives", []) if a != move.get("best_move")]
    alt_text = f", or alternatively {', '.join(alts[:2])}" if alts else ""
    return (
        f"On move {move['move_number']}, instead of {move['move']}, the engine "
        f"recommends {move['best_move']}{alt_text}. "
        f"That would have avoided the {move['cp_loss']}cp loss."
    )

def _answer_all_best_moves(game_analysis: dict) -> str:
    moves = game_analysis.get("moves", [])
    bad_moves = [m for m in moves if m["classification"] in ("Blunder", "Mistake", "Inaccuracy")]
    
    if not bad_moves:
        return "You played a brilliant game! There are no significant mistakes or blunders to show alternatives for."
        
    lines = ["Here are better alternatives for your significant mistakes:\n"]
    for m in bad_moves:
        lines.append(f"• **Move {m['move_number']}**")
        lines.append(f"  Played: `{m['move']}` ({m['classification']}, {m['cp_loss']}cp lost)")
        lines.append(f"  Best: `{m['best_move']}`")
        lines.append("")
        
    return "\n".join(lines)

def _answer_improvement(game_analysis: dict) -> str:
    s = game_analysis.get("summary", {})
    weakness = s.get("primary_weakness", "general").replace("_"," ")
    return (
        f"Your biggest area for improvement in THIS game was {weakness}. Focus on reducing "
        f"{s.get('blunders', 0)} blunders and {s.get('inaccuracies', 0)} inaccuracies. "
        f"Improving piece safety will have the biggest impact."
    )

def get_opening_message(game_analysis: dict, player_profile: dict, personality: str = DEFAULT_PERSONALITY) -> str:
    tone = get_personality(personality)
    return _answer_summary(game_analysis) + " Ask me about any specific move, or type 'why' for a deeper explanation! " + tone["greeting_suffix"]


def _get_system_header(personality: str) -> str:
    tone = get_personality(personality)
    return f"""You are ChessRL, a personalized chess coach. {tone['tone_instruction']}
Use ONLY the verified information provided in the sections below.

Rules you MUST follow:
  - Never invent statistics, percentages, or trends not present in the data.
  - Never invent previous games or moves not listed here.
  - CRITICAL: Do NOT mention "Game 1", "Game 2", or fabricate previous games. Discuss ONLY the verified data.
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
    if not player_profile: return "No profile data yet."
    return (
        f"Estimated Elo: {player_profile.get('est_elo', '?')}\n"
        f"Games analyzed: {player_profile.get('games_played', 0)}\n"
        f"Primary weakness (all-time): {player_profile.get('primary_weakness', 'general')}"
    )

def _build_recent_mistakes_section(recent_mistakes: list) -> str:
    if not recent_mistakes: return "No previous mistakes on record."
    lines = []
    for m in recent_mistakes[:10]:
        lines.append(
            f"  Game {m['game_id']} — Move {m['move']} "
            f"({m['classification']}, {m['cp_loss']}cp, type: {m['mistake_type']})"
        )
    return "\n".join(lines)

def _build_trend_section(trend: list) -> str:
    if not trend: return "No game history available."
    if len(trend) == 1:
        return f"Only 1 game on record. Avg CP loss: {trend[0]['avg_cp_loss']:.1f}. Need more games to show trend."
    lines = []
    for i, g in enumerate(trend, 1):
        lines.append(f"  Game {i}: avg CP loss {g['avg_cp_loss']:.1f}, blunders: {g['blunders']}, weakness: {g['primary_weakness']}")
    first_cp, last_cp = trend[0]["avg_cp_loss"], trend[-1]["avg_cp_loss"]
    direction = "improving" if last_cp < first_cp else "getting worse" if last_cp > first_cp else "stable"
    lines.append(f"\nTrend: {direction} ({first_cp:.1f} → {last_cp:.1f} avg CP loss)")
    return "\n".join(lines)

def _build_patterns_section(patterns: dict) -> str:
    if not patterns: return "No pattern data yet."
    lines = []
    for mistake_type, count in sorted(patterns.items(), key=lambda x: -x[1]):
        lines.append(f"  {mistake_type}: {count} occurrence(s)")
    return "\n".join(lines)

def _build_rulebook_section(question: str) -> str:
    results = search_rulebook(question.lower())
    if not results: return ""
    entry = results[0]
    principles = "\n".join(f"  - {p}" for p in entry["key_principles"][:3])
    mistakes = "\n".join(f"  - {m}" for m in entry["common_mistakes"][:2])
    return f"{entry['title']}: {entry['description']}\n\nKey principles:\n{principles}\n\nCommon mistakes:\n{mistakes}"

def _build_play_engine_section(play_context: dict) -> str:
    moves = play_context.get("moves", [])
    difficulty = play_context.get("difficulty", "easy")
    game_over = play_context.get("game_over", False)
    result = play_context.get("result")
    lines = [
        f"Difficulty: {difficulty}",
        f"Status: {'Result: ' + result if game_over else 'Game still in progress'}",
        f"Total moves played: {len(moves)}",
        "", "Move history:",
    ]
    for i, m in enumerate(moves[:15], 1):
        lines.append(f"  {i}. You: {m['user_move']}  →  Engine: {m.get('engine_move') or 'game ended'}")
    return "\n".join(lines)

def _build_prompt_for_intent(intent, question, game_analysis, player_profile, chat_history, user_id, personality):
    sections = [_get_system_header(personality)]

    if game_analysis.get("summary"):
        sections.append(_section("CURRENT GAME", _build_game_section(game_analysis)))

    if intent == INTENT_EXPLANATION:
        if player_profile:
            sections.append(_section("PLAYER PROFILE", _build_profile_section(player_profile)))

    elif intent == INTENT_TREND:
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

def _format_history(chat_history: list) -> str:
    if not chat_history: return ""
    lines = []
    for msg in chat_history[-8:]:
        if msg.get("role") == "system": continue
        role = "User" if msg["role"] == "user" else "Coach"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def call_llm(prompt: str, max_tokens: int = 300) -> str:
    if groq_client is None:
        return "GROQ_API_KEY is missing or invalid. Please check your deployment variables."
    try:
        resp = groq_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GROQ ERROR] {type(e).__name__}: {e}")
        return f"GROQ API Error: {e}"

def call_llm_fast(prompt: str) -> str:
    return call_llm(prompt, max_tokens=180)

call_ollama_fast = call_llm_fast
_call_ollama = call_llm


def answer_question(
    question: str,
    game_analysis: dict,
    player_profile: dict,
    chat_history: list,
    play_context: Optional[dict] = None,
    user_id: Optional[str] = None,
    personality: str = DEFAULT_PERSONALITY
) -> str:
    
    if play_context is not None:
        play_section = _build_play_engine_section(play_context)
        history_text = _format_history(chat_history)
        prompt = (
            f"{_get_system_header(personality)}"
            f"{_section('LIVE GAME', play_section)}"
            + (f"{_section('CONVERSATION HISTORY', history_text)}" if history_text else "")
            + f"\nUser: {question}\nCoach:"
        )
        response = call_llm(prompt, max_tokens=200)
        return response if not response.startswith("GROQ") else "I'm having trouble analyzing the live game right now. Please try again in a moment."

    moves_data = game_analysis.get("moves", [])
    move_num = _extract_move_number(question, chat_history)
    
    if move_num is not None:
        valid_nums = {m["move_number"] for m in moves_data}
        if valid_nums and move_num not in valid_nums:
            return f"Move {move_num} wasn't played in this game. This game only has {len(moves_data)} moves."

    allowed_moves = _build_allowed_moves(moves_data)
    for qm in _extract_piece_moves(question):
        if allowed_moves and qm.rstrip('+#') not in {m.rstrip('+#') for m in allowed_moves}:
            return f"{qm} was not played in this game. Would you like to ask about a move that was actually played?"

    intent = detect_intent(question)

    if intent == INTENT_SUMMARY:
        return _answer_summary(game_analysis)
    if intent == INTENT_ALL_BEST_MOVES:
        return _answer_all_best_moves(game_analysis)
    if intent == INTENT_MOVE_DETAILS:
        return _answer_move_details(_find_move(moves_data, move_num))
    if intent == INTENT_BEST_MOVE:
        return _answer_best_move(_find_move(moves_data, move_num))
    if intent == INTENT_IMPROVEMENT:
        return _answer_improvement(game_analysis)

    for attempt in range(MAX_RETRIES):
        prompt = _build_prompt_for_intent(
            intent, question, game_analysis, player_profile, chat_history, user_id, personality
        )
        response = call_llm(prompt, max_tokens=300)

        if response.startswith("GROQ"):
            return "I encountered a issue reaching my AI thought engine. Please ask your question again!"

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
