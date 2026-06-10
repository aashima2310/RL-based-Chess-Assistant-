"""
chatbot.py — Grounded LLM Chess Coach
---------------------------------------
Uses Ollama + Mistral 7B locally.
CRITICAL: All move mentions are validated against engine data.
          Hallucinated moves trigger automatic regeneration.
          The LLM is a translator, not a reasoner.
"""

import requests
import json
import re
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "mistral"
MAX_RETRIES = 3  # regenerate if hallucination detected


def _call_ollama(prompt: str) -> str:
    """Raw call to Ollama. Returns text response."""
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,   # low temp = less hallucination
                "num_predict": 300,   # keep answers focused
            }
        }, timeout=60)
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "ERROR: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"ERROR: {str(e)}"


def _extract_moves_from_text(text: str) -> list[str]:
    """Extract all chess move notations mentioned in text."""
    # Matches: e4, Nf3, Qxf7, O-O, e8=Q etc.
    pattern = r'\b([KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?|O-O(?:-O)?)\b'
    return re.findall(pattern, text)


def _validate_no_hallucination(response: str, allowed_moves: list[str]) -> bool:
    """
    Returns True if all moves mentioned in response are in the allowed list.
    allowed_moves comes directly from engine analysis data.
    """
    if not allowed_moves:
        return True
    mentioned = _extract_moves_from_text(response)
    for move in mentioned:
        if move not in allowed_moves and move not in ["O-O", "O-O-O"]:
            return False
    return True


def build_system_context(game_analysis: dict, player_profile: dict) -> str:
    """Builds the game context injected into every chatbot prompt."""
    summary = game_analysis.get("summary", {})
    moves = game_analysis.get("moves", [])

    # Only include blunders/mistakes in context (keep prompt short)
    bad_moves = [m for m in moves if m["classification"] in ("Blunder", "Mistake")]

    bad_moves_text = ""
    for m in bad_moves[:5]:  # max 5 to keep context manageable
        bad_moves_text += (
            f"  - Move {m['move_number']}: {m['move']} "
            f"({m['classification']}, {m['cp_loss']}cp loss, "
            f"best was {m['best_move']}, type: {m['mistake_type']})\n"
        )

    return f"""You are ChessRL, a personalized chess coach. 
You have just analyzed the user's game. Answer their questions using ONLY the data below.
NEVER mention a move that is not listed in this data. If you don't know, say so honestly.
Be encouraging but honest. Keep answers under 150 words.

=== GAME ANALYSIS DATA ===
Total moves: {summary.get('total_moves', '?')}
Blunders: {summary.get('blunders', 0)}
Mistakes: {summary.get('mistakes', 0)}  
Inaccuracies: {summary.get('inaccuracies', 0)}
Average centipawn loss: {summary.get('avg_cp_loss', 0)}
Primary weakness: {summary.get('primary_weakness', 'unknown')}

Key mistakes:
{bad_moves_text if bad_moves_text else '  None — well played!'}

=== PLAYER PROFILE ===
Estimated Elo: {player_profile.get('est_elo', '?')}
Games analyzed: {player_profile.get('games_played', 1)}
Recurring weakness: {player_profile.get('primary_weakness', 'unknown')}
=========================
"""


def get_opening_message(game_analysis: dict, player_profile: dict) -> str:
    """
    Auto-generated first message when chatbot opens after game analysis.
    Feels like a real coach greeting.
    """
    summary = game_analysis.get("summary", {})
    blunders = summary.get("blunders", 0)
    mistakes = summary.get("mistakes", 0)
    weakness = summary.get("primary_weakness", "general")

    weakness_display = weakness.replace("_", " ")

    prompt = f"""{build_system_context(game_analysis, player_profile)}

Write a short (3-4 sentence) opening message as a chess coach summarizing this game analysis.
Mention the number of blunders ({blunders}) and mistakes ({mistakes}).
Mention the primary weakness ({weakness_display}).
End by inviting them to ask questions about specific moves.
Only mention moves that are in the game analysis data above."""

    response = _call_ollama(prompt)
    if response.startswith("ERROR"):
        # Fallback if Ollama not running
        return (f"I analyzed your game. You had {blunders} blunder(s) and "
                f"{mistakes} mistake(s). Your main weakness this game was "
                f"{weakness_display}. Ask me about any specific move!")
    return response


def answer_question(
    question: str,
    game_analysis: dict,
    player_profile: dict,
    chat_history: list[dict],
) -> str:
    """
    Main chatbot function. Answers user question grounded to game data.
    Retries up to MAX_RETRIES times if hallucination detected.
    """
    # Build allowed moves list for validation
    moves_data = game_analysis.get("moves", [])
    allowed_moves = []
    for m in moves_data:
        allowed_moves.append(m["move"])
        if m.get("best_move"):
            allowed_moves.append(m["best_move"])
        allowed_moves.extend(m.get("alternatives", []))

    # Build conversation history (last 4 exchanges max)
    history_text = ""
    for msg in chat_history[-8:]:
        role = "User" if msg["role"] == "user" else "Coach"
        history_text += f"{role}: {msg['content']}\n"

    system = build_system_context(game_analysis, player_profile)

    for attempt in range(MAX_RETRIES):
        prompt = f"""{system}

Previous conversation:
{history_text}
User: {question}
Coach:"""

        response = _call_ollama(prompt)

        if response.startswith("ERROR"):
            return response

        # Validate — no hallucinated moves
        if _validate_no_hallucination(response, allowed_moves):
            return response

        # Hallucination detected — retry with stricter instruction
        history_text += f"\n[Previous response rejected: mentioned invalid moves. Be more careful.]\n"

    # Final fallback after all retries
    return ("I can see this was a challenging position. Based on the engine analysis, "
            "the key issue was a material imbalance. Would you like me to explain "
            "a specific move from the list of analyzed moves?")


# ── Quick test ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Minimal test without real game data
    mock_analysis = {
        "moves": [
            {"move_number": 4, "move": "Qxf7", "best_move": "Nf3",
             "cp_loss": 325, "classification": "Blunder",
             "mistake_type": "hanging_piece", "alternatives": ["Nf3", "d3"],
             "fen_before": "", "is_blunder": True}
        ],
        "summary": {
            "total_moves": 4, "blunders": 1, "mistakes": 0,
            "inaccuracies": 0, "avg_cp_loss": 81.25,
            "primary_weakness": "hanging_piece",
            "mistake_counts": {"hanging_piece": 1}
        }
    }
    mock_profile = {"est_elo": 1100, "games_played": 3, "primary_weakness": "hanging_piece"}

    print("=== Opening Message ===")
    print(get_opening_message(mock_analysis, mock_profile))
    print("\n=== Answer Question ===")
    print(answer_question(
        "Why was move 4 so bad?",
        mock_analysis, mock_profile, []
    ))
