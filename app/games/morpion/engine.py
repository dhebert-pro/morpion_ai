from app.ai.strategies import choose_ai_move

from app.games.morpion.rules import (
    is_valid_move,
    get_game_result,
)


def play_turn(game, human_move, ai_strategy=choose_ai_move):
    messages = []

    if not is_valid_move(game, human_move):
        return {
            "success": False,
            "finished": False,
            "ai_move": None,
            "messages": ["Coup impossible."]
        }

    game["board"][human_move] = "X"
    messages.append("Coup enregistré : " + str(human_move))

    result = get_game_result(game)
    if result == "X":
        messages.append("Tu as gagné.")
        return {
            "success": True,
            "finished": True,
            "ai_move": None,
            "messages": messages
        }

    if result == "draw":
        messages.append("Match nul : le plateau est plein.")
        return {
            "success": True,
            "finished": True,
            "ai_move": None,
            "messages": messages
        }

    ai_move = ai_strategy(game)
    game["board"][ai_move] = "O"
    messages.append("IA joue : " + str(ai_move))

    result = get_game_result(game)
    if result == "O":
        messages.append("L'IA a gagné.")
        return {
            "success": True,
            "finished": True,
            "ai_move": ai_move,
            "messages": messages
        }

    if result == "draw":
        messages.append("Match nul : le plateau est plein.")
        return {
            "success": True,
            "finished": True,
            "ai_move": ai_move,
            "messages": messages
        }

    return {
        "success": True,
        "finished": False,
        "ai_move": ai_move,
        "messages": messages
    }