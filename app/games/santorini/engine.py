from app.games.santorini.action_format import format_action
from app.games.santorini.coordinates import index_to_cell
from app.games.santorini.rules import (
    apply_action,
    is_valid_action,
    is_valid_placement,
    place_worker,
)


def play_input(game, parsed_input):
    if game["phase"] == "placement":
        return play_placement(game, parsed_input)

    return play_action(game, parsed_input)


def play_placement(game, cell):
    if not isinstance(cell, int) or not is_valid_placement(game, cell):
        return {
            "success": False,
            "finished": False,
            "messages": ["Placement impossible."],
        }

    player = game["current_player"]
    place_worker(game, cell)
    return {
        "success": True,
        "finished": game["phase"] == "finished",
        "messages": ["Placement " + player + " en " + index_to_cell(cell) + "."],
    }


def play_action(game, action):
    if not isinstance(action, dict) or not is_valid_action(game, action):
        return {
            "success": False,
            "finished": False,
            "messages": ["Coup impossible."],
        }

    player = game["current_player"]
    action_text = format_action(action)
    apply_action(game, action)
    messages = ["Coup " + player + " : " + action_text + "."]

    if game["phase"] == "finished":
        messages.append("Victoire de " + str(game["winner"]) + ".")

    return {
        "success": True,
        "finished": game["phase"] == "finished",
        "messages": messages,
    }
