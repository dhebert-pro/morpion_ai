from app.games.santorini.action_index import action_to_output_index, output_index_to_action
from app.games.santorini.rules import get_legal_actions


def get_indexed_legal_actions(game, player=None):
    actions = []
    active_player = player or game["current_player"]

    for action in get_legal_actions(game, active_player):
        output_index = action_to_output_index(game, action, active_player)

        if output_index is None:
            continue

        indexed_action = action.copy()
        indexed_action["output_index"] = output_index
        actions.append(indexed_action)

    return actions


def move_to_output_index(move):
    if isinstance(move, dict) and "output_index" in move:
        return move["output_index"]

    raise ValueError("Coup Santorini non indexé : " + str(move))


def output_index_to_indexed_action(game, index, player=None):
    action = output_index_to_action(game, index, player)

    if action is None:
        return None

    action["output_index"] = index
    return action
