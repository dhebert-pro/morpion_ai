from app.games.santorini.coordinates import get_neighbors
from app.games.santorini.rules import (
    apply_action,
    copy_game,
    get_game_result,
    get_legal_actions,
    get_worker_at,
)


HIGH_RISK = 9999


def santorini_action_risk_key(game, action):
    """Return a shallow tactical risk tuple for an action.

    Lower is safer. The tuple is intentionally generic and does not solve the game:
    - immediate wins offered to the opponent;
    - opponent replies that create a direct level-3 threat;
    - opponent ability to climb high next turn.
    """
    after = copy_game(game)
    active_player = after["current_player"]

    if not apply_action(after, action):
        return (HIGH_RISK, HIGH_RISK, HIGH_RISK, HIGH_RISK)

    if get_game_result(after) == active_player:
        return (0, 0, 0, 0)

    opponent = after["current_player"]
    return (
        count_immediate_winning_actions(after, opponent),
        count_threat_creating_replies(after, opponent),
        max_destination_height(after, opponent),
        count_high_climb_actions(after, opponent),
    )


def count_immediate_winning_actions(game, player):
    return sum(1 for action in get_legal_actions(game, player) if is_winning_action(game, action))


def count_threat_creating_replies(game, player):
    total = 0

    for action in get_legal_actions(game, player):
        if is_winning_action(game, action):
            total += HIGH_RISK
        elif _reply_creates_level_three_threat(game, action, player):
            total += 1

    return total


def max_destination_height(game, player):
    heights = [game["heights"][action["to"]] for action in get_legal_actions(game, player)]
    return max(heights) if heights else 0


def count_high_climb_actions(game, player):
    return sum(1 for action in get_legal_actions(game, player) if game["heights"][action["to"]] >= 2)


def is_winning_action(game, action):
    return game["heights"][action["to"]] == 3


def _reply_creates_level_three_threat(game, action, player):
    if game["heights"][action["to"]] != 2:
        return False

    for neighbor in get_neighbors(action["to"]):
        if _is_existing_level_three_target(game, neighbor):
            return True

        if _is_new_level_three_target(game, neighbor, action.get("build")):
            return True

    return False


def _is_existing_level_three_target(game, cell):
    if game["heights"][cell] != 3:
        return False
    if game["domes"][cell]:
        return False
    return get_worker_at(game, cell) is None


def _is_new_level_three_target(game, cell, build_cell):
    return cell == build_cell and game["heights"][cell] == 2


def describe_santorini_risk_after_action(game, action):
    after = copy_game(game)
    active_player = after["current_player"]

    if not apply_action(after, action):
        return {
            "valid": False,
            "opponent_immediate_wins": HIGH_RISK,
            "opponent_threat_replies": HIGH_RISK,
            "opponent_max_destination_height": HIGH_RISK,
            "opponent_high_climb_actions": HIGH_RISK,
        }

    if get_game_result(after) == active_player:
        return {
            "valid": True,
            "winning_action": True,
            "opponent_immediate_wins": 0,
            "opponent_threat_replies": 0,
            "opponent_max_destination_height": 0,
            "opponent_high_climb_actions": 0,
        }

    opponent = after["current_player"]
    return {
        "valid": True,
        "winning_action": False,
        "opponent_immediate_wins": count_immediate_winning_actions(after, opponent),
        "opponent_threat_replies": count_threat_creating_replies(after, opponent),
        "opponent_max_destination_height": max_destination_height(after, opponent),
        "opponent_high_climb_actions": count_high_climb_actions(after, opponent),
    }
