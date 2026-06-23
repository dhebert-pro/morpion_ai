from app.games.santorini.rules import apply_action, copy_game, get_game_result, get_legal_actions
from app.games.santorini.tactical_risk import santorini_action_risk_key


def filter_santorini_tactical_actions(game, actions):
    """Keep tactically forced/safe Santorini actions.

    The filter is deliberately shallow and generic:
    - take an immediate win when it exists;
    - otherwise prefer actions that do not give the opponent an immediate win.
    """
    candidate_actions = list(actions)

    if not candidate_actions:
        return []

    winning_actions = [action for action in candidate_actions if is_winning_action(game, action)]

    if winning_actions:
        return winning_actions

    risks = [(santorini_action_risk_key(game, action), action) for action in candidate_actions]
    lowest_risk = min(risk for risk, _action in risks)
    return [action for risk, action in risks if risk == lowest_risk]


def is_winning_action(game, action):
    return game["heights"][action["to"]] == 3


def count_immediate_winning_actions(game, player=None):
    return sum(1 for action in get_legal_actions(game, player) if is_winning_action(game, action))


def _opponent_immediate_wins_after(game, action):
    copied_game = copy_game(game)
    active_player = copied_game["current_player"]

    if not apply_action(copied_game, action):
        return 999

    if get_game_result(copied_game) == active_player:
        return 0

    opponent = copied_game["current_player"]
    return count_immediate_winning_actions(copied_game, opponent)
