import random

from app.games.santorini.agents import choose_random_action
from app.games.santorini.rules import (
    apply_action,
    copy_game,
    get_legal_actions,
    get_game_result,
    switch_player,
)

REFERENCE_AGENT_NAMES = ["random", "climber", "blocker"]


def choose_reference_action(game, agent_name="random", rng=None):
    name = (agent_name or "random").lower()

    if name == "random":
        return choose_random_action(game, rng)
    if name == "climber":
        return choose_climber_action(game, rng)
    if name == "blocker":
        return choose_blocker_action(game, rng)

    raise ValueError("Adversaire Santorini inconnu : " + str(agent_name))


def choose_climber_action(game, rng=None):
    actions = _shuffled_legal_actions(game, rng)
    if not actions:
        return None

    return max(actions, key=lambda action: _climber_key(game, action))


def choose_blocker_action(game, rng=None):
    actions = _shuffled_legal_actions(game, rng)
    if not actions:
        return None

    return max(actions, key=lambda action: _blocker_key(game, action))


def count_immediate_wins(game, player):
    return sum(1 for action in get_legal_actions(game, player) if _is_winning_move(game, action))


def _shuffled_legal_actions(game, rng=None):
    actions = list(get_legal_actions(game))
    generator = rng or random
    generator.shuffle(actions)
    return actions


def _climber_key(game, action):
    return (
        _is_winning_move(game, action),
        game["heights"][action["to"]],
        _built_level_after_action(game, action),
    )


def _blocker_key(game, action):
    after = copy_game(game)
    apply_action(after, action)

    if get_game_result(after) == game["current_player"]:
        return (1, 0, 4, 4)

    opponent = after["current_player"]
    immediate_wins = count_immediate_wins(after, opponent)
    return (
        0,
        -immediate_wins,
        game["heights"][action["to"]],
        _built_level_after_action(game, action),
    )


def _is_winning_move(game, action):
    return game["heights"][action["to"]] == 3


def _built_level_after_action(game, action):
    build_cell = action.get("build")
    if build_cell is None:
        return 0
    if game["heights"][build_cell] == 3:
        return 4
    return game["heights"][build_cell] + 1
