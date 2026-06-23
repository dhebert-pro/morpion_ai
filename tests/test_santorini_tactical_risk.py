from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.rules import create_new_game
from app.games.santorini.tactical_guard import filter_santorini_tactical_actions
from app.games.santorini.tactical_risk import (
    count_threat_creating_replies,
    describe_santorini_risk_after_action,
)


def _play_game(workers, current_player="O"):
    game = create_new_game()
    game["phase"] = "play"
    game["current_player"] = current_player
    game["workers"] = workers
    return game


def test_counts_opponent_replies_that_create_level_three_threats():
    game = _play_game({"X": [12, 24], "O": [8, 0]})
    game["heights"][12] = 1
    game["heights"][13] = 2
    game["heights"][14] = 3
    actions = get_indexed_legal_actions(game, "O")
    bad_action = _find_action(actions, from_cell=8, to_cell=9, build_cell=4)

    risk = describe_santorini_risk_after_action(game, bad_action)

    assert_equal(risk["opponent_immediate_wins"], 0)
    assert_true(risk["opponent_threat_replies"] > 0)


def test_guard_prefers_blocking_future_level_three_threat():
    game = _play_game({"X": [12, 24], "O": [8, 0]})
    game["heights"][12] = 1
    game["heights"][13] = 2
    game["heights"][14] = 3
    actions = get_indexed_legal_actions(game, "O")

    filtered = filter_santorini_tactical_actions(game, actions)
    threat_counts = []

    for action in filtered:
        risk = describe_santorini_risk_after_action(game, action)
        threat_counts.append(risk["opponent_threat_replies"])

    assert_true(len(filtered) > 0)
    assert_equal(min(threat_counts), 0)
    assert_true(all(value == 0 for value in threat_counts))


def _find_action(actions, from_cell, to_cell, build_cell):
    for action in actions:
        if action["from"] == from_cell and action["to"] == to_cell and action.get("build") == build_cell:
            return action

    raise AssertionError("Action attendue introuvable")


TESTS = [
    (
        "Santorini risque compte les réponses créant menace niveau 3",
        test_counts_opponent_replies_that_create_level_three_threats,
    ),
    (
        "Santorini garde évite menace niveau 3 au tour suivant",
        test_guard_prefers_blocking_future_level_three_threat,
    ),
]
