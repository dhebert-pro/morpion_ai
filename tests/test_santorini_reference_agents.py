from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.reference_agents import (
    REFERENCE_AGENT_NAMES,
    choose_reference_action,
    count_immediate_wins,
)
from app.games.santorini.rules import apply_action, copy_game, create_new_game


def test_reference_agent_names_are_available():
    assert_true("random" in REFERENCE_AGENT_NAMES)
    assert_true("climber" in REFERENCE_AGENT_NAMES)
    assert_true("blocker" in REFERENCE_AGENT_NAMES)


def test_climber_prefers_winning_move_to_level_three():
    game = create_new_game()
    game["phase"] = "play"
    game["current_player"] = "O"
    game["workers"] = {"X": [0, 24], "O": [6, 18]}
    game["heights"][6] = 2
    game["heights"][7] = 3

    action = choose_reference_action(game, "climber")

    assert_equal(action["from"], 6)
    assert_equal(action["to"], 7)
    assert_equal(action["build"], None)


def test_blocker_avoids_leaving_immediate_win_when_possible():
    game = create_new_game()
    game["phase"] = "play"
    game["current_player"] = "O"
    game["workers"] = {"X": [6, 24], "O": [12, 18]}
    game["heights"][7] = 3

    action = choose_reference_action(game, "blocker")
    copied = copy_game(game)
    apply_action(copied, action)

    assert_equal(count_immediate_wins(copied, "X"), 0)


TESTS = [
    ("Santorini adversaires référence disponibles", test_reference_agent_names_are_available),
    ("Santorini climber prend une victoire immédiate", test_climber_prefers_winning_move_to_level_three),
    ("Santorini blocker évite une victoire immédiate", test_blocker_avoids_leaving_immediate_win_when_possible),
]
