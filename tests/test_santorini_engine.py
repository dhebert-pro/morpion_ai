from tests.test_helpers import assert_equal

from app.games.santorini.action_format import parse_human_input
from app.games.santorini.engine import play_input
from app.games.santorini.rules import create_new_game


def test_play_input_placement_message():
    game = create_new_game()
    result = play_input(game, parse_human_input("A1"))
    assert_equal(result["success"], True)
    assert_equal(result["messages"], ["Placement X en A1."])


def test_play_input_rejects_bad_placement():
    game = create_new_game()
    play_input(game, parse_human_input("A1"))
    result = play_input(game, parse_human_input("A1"))
    assert_equal(result["success"], False)
    assert_equal(result["messages"], ["Placement impossible."])


def test_play_input_action_message():
    game = create_new_game()
    for text in ["A1", "E1", "A5", "E5"]:
        play_input(game, parse_human_input(text))

    result = play_input(game, parse_human_input("A1-B2:B1"))
    assert_equal(result["success"], True)
    assert_equal(result["messages"], ["Coup X : A1-B2:B1."])


TESTS = [
    ("Santorini joue un placement", test_play_input_placement_message),
    ("Santorini refuse un mauvais placement", test_play_input_rejects_bad_placement),
    ("Santorini joue un coup complet", test_play_input_action_message),
]
