from tests.test_helpers import assert_equal

from app.games.morpion.rules import create_new_game
from app.games.morpion.engine import play_turn


def test_play_turn_valid_move_and_ai_response():
    game = create_new_game()

    def fixed_ai_strategy(game):
        return 0

    result = play_turn(game, 4, fixed_ai_strategy)

    assert_equal(result["success"], True)
    assert_equal(result["finished"], False)
    assert_equal(result["ai_move"], 0)
    assert_equal(game["board"][4], "X")
    assert_equal(game["board"][0], "O")
    assert_equal(result["messages"], ["Coup enregistré : 4", "IA joue : 0"])


def test_play_turn_invalid_move_does_not_change_board():
    game = create_new_game()
    game["board"][4] = "O"

    def fixed_ai_strategy(game):
        return 0

    result = play_turn(game, 4, fixed_ai_strategy)

    assert_equal(result["success"], False)
    assert_equal(result["finished"], False)
    assert_equal(result["ai_move"], None)
    assert_equal(game["board"][4], "O")
    assert_equal(game["board"][0], None)
    assert_equal(result["messages"], ["Coup impossible."])


def test_play_turn_human_win_stops_before_ai():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "X"
    game["board"][4] = "O"

    def ai_should_not_play(game):
        raise AssertionError("L'IA n'aurait pas dû jouer.")

    result = play_turn(game, 2, ai_should_not_play)

    assert_equal(result["success"], True)
    assert_equal(result["finished"], True)
    assert_equal(result["ai_move"], None)
    assert_equal(game["board"][2], "X")
    assert_equal(result["messages"], ["Coup enregistré : 2", "Tu as gagné."])


def test_play_turn_draw_after_human_move_stops_before_ai():
    game = create_new_game()
    game["board"] = [
        "X", "O", "X",
        "X", "O", "O",
        "O", "X", None,
    ]

    def ai_should_not_play(game):
        raise AssertionError("L'IA n'aurait pas dû jouer.")

    result = play_turn(game, 8, ai_should_not_play)

    assert_equal(result["success"], True)
    assert_equal(result["finished"], True)
    assert_equal(result["ai_move"], None)
    assert_equal(game["board"][8], "X")
    assert_equal(result["messages"], ["Coup enregistré : 8", "Match nul : le plateau est plein."])


def test_play_turn_ai_win():
    game = create_new_game()
    game["board"][0] = "O"
    game["board"][1] = "O"
    game["board"][3] = "X"
    game["board"][4] = "X"

    def fixed_ai_strategy(game):
        return 2

    result = play_turn(game, 8, fixed_ai_strategy)

    assert_equal(result["success"], True)
    assert_equal(result["finished"], True)
    assert_equal(result["ai_move"], 2)
    assert_equal(game["board"][8], "X")
    assert_equal(game["board"][2], "O")
    assert_equal(result["messages"], ["Coup enregistré : 8", "IA joue : 2", "L'IA a gagné."])


TESTS = [
    ("Jouer un tour valide avec réponse IA", test_play_turn_valid_move_and_ai_response),
    ("Un coup invalide ne change pas le plateau", test_play_turn_invalid_move_does_not_change_board),
    ("Une victoire humaine arrête le tour avant l'IA", test_play_turn_human_win_stops_before_ai),
    ("Un match nul après le coup humain arrête le tour avant l'IA", test_play_turn_draw_after_human_move_stops_before_ai),
    ("Une victoire IA est détectée après sa réponse", test_play_turn_ai_win),
]
