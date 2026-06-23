from tests.test_helpers import assert_equal, assert_true

from app.ai.move_scoring import (
    score_legal_moves,
    choose_best_scored_move,
    create_move_score_example,
)

from app.games.morpion.rules import create_new_game


def create_game_with_board(board):
    game = create_new_game()
    game["board"] = board.copy()
    return game


def test_score_legal_moves_scores_only_legal_moves():
    game = create_game_with_board([
        "O", None, None,
        "X", None, None,
        None, None, "X",
    ])

    scores = score_legal_moves(game, "O", simulations_per_move=1)

    assert_equal(set(scores.keys()), {1, 2, 4, 5, 6, 7})

    for score in scores.values():
        assert_true(0.0 <= score <= 1.0)


def test_score_legal_moves_gives_one_to_immediate_win():
    game = create_game_with_board([
        "O", "O", None,
        "X", "X", None,
        None, None, None,
    ])

    scores = score_legal_moves(game, "O", simulations_per_move=1)

    assert_equal(scores[2], 1.0)


def test_choose_best_scored_move_returns_best_score():
    scores = {
        0: 0.25,
        1: 0.75,
        2: 0.50,
    }

    assert_equal(choose_best_scored_move(scores), 1)


def test_choose_best_scored_move_returns_none_without_moves():
    assert_equal(choose_best_scored_move({}), None)


def test_create_move_score_example_identifies_immediate_win():
    game = create_game_with_board([
        "O", "O", None,
        "X", "X", None,
        None, None, None,
    ])

    example = create_move_score_example(game, simulations_per_move=1)

    assert_equal(example["state_key"], "OO.XX....")
    assert_equal(example["player"], "O")
    assert_equal(example["best_move"], 2)
    assert_equal(len(example["move_scores"]), 5)


TESTS = [
    ("Le scoring note uniquement les coups légaux", test_score_legal_moves_scores_only_legal_moves),
    ("Le scoring donne 1 à une victoire immédiate", test_score_legal_moves_gives_one_to_immediate_win),
    ("Le meilleur coup scoré est sélectionné", test_choose_best_scored_move_returns_best_score),
    ("Aucun meilleur coup n'est choisi sans coup disponible", test_choose_best_scored_move_returns_none_without_moves),
    ("Un exemple de scoring identifie une victoire immédiate", test_create_move_score_example_identifies_immediate_win),
]