from tests.test_helpers import assert_equal

from app.games.morpion.rules import create_new_game, get_legal_moves

from app.ai.strategies import (
    find_winning_move,
    choose_random_move,
    choose_ai_move,
    choose_model_move,
)


def test_ai_move_is_legal():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"

    ai_move = choose_ai_move(game)

    assert ai_move in get_legal_moves(game), f"L'IA a choisi un coup illégal : {ai_move}"


def test_ai_plays_only_remaining_cell():
    for empty_cell in range(9):
        game = create_new_game()

        for index in range(9):
            if index != empty_cell:
                game["board"][index] = "X"

        ai_move = choose_ai_move(game)

        assert_equal(ai_move, empty_cell)


def test_find_winning_move_for_x():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "X"

    assert_equal(find_winning_move(game, "X"), 2)


def test_find_winning_move_for_o():
    game = create_new_game()
    game["board"][3] = "O"
    game["board"][4] = "O"

    assert_equal(find_winning_move(game, "O"), 5)


def test_find_winning_move_returns_none_when_no_win_available():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"

    assert_equal(find_winning_move(game, "X"), None)
    assert_equal(find_winning_move(game, "O"), None)


def test_ai_takes_winning_move():
    game = create_new_game()
    game["board"][0] = "O"
    game["board"][1] = "O"
    game["board"][4] = "X"

    assert_equal(choose_ai_move(game), 2)


def test_ai_blocks_human_winning_move():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "X"
    game["board"][4] = "O"

    assert_equal(choose_ai_move(game), 2)


def test_random_move_is_legal():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"

    random_move = choose_random_move(game)

    assert random_move in get_legal_moves(game), f"Coup aléatoire illégal : {random_move}"


def test_model_move_is_used_when_known():
    game = create_new_game()
    game["board"][4] = "X"

    model = {
        "....X....": 0
    }

    def fallback_should_not_be_used(game):
        raise AssertionError("Le modèle connaissait l'état, donc le fallback ne devait pas être utilisé.")

    move = choose_model_move(game, model, fallback_should_not_be_used)

    assert_equal(move, 0)


def test_model_move_is_ignored_if_illegal():
    game = create_new_game()
    game["board"][0] = "O"
    game["board"][4] = "X"

    model = {
        "O...X....": 0
    }

    def fallback_strategy(game):
        return 8

    move = choose_model_move(game, model, fallback_strategy)

    assert_equal(move, 8)


def test_fallback_is_used_when_state_unknown():
    game = create_new_game()
    game["board"][4] = "X"

    model = {}

    def fallback_strategy(game):
        return 0

    move = choose_model_move(game, model, fallback_strategy)

    assert_equal(move, 0)


TESTS = [
    ("L'IA choisit un coup légal", test_ai_move_is_legal),
    ("L'IA joue la seule case restante", test_ai_plays_only_remaining_cell),
    ("Trouver un coup gagnant pour X", test_find_winning_move_for_x),
    ("Trouver un coup gagnant pour O", test_find_winning_move_for_o),
    ("Ne pas inventer de coup gagnant", test_find_winning_move_returns_none_when_no_win_available),
    ("L'IA joue un coup gagnant si possible", test_ai_takes_winning_move),
    ("L'IA bloque une victoire humaine immédiate", test_ai_blocks_human_winning_move),
    ("Le coup aléatoire reste légal", test_random_move_is_legal),
    ("Utiliser le modèle quand l'état est connu", test_model_move_is_used_when_known),
    ("Ignorer un coup de modèle devenu illégal", test_model_move_is_ignored_if_illegal),
    ("Utiliser le fallback quand l'état est inconnu", test_fallback_is_used_when_state_unknown),
]
