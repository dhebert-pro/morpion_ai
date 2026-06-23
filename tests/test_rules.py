from tests.test_helpers import assert_equal, assert_true, assert_false

from app.games.morpion.rules import (
    WINNING_LINES,
    create_new_game,
    copy_game,
    encode_game_state,
    parse_human_input,
    switch_player,
    is_valid_move,
    get_legal_moves,
    get_winner,
    get_game_result,
    get_score_for_o,
)


def test_create_new_game():
    game = create_new_game()

    assert_equal(game["current_player"], "X")
    assert_equal(len(game["board"]), 9)
    assert_equal(game["board"], [None, None, None, None, None, None, None, None, None])


def test_parse_human_input():
    assert_equal(parse_human_input("0"), 0)
    assert_equal(parse_human_input("4"), 4)
    assert_equal(parse_human_input("8"), 8)
    assert_equal(parse_human_input(" 4 "), 4)
    assert_equal(parse_human_input("q"), "quit")
    assert_equal(parse_human_input("Q"), "quit")
    assert_equal(parse_human_input(" q "), "quit")
    assert_equal(parse_human_input("salut"), None)
    assert_equal(parse_human_input(""), None)
    assert_equal(parse_human_input("   "), None)
    assert_equal(parse_human_input("4.5"), None)
    assert_equal(parse_human_input("-1"), -1)


def test_all_empty_cells_are_valid_at_start():
    game = create_new_game()

    for move in range(9):
        assert_true(is_valid_move(game, move))


def test_moves_outside_board_are_invalid():
    game = create_new_game()

    assert_false(is_valid_move(game, -100))
    assert_false(is_valid_move(game, -1))
    assert_false(is_valid_move(game, 9))
    assert_false(is_valid_move(game, 10))
    assert_false(is_valid_move(game, 100))


def test_taken_cells_are_invalid():
    for move in range(9):
        game = create_new_game()
        game["board"][move] = "X"

        assert_false(is_valid_move(game, move))


def test_cells_taken_by_ai_are_invalid_too():
    for move in range(9):
        game = create_new_game()
        game["board"][move] = "O"

        assert_false(is_valid_move(game, move))


def test_get_legal_moves_empty_board():
    game = create_new_game()

    assert_equal(get_legal_moves(game), [0, 1, 2, 3, 4, 5, 6, 7, 8])


def test_get_legal_moves_partial_board():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"
    game["board"][8] = "X"

    assert_equal(get_legal_moves(game), [1, 2, 3, 5, 6, 7])


def test_get_legal_moves_full_board():
    game = create_new_game()
    game["board"] = [
        "X", "O", "X",
        "X", "O", "O",
        "O", "X", "X",
    ]

    assert_equal(get_legal_moves(game), [])


def test_winner_all_lines_for_x():
    for line in WINNING_LINES:
        game = create_new_game()

        for index in line:
            game["board"][index] = "X"

        assert_equal(get_winner(game), "X")


def test_winner_all_lines_for_o():
    for line in WINNING_LINES:
        game = create_new_game()

        for index in line:
            game["board"][index] = "O"

        assert_equal(get_winner(game), "O")


def test_no_winner_empty_board():
    game = create_new_game()

    assert_equal(get_winner(game), None)


def test_no_winner_with_only_two_aligned_symbols():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "X"

    assert_equal(get_winner(game), None)


def test_no_winner_with_mixed_line():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "O"
    game["board"][2] = "X"

    assert_equal(get_winner(game), None)


def test_no_winner_on_full_draw_board():
    game = create_new_game()
    game["board"] = [
        "X", "O", "X",
        "X", "O", "O",
        "O", "X", "X",
    ]

    assert_equal(get_winner(game), None)
    assert_equal(get_legal_moves(game), [])


def test_copy_game_creates_independent_board():
    game = create_new_game()
    copied_game = copy_game(game)

    copied_game["board"][4] = "X"

    assert_equal(game["board"][4], None)
    assert_equal(copied_game["board"][4], "X")


def test_game_result_ongoing():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"

    assert_equal(get_game_result(game), "ongoing")


def test_game_result_x_wins():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][1] = "X"
    game["board"][2] = "X"

    assert_equal(get_game_result(game), "X")


def test_game_result_o_wins():
    game = create_new_game()
    game["board"][3] = "O"
    game["board"][4] = "O"
    game["board"][5] = "O"

    assert_equal(get_game_result(game), "O")


def test_game_result_draw():
    game = create_new_game()
    game["board"] = [
        "X", "O", "X",
        "X", "O", "O",
        "O", "X", "X",
    ]

    assert_equal(get_game_result(game), "draw")


def test_encode_empty_game_state():
    game = create_new_game()

    assert_equal(encode_game_state(game), ".........")


def test_encode_partial_game_state():
    game = create_new_game()
    game["board"][0] = "X"
    game["board"][4] = "O"
    game["board"][8] = "X"

    assert_equal(encode_game_state(game), "X...O...X")


def test_switch_player():
    assert_equal(switch_player("X"), "O")
    assert_equal(switch_player("O"), "X")


def test_get_score_for_o():
    assert_equal(get_score_for_o("O"), 1.0)
    assert_equal(get_score_for_o("draw"), 0.5)
    assert_equal(get_score_for_o("X"), 0.0)


TESTS = [
    ("Créer une nouvelle partie", test_create_new_game),
    ("Lire les entrées utilisateur", test_parse_human_input),
    ("Tous les coups sont valides au début", test_all_empty_cells_are_valid_at_start),
    ("Les coups hors plateau sont invalides", test_moves_outside_board_are_invalid),
    ("Une case déjà prise est invalide", test_taken_cells_are_invalid),
    ("Une case prise par l'IA est invalide aussi", test_cells_taken_by_ai_are_invalid_too),
    ("Coups légaux sur plateau vide", test_get_legal_moves_empty_board),
    ("Coups légaux sur plateau partiel", test_get_legal_moves_partial_board),
    ("Coups légaux sur plateau plein", test_get_legal_moves_full_board),
    ("Toutes les lignes gagnantes pour X", test_winner_all_lines_for_x),
    ("Toutes les lignes gagnantes pour O", test_winner_all_lines_for_o),
    ("Pas de gagnant sur plateau vide", test_no_winner_empty_board),
    ("Pas de gagnant avec seulement deux symboles alignés", test_no_winner_with_only_two_aligned_symbols),
    ("Pas de gagnant sur une ligne mélangée", test_no_winner_with_mixed_line),
    ("Pas de gagnant sur un match nul complet", test_no_winner_on_full_draw_board),
    ("Copier une partie sans modifier l'original", test_copy_game_creates_independent_board),
    ("Résultat : partie en cours", test_game_result_ongoing),
    ("Résultat : victoire de X", test_game_result_x_wins),
    ("Résultat : victoire de O", test_game_result_o_wins),
    ("Résultat : match nul", test_game_result_draw),
    ("Encoder un plateau vide", test_encode_empty_game_state),
    ("Encoder un plateau partiel", test_encode_partial_game_state),
    ("Changer de joueur", test_switch_player),
    ("Score du résultat pour O", test_get_score_for_o),
]
