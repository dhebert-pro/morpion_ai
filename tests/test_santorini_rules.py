from tests.test_helpers import assert_equal, assert_true, assert_false

from app.games.santorini.coordinates import cell_to_index
from app.games.santorini.rules import (
    create_new_game,
    copy_game,
    get_game_result,
    get_legal_actions,
    is_valid_action,
    is_valid_placement,
    place_worker,
    apply_action,
)


def c(text):
    return cell_to_index(text)


def setup_standard_game():
    game = create_new_game()
    place_worker(game, c("A1"))
    place_worker(game, c("E1"))
    place_worker(game, c("A5"))
    place_worker(game, c("E5"))
    return game


def action(from_cell, to_cell, build_cell=None):
    return {"from": c(from_cell), "to": c(to_cell), "build": c(build_cell) if build_cell else None}


def test_create_new_santorini_game():
    game = create_new_game()
    assert_equal(len(game["heights"]), 25)
    assert_equal(game["workers"], {"X": [None, None], "O": [None, None]})
    assert_equal(game["phase"], "placement")
    assert_equal(game["current_player"], "X")


def test_placement_order_and_phase_change():
    game = create_new_game()
    assert_true(is_valid_placement(game, c("A1")))
    place_worker(game, c("A1"))
    assert_equal(game["current_player"], "X")
    place_worker(game, c("B1"))
    assert_equal(game["current_player"], "O")
    place_worker(game, c("C1"))
    place_worker(game, c("D1"))
    assert_equal(game["phase"], "play")
    assert_equal(game["current_player"], "X")


def test_cannot_place_on_occupied_cell():
    game = create_new_game()
    place_worker(game, c("A1"))
    assert_false(is_valid_placement(game, c("A1")))


def test_valid_action_moves_then_builds():
    game = setup_standard_game()
    move = action("A1", "B2", "B1")
    assert_true(is_valid_action(game, move))
    apply_action(game, move)
    assert_equal(game["workers"]["X"], [c("B2"), c("E1")])
    assert_equal(game["heights"][c("B1")], 1)
    assert_equal(game["current_player"], "O")


def test_cannot_move_up_more_than_one_level():
    game = setup_standard_game()
    game["heights"][c("B2")] = 2
    assert_false(is_valid_action(game, action("A1", "B2", "B1")))


def test_can_move_down_any_number_of_levels():
    game = setup_standard_game()
    game["heights"][c("A1")] = 3
    game["heights"][c("B2")] = 0
    assert_true(is_valid_action(game, action("A1", "B2", "B1")))


def test_building_on_level_three_adds_dome():
    game = setup_standard_game()
    game["heights"][c("B1")] = 3
    apply_action(game, action("A1", "B2", "B1"))
    assert_equal(game["heights"][c("B1")], 3)
    assert_true(game["domes"][c("B1")])


def test_move_to_level_three_wins_without_build():
    game = setup_standard_game()
    game["heights"][c("A1")] = 2
    game["heights"][c("B2")] = 3
    move = action("A1", "B2")
    assert_true(is_valid_action(game, move))
    apply_action(game, move)
    assert_equal(get_game_result(game), "X")
    assert_equal(game["phase"], "finished")


def test_action_to_level_three_must_not_include_build():
    game = setup_standard_game()
    game["heights"][c("A1")] = 2
    game["heights"][c("B2")] = 3
    assert_false(is_valid_action(game, action("A1", "B2", "B1")))


def test_copy_game_is_independent():
    game = setup_standard_game()
    copied = copy_game(game)
    copied["heights"][c("A1")] = 3
    copied["workers"]["X"][0] = c("B2")
    assert_equal(game["heights"][c("A1")], 0)
    assert_equal(game["workers"]["X"][0], c("A1"))


def test_get_legal_actions_contains_expected_action():
    game = setup_standard_game()
    legal_actions = get_legal_actions(game)
    assert_true(action("A1", "B2", "B1") in legal_actions)


TESTS = [
    ("Santorini crée une nouvelle partie", test_create_new_santorini_game),
    ("Santorini respecte l'ordre de placement", test_placement_order_and_phase_change),
    ("Santorini interdit un placement occupé", test_cannot_place_on_occupied_cell),
    ("Santorini déplace puis construit", test_valid_action_moves_then_builds),
    ("Santorini interdit de monter de plus d'un niveau", test_cannot_move_up_more_than_one_level),
    ("Santorini autorise à descendre de plusieurs niveaux", test_can_move_down_any_number_of_levels),
    ("Santorini construit un dôme au-dessus du niveau 3", test_building_on_level_three_adds_dome),
    ("Santorini gagne en montant au niveau 3", test_move_to_level_three_wins_without_build),
    ("Santorini ne construit pas après une montée gagnante", test_action_to_level_three_must_not_include_build),
    ("Santorini copie l'état sans le partager", test_copy_game_is_independent),
    ("Santorini liste les actions légales", test_get_legal_actions_contains_expected_action),
]
