from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.action_index import (
    ACTION_OUTPUT_SIZE,
    action_to_output_index,
    decode_output_index,
    output_index_to_action,
)
from app.games.santorini.coordinates import cell_to_index
from app.games.santorini.rules import create_new_game, place_worker


def c(text):
    return cell_to_index(text)


def setup_game():
    game = create_new_game()
    for text in ["A1", "E1", "A5", "E5"]:
        place_worker(game, c(text))
    return game


def test_action_output_size_is_fixed():
    assert_equal(ACTION_OUTPUT_SIZE, 144)


def test_action_to_output_index_round_trip():
    game = setup_game()
    action = {"from": c("A1"), "to": c("B2"), "build": c("B1")}
    index = action_to_output_index(game, action)
    decoded_action = output_index_to_action(game, index)
    assert_equal(decoded_action, action)


def test_action_to_output_index_handles_winning_move_without_build():
    game = setup_game()
    game["heights"][c("A1")] = 2
    game["heights"][c("B2")] = 3
    action = {"from": c("A1"), "to": c("B2"), "build": None}
    index = action_to_output_index(game, action)
    assert_true(index is not None)
    assert_equal(output_index_to_action(game, index), action)


def test_decode_rejects_invalid_index():
    assert_equal(decode_output_index(-1), None)
    assert_equal(decode_output_index(ACTION_OUTPUT_SIZE), None)


TESTS = [
    ("Santorini fixe 144 sorties d'action", test_action_output_size_is_fixed),
    ("Santorini encode puis décode un coup", test_action_to_output_index_round_trip),
    ("Santorini encode un coup gagnant sans construction", test_action_to_output_index_handles_winning_move_without_build),
    ("Santorini refuse un index de sortie invalide", test_decode_rejects_invalid_index),
]
