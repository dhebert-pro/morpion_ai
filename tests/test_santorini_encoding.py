from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.coordinates import CELL_COUNT, cell_to_index
from app.games.santorini.encoding import (
    SANTORINI_INPUT_SIZE,
    encode_santorini_state,
    get_input_plane_slice,
)
from app.games.santorini.rules import create_new_game, place_worker


def c(text):
    return cell_to_index(text)


def setup_game():
    game = create_new_game()
    for text in ["A1", "E1", "A5", "E5"]:
        place_worker(game, c(text))
    return game


def test_santorini_input_size_is_fixed():
    game = setup_game()
    vector = encode_santorini_state(game, "O")
    assert_equal(len(vector), SANTORINI_INPUT_SIZE)
    assert_equal(SANTORINI_INPUT_SIZE, CELL_COUNT * 9)


def test_height_plane_is_normalized():
    game = setup_game()
    game["heights"][c("C3")] = 3
    vector = encode_santorini_state(game, "O")
    start, _ = get_input_plane_slice("heights")
    assert_equal(vector[start + c("C3")], 1.0)


def test_worker_slots_are_encoded_from_player_perspective():
    game = setup_game()
    vector = encode_santorini_state(game, "O")
    own_start, _ = get_input_plane_slice("own_worker_1")
    opponent_start, _ = get_input_plane_slice("opponent_worker_1")
    assert_equal(vector[own_start + c("A5")], 1.0)
    assert_equal(vector[opponent_start + c("A1")], 1.0)


def test_winning_destinations_are_encoded():
    game = setup_game()
    game["heights"][c("A5")] = 2
    game["heights"][c("B4")] = 3
    vector = encode_santorini_state(game, "O")
    start, _ = get_input_plane_slice("own_winning_destinations")
    assert_equal(vector[start + c("B4")], 1.0)


def test_legal_destinations_plane_is_not_empty_in_play_phase():
    game = setup_game()
    vector = encode_santorini_state(game, "O")
    start, end = get_input_plane_slice("own_legal_destinations")
    assert_true(sum(vector[start:end]) > 0.0)


TESTS = [
    ("Santorini fixe la taille d'entrée", test_santorini_input_size_is_fixed),
    ("Santorini normalise les hauteurs", test_height_plane_is_normalized),
    ("Santorini encode les ouvriers par perspective", test_worker_slots_are_encoded_from_player_perspective),
    ("Santorini encode les destinations gagnantes", test_winning_destinations_are_encoded),
    ("Santorini encode les destinations légales", test_legal_destinations_plane_is_not_empty_in_play_phase),
]
