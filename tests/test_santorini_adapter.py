from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.adapter import SANTORINI_ADAPTER
from app.games.santorini.coordinates import cell_to_index
from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.rules import place_worker


def c(text):
    return cell_to_index(text)


def setup_game():
    game = SANTORINI_ADAPTER.create_new_game()
    for text in ["A1", "E1", "A5", "E5"]:
        place_worker(game, c(text))
    return game


def test_santorini_adapter_exposes_basic_contract():
    assert_equal(SANTORINI_ADAPTER.name, "santorini")
    assert_equal(SANTORINI_ADAPTER.output_size, 144)
    assert_equal(SANTORINI_ADAPTER.trained_player, "O")


def test_santorini_adapter_lists_indexed_legal_moves():
    game = setup_game()
    moves = SANTORINI_ADAPTER.get_legal_moves(game)
    assert_true(len(moves) > 0)
    assert_true("output_index" in moves[0])
    assert_true(0 <= SANTORINI_ADAPTER.move_to_index(moves[0]) < 144)


def test_santorini_adapter_can_apply_indexed_move():
    game = setup_game()
    move = get_indexed_legal_actions(game)[0]
    current_player = game["current_player"]
    assert_true(SANTORINI_ADAPTER.is_valid_move(game, move))
    SANTORINI_ADAPTER.apply_move(game, move, current_player)
    assert_equal(game["current_player"], "O")


def test_santorini_adapter_state_key_changes_after_move():
    game = setup_game()
    before = SANTORINI_ADAPTER.encode_game_state(game)
    move = get_indexed_legal_actions(game)[0]
    SANTORINI_ADAPTER.apply_move(game, move, game["current_player"])
    after = SANTORINI_ADAPTER.encode_game_state(game)
    assert_true(before != after)


TESTS = [
    ("Santorini adapter expose le contrat", test_santorini_adapter_exposes_basic_contract),
    ("Santorini adapter liste des coups indexés", test_santorini_adapter_lists_indexed_legal_moves),
    ("Santorini adapter applique un coup indexé", test_santorini_adapter_can_apply_indexed_move),
    ("Santorini adapter change la clé d'état", test_santorini_adapter_state_key_changes_after_move),
]
