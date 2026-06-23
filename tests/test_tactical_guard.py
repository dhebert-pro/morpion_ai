from tests.test_helpers import assert_equal, assert_true

from app.ai.tactical_guard import (
    TACTICAL_GUARD_BLOCK_REASON,
    TACTICAL_GUARD_WIN_REASON,
    find_tactical_guard_move,
    find_tactical_guard_result,
)
from app.ai.morpion_tactical_probes import create_default_morpion_tactical_probes
from app.ai.tactical_evaluation import create_game_from_board
from app.games.morpion.rules import create_new_game


def create_game_with_board(board):
    game = create_new_game()
    game["board"] = board.copy()
    return game


def test_tactical_guard_prefers_immediate_win():
    game = create_game_with_board([
        "X", "X", None,
        "O", "O", None,
        None, None, None,
    ])

    result = find_tactical_guard_result(game)

    assert_equal(result["move"], 5)
    assert_equal(result["reason"], TACTICAL_GUARD_WIN_REASON)


def test_tactical_guard_blocks_immediate_loss():
    game = create_game_with_board([
        "X", "X", None,
        "O", None, None,
        None, "O", None,
    ])

    result = find_tactical_guard_result(game)

    assert_equal(result["move"], 2)
    assert_equal(result["reason"], TACTICAL_GUARD_BLOCK_REASON)


def test_tactical_guard_returns_none_without_forced_move():
    game = create_game_with_board([
        "X", None, None,
        None, "O", None,
        None, None, None,
    ])

    move = find_tactical_guard_move(game)

    assert_equal(move, None)


def test_default_tactical_probes_are_consistent_with_guard():
    probes = create_default_morpion_tactical_probes()

    for probe in probes:
        if not probe["name"].startswith(("win_line_", "block_line_")):
            continue

        game = create_game_from_board(probe["board"])
        result = find_tactical_guard_result(game)

        assert_true(result is not None)
        assert_true(result["move"] in probe["expected_moves"])

        if probe["name"].startswith("win_line_"):
            assert_equal(result["reason"], TACTICAL_GUARD_WIN_REASON)

        if probe["name"].startswith("block_line_"):
            assert_equal(result["reason"], TACTICAL_GUARD_BLOCK_REASON)


TESTS = [
    ("La garde tactique préfère gagner immédiatement", test_tactical_guard_prefers_immediate_win),
    ("La garde tactique bloque une victoire adverse immédiate", test_tactical_guard_blocks_immediate_loss),
    ("La garde tactique ne force rien sans coup obligatoire", test_tactical_guard_returns_none_without_forced_move),
    ("Les probes tactiques par défaut sont cohérentes avec la garde", test_default_tactical_probes_are_consistent_with_guard),
]
