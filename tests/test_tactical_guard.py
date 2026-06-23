from tests.test_helpers import assert_equal

from app.ai.tactical_guard import (
    TACTICAL_GUARD_BLOCK_REASON,
    TACTICAL_GUARD_WIN_REASON,
    find_tactical_guard_move,
    find_tactical_guard_result,
)
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


TESTS = [
    ("La garde tactique préfère gagner immédiatement", test_tactical_guard_prefers_immediate_win),
    ("La garde tactique bloque une victoire adverse immédiate", test_tactical_guard_blocks_immediate_loss),
    ("La garde tactique ne force rien sans coup obligatoire", test_tactical_guard_returns_none_without_forced_move),
]
