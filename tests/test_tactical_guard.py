from tests.test_helpers import assert_equal, assert_true

from app.ai.tactical_guard import (
    TACTICAL_GUARD_BLOCK_REASON,
    TACTICAL_GUARD_WIN_REASON,
    count_opponent_fork_replies_after_move,
    count_opponent_forcing_fork_replies_after_move,
    filter_move_scores_to_safe_moves,
    find_tactical_guard_move,
    find_tactical_guard_result,
    get_safe_moves_against_immediate_loss,
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


def test_safe_moves_remove_immediate_blunders():
    game = create_game_with_board([
        "X", "X", None,
        "O", None, None,
        None, "O", None,
    ])

    safe_moves = get_safe_moves_against_immediate_loss(game)

    assert_equal(safe_moves, [2])


def test_filter_move_scores_keeps_only_safe_scores_when_possible():
    game = create_game_with_board([
        "X", "X", None,
        "O", None, None,
        None, "O", None,
    ])
    scores = {1: 0.9, 2: 0.1, 5: 0.8}

    filtered_scores = filter_move_scores_to_safe_moves(game, scores)

    assert_equal(filtered_scores, {2: 0.1})


def test_fork_risk_counts_opponent_fork_replies():
    game = create_game_with_board([
        "X", None, None,
        None, "O", None,
        None, "X", None,
    ])

    assert_equal(count_opponent_fork_replies_after_move(game, 1), 1)
    assert_equal(count_opponent_fork_replies_after_move(game, 3), 0)


def test_filter_move_scores_keeps_lowest_fork_risk_when_possible():
    game = create_game_with_board([
        "X", None, None,
        None, "O", None,
        None, "X", None,
    ])
    scores = {1: 0.9, 2: 0.8, 3: 0.1, 5: 0.7, 6: 0.6, 8: 0.5}

    filtered_scores = filter_move_scores_to_safe_moves(game, scores)

    assert_equal(filtered_scores, {3: 0.1, 6: 0.6, 8: 0.5})


def test_forced_fork_risk_detects_two_step_trap():
    game = create_game_with_board([
        None, None, None,
        "X", None, None,
        None, None, None,
    ])

    assert_equal(count_opponent_forcing_fork_replies_after_move(game, 8), 2)
    assert_equal(count_opponent_forcing_fork_replies_after_move(game, 4), 0)


def test_filter_move_scores_removes_two_step_fork_traps():
    game = create_game_with_board([
        None, None, None,
        "X", None, None,
        None, None, None,
    ])
    scores = {0: 0.1, 1: 0.8, 2: 0.7, 4: 0.2, 5: 0.3, 6: 0.4, 7: 0.6, 8: 0.9}

    filtered_scores = filter_move_scores_to_safe_moves(game, scores)

    assert_equal(filtered_scores, {0: 0.1, 4: 0.2, 5: 0.3, 6: 0.4})


TESTS = [
    ("La garde tactique préfère gagner immédiatement", test_tactical_guard_prefers_immediate_win),
    ("La garde tactique bloque une victoire adverse immédiate", test_tactical_guard_blocks_immediate_loss),
    ("La garde tactique ne force rien sans coup obligatoire", test_tactical_guard_returns_none_without_forced_move),
    ("Les probes tactiques par défaut sont cohérentes avec la garde", test_default_tactical_probes_are_consistent_with_guard),
    ("La garde repère les coups qui perdent immédiatement", test_safe_moves_remove_immediate_blunders),
    ("La garde filtre les scores vers les coups sûrs", test_filter_move_scores_keeps_only_safe_scores_when_possible),
    ("La garde compte les réponses adverses qui créent une fourchette", test_fork_risk_counts_opponent_fork_replies),
    ("La garde préfère les coups qui minimisent les fourchettes", test_filter_move_scores_keeps_lowest_fork_risk_when_possible),
    ("La garde détecte un piège forcé en deux temps", test_forced_fork_risk_detects_two_step_trap),
    ("La garde filtre les pièges forcés en deux temps", test_filter_move_scores_removes_two_step_fork_traps),
]
