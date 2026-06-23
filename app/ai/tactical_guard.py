from app.ai.strategies import find_winning_move
from app.ai.tactical_move_filter import (
    filter_move_scores_to_safe_moves,
    get_safe_moves_against_immediate_loss,
)
from app.ai.tactical_risk import (
    count_fork_moves_for_player,
    count_immediate_winning_moves_for_player,
    count_opponent_fork_replies_after_move,
    count_opponent_forcing_fork_replies_after_move,
    move_allows_immediate_opponent_win,
    opponent_reply_forces_fork_after_defense,
)
from app.games.morpion.adapter import MORPION_ADAPTER


TACTICAL_GUARD_WIN_REASON = "immediate_win"
TACTICAL_GUARD_BLOCK_REASON = "immediate_block"
TACTICAL_GUARD_SAFE_REASON = "avoid_immediate_loss"
TACTICAL_GUARD_FORK_REASON = "avoid_opponent_fork"
TACTICAL_GUARD_FORCED_FORK_REASON = "avoid_forced_fork"


def find_tactical_guard_move(game, game_adapter=MORPION_ADAPTER):
    """Trouve un coup tactique forcé avant d'utiliser le réseau."""

    result = find_tactical_guard_result(game, game_adapter)

    if result is None:
        return None

    return result["move"]


def find_tactical_guard_result(game, game_adapter=MORPION_ADAPTER):
    winning_move = find_winning_move(
        game,
        game_adapter.trained_player,
        game_adapter,
    )

    if winning_move is not None:
        return {
            "move": winning_move,
            "reason": TACTICAL_GUARD_WIN_REASON,
        }

    blocking_move = find_winning_move(
        game,
        game_adapter.opponent_player,
        game_adapter,
    )

    if blocking_move is not None:
        return {
            "move": blocking_move,
            "reason": TACTICAL_GUARD_BLOCK_REASON,
        }

    return None
