from app.ai.strategies import find_winning_move
from app.games.morpion.adapter import MORPION_ADAPTER


TACTICAL_GUARD_WIN_REASON = "immediate_win"
TACTICAL_GUARD_BLOCK_REASON = "immediate_block"


def find_tactical_guard_move(game, game_adapter=MORPION_ADAPTER):
    """Trouve un coup tactique forcé avant d'utiliser le réseau.

    Priorité :
    1. gagner immédiatement ;
    2. bloquer une victoire immédiate adverse.

    Retourne None si aucun coup forcé n'est trouvé.
    """

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
