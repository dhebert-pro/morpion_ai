from app.ai.tactical_risk import (
    count_opponent_fork_replies_after_move,
    count_opponent_forcing_fork_replies_after_move,
    move_allows_immediate_opponent_win,
)
from app.games.morpion.adapter import MORPION_ADAPTER


def get_safe_moves_against_immediate_loss(
    game,
    game_adapter=MORPION_ADAPTER,
):
    safe_moves = []

    for move in game_adapter.get_legal_moves(game):
        if not move_allows_immediate_opponent_win(
            game,
            move,
            game_adapter,
        ):
            safe_moves.append(move)

    return safe_moves


def filter_move_scores_to_safe_moves(
    game,
    move_scores,
    game_adapter=MORPION_ADAPTER,
):
    if len(move_scores) == 0:
        return move_scores

    safe_scores = filter_scores_to_immediate_safe_moves(
        game,
        move_scores,
        game_adapter,
    )
    low_fork_scores = filter_scores_to_lowest_risk(
        game,
        safe_scores,
        count_opponent_fork_replies_after_move,
        game_adapter,
    )
    return filter_scores_to_lowest_risk(
        game,
        low_fork_scores,
        count_opponent_forcing_fork_replies_after_move,
        game_adapter,
    )


def filter_scores_to_immediate_safe_moves(
    game,
    move_scores,
    game_adapter=MORPION_ADAPTER,
):
    safe_moves = get_safe_moves_against_immediate_loss(game, game_adapter)

    if len(safe_moves) == 0:
        return move_scores

    safe_scores = {}

    for move, score in move_scores.items():
        if move in safe_moves:
            safe_scores[move] = score

    if len(safe_scores) == 0:
        return move_scores

    return safe_scores


def filter_scores_to_lowest_risk(
    game,
    move_scores,
    risk_counter,
    game_adapter=MORPION_ADAPTER,
):
    if len(move_scores) <= 1:
        return move_scores

    risks = {}
    lowest_risk = None

    for move in move_scores:
        risk = risk_counter(game, move, game_adapter)
        risks[move] = risk

        if lowest_risk is None or risk < lowest_risk:
            lowest_risk = risk

    return {
        move: score
        for move, score in move_scores.items()
        if risks[move] == lowest_risk
    }
