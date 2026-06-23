import random

from app.ai.strategies import find_winning_move, choose_random_move
from app.games.morpion.adapter import MORPION_ADAPTER


REFERENCE_RANDOM = "random"
REFERENCE_TACTICAL = "tactical"
REFERENCE_GREEDY = "greedy"


def score_result_for_player(result, player, game_adapter=MORPION_ADAPTER):
    if result == player:
        return 1.0

    if result == "draw":
        return 0.5

    if result == "ongoing":
        return 0.5

    return 0.0


def choose_greedy_immediate_move(
    game,
    player,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    legal_moves = game_adapter.get_legal_moves(game)

    if len(legal_moves) == 0:
        return None

    best_score = None
    best_moves = []

    for move in legal_moves:
        simulated_game = game_adapter.copy_game(game)
        game_adapter.apply_move(simulated_game, move, player)
        result = game_adapter.get_game_result(simulated_game)
        score = score_result_for_player(result, player, game_adapter)

        if best_score is None or score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)

    if rng is None:
        return random.choice(best_moves)

    return rng.choice(best_moves)


def choose_tactical_reference_move(
    game,
    player,
    opponent,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    winning_move = find_winning_move(game, player, game_adapter)

    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(game, opponent, game_adapter)

    if blocking_move is not None:
        return blocking_move

    return choose_random_move(game, game_adapter, rng=rng)


def choose_reference_opponent_move(
    game,
    player,
    opponent,
    reference_name=REFERENCE_RANDOM,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    if reference_name == REFERENCE_RANDOM:
        return choose_random_move(game, game_adapter, rng=rng)

    if reference_name == REFERENCE_TACTICAL:
        return choose_tactical_reference_move(
            game,
            player,
            opponent,
            game_adapter,
            rng=rng,
        )

    if reference_name == REFERENCE_GREEDY:
        return choose_greedy_immediate_move(
            game,
            player,
            game_adapter,
            rng=rng,
        )

    raise ValueError("Adversaire de référence inconnu : " + str(reference_name))


def get_default_reference_opponents():
    return [
        REFERENCE_RANDOM,
        REFERENCE_TACTICAL,
        REFERENCE_GREEDY,
    ]
