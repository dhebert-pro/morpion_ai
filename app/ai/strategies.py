import random

from app.games.morpion.adapter import MORPION_ADAPTER

random.seed(0)


TRAINED_MODEL = {
    # Cette table est remplie quand on charge le modèle entraîné.
    # Format :
    # "....X...." -> 0
}


def find_winning_move(game, player, game_adapter=MORPION_ADAPTER):
    legal_moves = game_adapter.get_legal_moves(game)

    for move in legal_moves:
        simulated_game = game_adapter.copy_game(game)
        game_adapter.apply_move(simulated_game, move, player)

        if game_adapter.get_winner(simulated_game) == player:
            return move

    return None


def choose_random_move(game, game_adapter=MORPION_ADAPTER, rng=None):
    legal_moves = game_adapter.get_legal_moves(game)

    if rng is not None:
        return rng.choice(legal_moves)

    return random.choice(legal_moves)


def choose_fallback_move(game, game_adapter=MORPION_ADAPTER):
    winning_move = find_winning_move(game, game_adapter.trained_player, game_adapter)
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(game, game_adapter.opponent_player, game_adapter)
    if blocking_move is not None:
        return blocking_move

    return choose_random_move(game, game_adapter)


def choose_model_move(
    game,
    model,
    fallback_strategy=choose_fallback_move,
    game_adapter=MORPION_ADAPTER,
):
    state_key = game_adapter.encode_game_state(game)
    legal_moves = game_adapter.get_legal_moves(game)

    if state_key in model:
        model_move = model[state_key]

        if model_move in legal_moves:
            return model_move

    if fallback_strategy == choose_fallback_move:
        return choose_fallback_move(game, game_adapter)

    return fallback_strategy(game)


def choose_ai_move(game, game_adapter=MORPION_ADAPTER):
    return choose_model_move(
        game,
        TRAINED_MODEL,
        fallback_strategy=choose_fallback_move,
        game_adapter=game_adapter,
    )