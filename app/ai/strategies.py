import random

from app.games.morpion.rules import (
    copy_game,
    encode_game_state,
    get_legal_moves,
    get_winner,
)

random.seed(0)


TRAINED_MODEL = {
    # Cette table est remplie quand on charge le modèle entraîné.
    # Format :
    # "....X...." -> 0
}


def find_winning_move(game, player):
    legal_moves = get_legal_moves(game)

    for move in legal_moves:
        simulated_game = copy_game(game)
        simulated_game["board"][move] = player

        if get_winner(simulated_game) == player:
            return move

    return None


def choose_random_move(game):
    legal_moves = get_legal_moves(game)
    return random.choice(legal_moves)


def choose_fallback_move(game):
    winning_move = find_winning_move(game, "O")
    if winning_move is not None:
        return winning_move

    blocking_move = find_winning_move(game, "X")
    if blocking_move is not None:
        return blocking_move

    return choose_random_move(game)


def choose_model_move(game, model, fallback_strategy=choose_fallback_move):
    state_key = encode_game_state(game)
    legal_moves = get_legal_moves(game)

    if state_key in model:
        model_move = model[state_key]

        if model_move in legal_moves:
            return model_move

    return fallback_strategy(game)


def choose_ai_move(game):
    return choose_model_move(game, TRAINED_MODEL)