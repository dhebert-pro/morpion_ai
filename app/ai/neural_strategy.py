from app.ai.neural_encoding import (
    encode_state_key_as_neural_input,
    validate_output_index,
)

from app.ai.neural_network import SimpleNeuralNetwork

from app.games.morpion.adapter import MORPION_ADAPTER


def predict_legal_move_scores(
    game,
    model_data,
    game_adapter=MORPION_ADAPTER,
):
    """Prédit un score pour chaque coup légal d'un état.

    Le réseau produit un score pour toutes les sorties possibles.
    Cette fonction ne conserve que les coups réellement légaux dans l'état actuel.
    """

    network = SimpleNeuralNetwork.from_dict(model_data)

    state_key = game_adapter.encode_game_state(game)
    inputs = encode_state_key_as_neural_input(
        state_key=state_key,
        trained_player=game_adapter.trained_player,
        opponent_player=game_adapter.opponent_player,
    )

    predictions = network.predict(inputs)
    legal_move_scores = {}

    for move in game_adapter.get_legal_moves(game):
        output_index = game_adapter.move_to_index(move)
        validate_output_index(output_index, game_adapter.output_size)

        legal_move_scores[move] = predictions[output_index]

    return legal_move_scores


def choose_best_predicted_move(move_scores):
    """Choisit le coup au meilleur score prédit.

    Retourne None si aucun coup n'est disponible.
    """

    if len(move_scores) == 0:
        return None

    best_move = None
    best_score = None

    for move, score in move_scores.items():
        if best_score is None or score > best_score:
            best_move = move
            best_score = score

    return best_move


def choose_neural_move(
    game,
    model_data,
    game_adapter=MORPION_ADAPTER,
    fallback_strategy=None,
):
    """Choisit un coup avec le réseau neuronal.

    Si aucun modèle n'est fourni, on peut utiliser une stratégie de secours.
    Si aucun coup légal n'existe, retourne None.
    """

    legal_moves = game_adapter.get_legal_moves(game)

    if len(legal_moves) == 0:
        return None

    if not model_data:
        if fallback_strategy is None:
            return None

        return fallback_strategy(game)

    move_scores = predict_legal_move_scores(
        game=game,
        model_data=model_data,
        game_adapter=game_adapter,
    )

    return choose_best_predicted_move(move_scores)