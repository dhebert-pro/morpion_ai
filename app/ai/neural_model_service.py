from app.ai.neural_pipeline import (
    train_neural_model_in_memory,
)

from app.ai.neural_evaluation import (
    evaluate_neural_model,
    summarize_neural_evaluation_results,
)

from app.storage.json_storage import (
    save_json,
    load_json,
)

from app.games.morpion.adapter import MORPION_ADAPTER
from app.ai.neural_encoding import encode_state_key_as_neural_input


def create_neural_model_package(
    training_result,
    game_adapter=MORPION_ADAPTER,
):
    """Prépare les données à sauvegarder dans neural_model.json.

    On ne sauvegarde pas seulement les poids du réseau.
    On garde aussi un résumé d'entraînement, pour savoir comment le modèle
    a été produit.
    """

    return {
        "type": "neural_model_package",
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "model_data": training_result["model_data"],
        "training_summary": training_result["summary"],
    }


def train_and_save_neural_model(
    file_path,
    training_games_count,
    simulations_per_move,
    max_examples,
    hidden_size,
    epochs,
    learning_rate,
    tactical_repeat_count=0,
    show_progress=False,
    seed=0,
    game_adapter=MORPION_ADAPTER,
    initial_model_data=None,
):
    """Entraîne un modèle neuronal puis le sauvegarde.

    Si initial_model_data est fourni :
    - l'entraînement continue depuis les poids existants.

    Si initial_model_data vaut None :
    - l'entraînement repart de zéro.

    tactical_repeat_count permet d'ajouter des exemples tactiques forcés.
    """

    training_result = train_neural_model_in_memory(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        seed=seed,
        game_adapter=game_adapter,
        initial_model_data=initial_model_data,
    )

    model_package = create_neural_model_package(
        training_result,
        game_adapter,
    )

    save_json(
        model_package,
        file_path,
    )

    return model_package


def load_neural_model_package(file_path):
    """Charge un modèle neuronal sauvegardé.

    Retourne {} si aucun fichier n'existe.
    """

    return load_json(file_path)


def get_model_data_from_package(model_package):
    if not model_package:
        return {}

    return model_package.get("model_data", {})


def train_and_save_neural_model_from_package(
    file_path,
    existing_model_package,
    training_games_count,
    simulations_per_move,
    max_examples,
    hidden_size,
    epochs,
    learning_rate,
    tactical_repeat_count=0,
    show_progress=False,
    seed=0,
    game_adapter=MORPION_ADAPTER,
):
    """Continue l'entraînement depuis un package existant.

    Si le package est vide ou invalide, l'entraînement repart de zéro.
    """

    initial_model_data = get_model_data_from_package(
        existing_model_package,
    )

    if not initial_model_data:
        initial_model_data = None
    elif not is_model_data_compatible_with_training(
        initial_model_data,
        hidden_size,
        game_adapter,
    ):
        initial_model_data = None

    return train_and_save_neural_model(
        file_path=file_path,
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        seed=seed,
        game_adapter=game_adapter,
        initial_model_data=initial_model_data,
    )


def evaluate_saved_neural_model_package(
    model_package,
    games_count,
    game_adapter=MORPION_ADAPTER,
):
    """Évalue un package de modèle neuronal déjà chargé."""

    model_data = get_model_data_from_package(model_package)

    if not model_data:
        return {
            "results": {},
            "summary": {
                "total_games": 0,
                "trained_player": game_adapter.trained_player,
                "opponent_player": game_adapter.opponent_player,
                "trained_player_wins": 0,
                "opponent_player_wins": 0,
                "draws": 0,
                "efficiency": 0.0,
            },
        }

    results = evaluate_neural_model(
        model_data=model_data,
        games_count=games_count,
        game_adapter=game_adapter,
    )

    summary = summarize_neural_evaluation_results(
        results,
        trained_player=game_adapter.trained_player,
        opponent_player=game_adapter.opponent_player,
    )

    return {
        "results": results,
        "summary": summary,
    }

def is_model_data_compatible_with_training(
    model_data,
    hidden_size,
    game_adapter=MORPION_ADAPTER,
):
    if not model_data:
        return False

    expected_input_size = get_current_neural_input_size(game_adapter)

    if model_data.get("input_size") != expected_input_size:
        return False

    if model_data.get("hidden_size") != hidden_size:
        return False

    if model_data.get("output_size") != game_adapter.output_size:
        return False

    return True


def get_current_neural_input_size(game_adapter=MORPION_ADAPTER):
    empty_game = game_adapter.create_new_game()
    empty_state_key = game_adapter.encode_game_state(empty_game)

    return len(
        encode_state_key_as_neural_input(
            empty_state_key,
            game_adapter.trained_player,
            game_adapter.opponent_player,
        )
    )
