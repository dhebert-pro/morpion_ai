from app.ai.neural_training_session import train_network_on_encoded_dataset
from app.games.santorini.dataset import (
    build_santorini_move_score_dataset,
    summarize_santorini_dataset,
)
from app.games.santorini.neural_dataset import encode_santorini_move_score_dataset
from app.storage.json_storage import load_json, save_json


def build_train_and_save_santorini_neural_model(
    model_file,
    dataset_file,
    games_count,
    max_examples,
    simulations_per_move,
    hidden_size,
    epochs,
    learning_rate,
    seed=0,
    show_progress=False,
):
    raw_dataset = build_santorini_move_score_dataset(
        games_count=games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        seed=seed,
        show_progress=show_progress,
    )
    encoded_dataset = encode_santorini_move_score_dataset(raw_dataset)

    save_json(raw_dataset, dataset_file)

    training_result = train_network_on_encoded_dataset(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        show_progress=show_progress,
    )

    package = create_santorini_model_package(
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        training_result=training_result,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )
    save_json(package, model_file)
    return package


def create_santorini_model_package(
    raw_dataset,
    encoded_dataset,
    training_result,
    hidden_size,
    epochs,
    learning_rate,
):
    return {
        "type": "neural_model_package",
        "game": "santorini",
        "trained_player": "O",
        "model_data": training_result["model_data"],
        "training_summary": create_santorini_training_summary(
            raw_dataset,
            encoded_dataset,
            training_result,
            hidden_size,
            epochs,
            learning_rate,
        ),
    }


def create_santorini_training_summary(
    raw_dataset,
    encoded_dataset,
    training_result,
    hidden_size,
    epochs,
    learning_rate,
):
    summary = summarize_santorini_dataset(raw_dataset)

    return {
        "game": "santorini",
        "training_games_count": raw_dataset.get("training_games_count", 0),
        "simulations_per_move": raw_dataset.get("simulations_per_move", 0),
        "available_states_count": raw_dataset.get("available_states_count", 0),
        "examples_count": encoded_dataset.get("encoded_examples_count", 0),
        "scored_moves_count": summary["scored_moves_count"],
        "average_legal_moves": summary["average_legal_moves"],
        "average_best_score": summary["average_best_score"],
        "average_score_spread": summary["average_score_spread"],
        "decisive_examples_count": summary["decisive_examples_count"],
        "input_size": training_result["input_size"],
        "hidden_size": hidden_size,
        "output_size": training_result["output_size"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "initial_error": training_result["initial_error"],
        "final_error": training_result["final_error"],
        "error_improvement": training_result["initial_error"] - training_result["final_error"],
    }


def load_santorini_model_package(model_file):
    return load_json(model_file)


def get_santorini_model_data(package):
    if not package:
        return {}

    if package.get("game") != "santorini":
        return {}

    return package.get("model_data", {})


def format_santorini_training_summary(summary):
    lines = []
    lines.append("Résumé entraînement neuronal Santorini")
    lines.append("Parties simulées : " + str(summary["training_games_count"]))
    lines.append("Simulations par coup : " + str(summary["simulations_per_move"]))
    lines.append("États disponibles : " + str(summary["available_states_count"]))
    lines.append("Exemples : " + str(summary["examples_count"]))
    lines.append("Coups scorés : " + str(summary["scored_moves_count"]))
    lines.append("Coups légaux moyens : " + str(summary["average_legal_moves"]))
    lines.append("Score moyen du meilleur coup : " + str(summary["average_best_score"]))
    lines.append("Écart moyen meilleur-pire coup : " + str(summary["average_score_spread"]))
    lines.append("Exemples réellement discriminants : " + str(summary["decisive_examples_count"]))
    lines.append("Taille entrée : " + str(summary["input_size"]))
    lines.append("Taille couche cachée : " + str(summary["hidden_size"]))
    lines.append("Taille sortie : " + str(summary["output_size"]))
    lines.append("Époques : " + str(summary["epochs"]))
    lines.append("Taux d'apprentissage : " + str(summary["learning_rate"]))
    lines.append("Erreur initiale : " + str(round(summary["initial_error"], 6)))
    lines.append("Erreur finale : " + str(round(summary["final_error"], 6)))
    lines.append("Amélioration erreur : " + str(round(summary["error_improvement"], 6)))
    return "\n".join(lines)
