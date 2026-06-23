from app.ai.training_dataset import (
    build_move_score_dataset,
    summarize_move_score_dataset,
)

from app.ai.tactical_training import (
    create_default_morpion_tactical_dataset,
    merge_move_score_datasets,
)

from app.ai.neural_encoding import encode_move_score_dataset

from app.ai.neural_training_session import (
    train_network_on_encoded_dataset,
)

from app.games.morpion.adapter import MORPION_ADAPTER


def build_augmented_move_score_dataset(
    training_games_count,
    simulations_per_move,
    max_examples,
    tactical_repeat_count=0,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
):
    """Construit le dataset utilisé par le réseau.

    Il contient :
    - des exemples générés par Monte-Carlo ;
    - éventuellement des exemples tactiques forcés.

    tactical_repeat_count :
    - 0 : aucun exemple tactique ajouté ;
    - > 0 : les positions tactiques de base sont répétées pour peser davantage.
    """

    base_dataset = build_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        show_progress=show_progress,
        game_adapter=game_adapter,
    )

    if tactical_repeat_count <= 0:
        base_dataset["base_examples_count"] = base_dataset["examples_count"]
        base_dataset["extra_examples_count"] = 0
        base_dataset["tactical_repeat_count"] = 0
        return base_dataset

    tactical_dataset = create_default_morpion_tactical_dataset(
        repeat_count=tactical_repeat_count,
        game_adapter=game_adapter,
    )

    augmented_dataset = merge_move_score_datasets(
        base_dataset,
        tactical_dataset,
    )

    augmented_dataset["tactical_repeat_count"] = tactical_repeat_count

    return augmented_dataset


def train_neural_model_in_memory(
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
    """Construit et entraîne un modèle neuronal complet en mémoire.

    Cette fonction ne sauvegarde rien.

    Si initial_model_data est fourni, l'entraînement continue depuis les poids
    existants. Sinon, un nouveau réseau est créé.

    Si tactical_repeat_count est supérieur à 0, on ajoute des exemples
    tactiques forcés au dataset avant l'encodage.
    """

    raw_dataset = build_augmented_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        game_adapter=game_adapter,
    )

    encoded_dataset = encode_move_score_dataset(
        raw_dataset,
        game_adapter=game_adapter,
    )

    training_result = train_network_on_encoded_dataset(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        show_progress=show_progress,
        seed=seed,
        initial_model_data=initial_model_data,
    )

    raw_dataset_summary = summarize_move_score_dataset(raw_dataset)

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "raw_dataset": raw_dataset,
        "encoded_dataset": encoded_dataset,
        "model_data": training_result["model_data"],
        "summary": {
            "game": game_adapter.name,
            "training_games_count": training_games_count,
            "simulations_per_move": simulations_per_move,
            "max_examples": max_examples,
            "tactical_repeat_count": raw_dataset.get("tactical_repeat_count", 0),
            "base_examples_count": raw_dataset.get(
                "base_examples_count",
                raw_dataset.get("examples_count", 0),
            ),
            "extra_examples_count": raw_dataset.get("extra_examples_count", 0),
            "examples_count": training_result["examples_count"],
            "scored_moves_count": raw_dataset_summary["scored_moves_count"],
            "average_legal_moves": raw_dataset_summary["average_legal_moves"],
            "average_best_score": raw_dataset_summary["average_best_score"],
            "input_size": training_result["input_size"],
            "hidden_size": training_result["hidden_size"],
            "output_size": training_result["output_size"],
            "epochs": training_result["epochs"],
            "learning_rate": training_result["learning_rate"],
            "started_from_existing_model": training_result["started_from_existing_model"],
            "initial_error": training_result["initial_error"],
            "final_error": training_result["final_error"],
            "error_improvement": training_result["initial_error"] - training_result["final_error"],
        },
    }


def format_neural_training_summary(summary):
    """Prépare un résumé lisible d'un entraînement neuronal en mémoire."""

    started_from_existing_model = summary.get(
        "started_from_existing_model",
        False,
    )
    tactical_repeat_count = summary.get(
        "tactical_repeat_count",
        0,
    )
    base_examples_count = summary.get(
        "base_examples_count",
        summary.get("examples_count", 0),
    )
    extra_examples_count = summary.get(
        "extra_examples_count",
        0,
    )

    lines = []

    lines.append("Résumé entraînement neuronal")
    lines.append("Jeu : " + str(summary["game"]))
    lines.append("Parties simulées : " + str(summary["training_games_count"]))
    lines.append("Simulations par coup : " + str(summary["simulations_per_move"]))
    lines.append("Exemples Monte-Carlo : " + str(base_examples_count))
    lines.append("Répétitions tactiques : " + str(tactical_repeat_count))
    lines.append("Exemples tactiques ajoutés : " + str(extra_examples_count))
    lines.append("Exemples totaux : " + str(summary["examples_count"]))
    lines.append("Coups scorés : " + str(summary["scored_moves_count"]))
    lines.append("Coups légaux moyens : " + str(summary["average_legal_moves"]))
    lines.append("Score moyen du meilleur coup : " + str(summary["average_best_score"]))
    lines.append("Taille entrée : " + str(summary["input_size"]))
    lines.append("Taille couche cachée : " + str(summary["hidden_size"]))
    lines.append("Taille sortie : " + str(summary["output_size"]))
    lines.append("Époques : " + str(summary["epochs"]))
    lines.append("Taux d'apprentissage : " + str(summary["learning_rate"]))
    lines.append("Reprise d'un modèle existant : " + str(started_from_existing_model))
    lines.append("Erreur initiale : " + str(round(summary["initial_error"], 6)))
    lines.append("Erreur finale : " + str(round(summary["final_error"], 6)))
    lines.append("Amélioration erreur : " + str(round(summary["error_improvement"], 6)))

    return "\n".join(lines)