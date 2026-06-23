from app.ai.training_dataset import (
    build_move_score_dataset,
    summarize_move_score_dataset,
)

from app.ai.tactical_training import (
    create_default_morpion_tactical_dataset,
    merge_move_score_datasets,
)

from app.ai.reference_training_dataset import build_reference_move_score_dataset

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
    seed=0,
    reference_training_games_count=0,
    reference_training_max_examples=0,
    reference_training_names=None,
):
    base_dataset = build_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        show_progress=show_progress,
        game_adapter=game_adapter,
        seed=seed,
    )
    augmented_dataset = base_dataset
    augmented_dataset["base_examples_count"] = base_dataset.get(
        "examples_count",
        0,
    )
    augmented_dataset["extra_examples_count"] = 0
    augmented_dataset["tactical_examples_count"] = 0
    augmented_dataset["reference_examples_count"] = 0

    if tactical_repeat_count > 0:
        tactical_dataset = create_default_morpion_tactical_dataset(
            repeat_count=tactical_repeat_count,
            game_adapter=game_adapter,
        )
        augmented_dataset = merge_move_score_datasets(
            augmented_dataset,
            tactical_dataset,
        )

    if _should_add_reference_training(reference_training_games_count, reference_training_names):
        reference_dataset = build_reference_move_score_dataset(
            training_games_count=reference_training_games_count,
            simulations_per_move=simulations_per_move,
            max_examples=reference_training_max_examples,
            reference_names=reference_training_names,
            show_progress=show_progress,
            game_adapter=game_adapter,
            seed=seed + 100,
        )
        augmented_dataset = merge_move_score_datasets(
            augmented_dataset,
            reference_dataset,
        )

    augmented_dataset["tactical_repeat_count"] = tactical_repeat_count
    augmented_dataset["reference_training_games_count"] = reference_training_games_count
    augmented_dataset["reference_training_max_examples"] = reference_training_max_examples
    augmented_dataset["reference_training_names"] = reference_training_names or []

    return augmented_dataset


def _should_add_reference_training(reference_training_games_count, reference_training_names):
    return (
        reference_training_games_count > 0
        and reference_training_names is not None
        and len(reference_training_names) > 0
    )


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
    dataset_seed=0,
):
    """Construit et entraîne un modèle neuronal complet en mémoire."""

    raw_dataset = build_augmented_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        game_adapter=game_adapter,
        seed=dataset_seed,
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
        "summary": _build_training_summary(
            game_adapter,
            raw_dataset,
            raw_dataset_summary,
            training_result,
            training_games_count,
            simulations_per_move,
            max_examples,
            dataset_seed,
            hidden_size,
            epochs,
            learning_rate,
        ),
    }


def _build_training_summary(
    game_adapter,
    raw_dataset,
    raw_dataset_summary,
    training_result,
    training_games_count,
    simulations_per_move,
    max_examples,
    dataset_seed,
    hidden_size,
    epochs,
    learning_rate,
):
    return {
        "game": game_adapter.name,
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "dataset_seed": dataset_seed,
        "available_states_count": raw_dataset_summary["available_states_count"],
        "tactical_repeat_count": raw_dataset.get("tactical_repeat_count", 0),
        "base_examples_count": raw_dataset.get("base_examples_count", 0),
        "extra_examples_count": raw_dataset.get("extra_examples_count", 0),
        "examples_count": training_result["examples_count"],
        "scored_moves_count": raw_dataset_summary["scored_moves_count"],
        "average_legal_moves": raw_dataset_summary["average_legal_moves"],
        "average_best_score": raw_dataset_summary["average_best_score"],
        "input_size": training_result["input_size"],
        "hidden_size": hidden_size,
        "output_size": training_result["output_size"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "started_from_existing_model": training_result["started_from_existing_model"],
        "initial_error": training_result["initial_error"],
        "final_error": training_result["final_error"],
        "error_improvement": training_result["initial_error"] - training_result["final_error"],
    }


def format_neural_training_summary(summary):
    lines = []
    lines.append("Résumé entraînement neuronal")
    lines.append("Jeu : " + str(summary["game"]))
    lines.append("Parties simulées : " + str(summary["training_games_count"]))
    lines.append("Simulations par coup : " + str(summary["simulations_per_move"]))
    lines.append("Exemples Monte-Carlo : " + str(summary.get("base_examples_count", 0)))
    lines.append("Répétitions tactiques : " + str(summary.get("tactical_repeat_count", 0)))
    lines.append("Exemples tactiques ajoutés : " + str(summary.get("extra_examples_count", 0)))
    lines.append("Exemples totaux : " + str(summary["examples_count"]))
    lines.append("Coups scorés : " + str(summary["scored_moves_count"]))
    lines.append("Coups légaux moyens : " + str(summary["average_legal_moves"]))
    lines.append("Score moyen du meilleur coup : " + str(summary["average_best_score"]))
    lines.append("Taille entrée : " + str(summary["input_size"]))
    lines.append("Taille couche cachée : " + str(summary["hidden_size"]))
    lines.append("Taille sortie : " + str(summary["output_size"]))
    lines.append("Époques : " + str(summary["epochs"]))
    lines.append("Taux d'apprentissage : " + str(summary["learning_rate"]))
    lines.append("Reprise d'un modèle existant : " + str(summary.get("started_from_existing_model", False)))
    lines.append("Erreur initiale : " + str(round(summary["initial_error"], 6)))
    lines.append("Erreur finale : " + str(round(summary["final_error"], 6)))
    lines.append("Amélioration erreur : " + str(round(summary["error_improvement"], 6)))
    return "\n".join(lines)
