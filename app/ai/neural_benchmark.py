from time import perf_counter

from app.ai.neural_pipeline import build_augmented_move_score_dataset
from app.ai.neural_encoding import encode_move_score_dataset

from app.ai.neural_training_session import (
    get_examples_from_encoded_dataset,
    create_or_load_neural_network_for_training,
    compute_average_error_on_encoded_examples,
)

from app.ai.neural_evaluation import (
    evaluate_neural_model,
    summarize_neural_evaluation_results,
)

from app.ai.tactical_evaluation import (
    run_default_morpion_tactical_evaluation,
    summarize_tactical_evaluation,
)

from app.utils.progress import print_progress

from app.games.morpion.adapter import MORPION_ADAPTER


def evaluate_network_checkpoint(
    checkpoint_index,
    total_epochs,
    elapsed_seconds,
    network,
    encoded_examples,
    evaluation_games_count,
    game_adapter=MORPION_ADAPTER,
):
    model_data = network.to_dict()

    training_error = compute_average_error_on_encoded_examples(
        network,
        encoded_examples,
    )

    evaluation_results = evaluate_neural_model(
        model_data=model_data,
        games_count=evaluation_games_count,
        game_adapter=game_adapter,
    )

    evaluation_summary = summarize_neural_evaluation_results(
        evaluation_results,
        trained_player=game_adapter.trained_player,
        opponent_player=game_adapter.opponent_player,
    )

    tactical_results = run_default_morpion_tactical_evaluation(
        model_data,
        game_adapter,
    )

    tactical_summary = summarize_tactical_evaluation(
        tactical_results,
    )

    return {
        "checkpoint_index": checkpoint_index,
        "total_epochs": total_epochs,
        "elapsed_seconds": elapsed_seconds,
        "training_error": training_error,
        "evaluation_results": evaluation_results,
        "evaluation_efficiency": evaluation_summary["efficiency"],
        "tactical_passed_count": tactical_summary["passed_count"],
        "tactical_total_count": tactical_summary["total_count"],
        "tactical_success_rate": tactical_summary["success_rate"],
    }


def run_neural_training_benchmark(
    training_games_count,
    simulations_per_move,
    max_examples,
    tactical_repeat_count,
    hidden_size,
    checkpoints_count,
    epochs_per_checkpoint,
    learning_rate,
    evaluation_games_count,
    show_progress=False,
    seed=0,
    game_adapter=MORPION_ADAPTER,
    initial_model_data=None,
):
    """Mesure l'évolution du modèle pendant l'entraînement.

    Cette fonction ne sauvegarde rien.

    Elle construit un dataset une seule fois, puis entraîne le réseau par
    paliers. Après chaque palier, elle mesure :
    - l'erreur sur le dataset ;
    - le score contre un adversaire aléatoire ;
    - la réussite sur les tests tactiques ;
    - le temps écoulé.
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

    encoded_examples = get_examples_from_encoded_dataset(
        encoded_dataset,
    )

    network = create_or_load_neural_network_for_training(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
        initial_model_data=initial_model_data,
    )

    start_time = perf_counter()
    checkpoints = []

    initial_checkpoint = evaluate_network_checkpoint(
        checkpoint_index=0,
        total_epochs=0,
        elapsed_seconds=0.0,
        network=network,
        encoded_examples=encoded_examples,
        evaluation_games_count=evaluation_games_count,
        game_adapter=game_adapter,
    )
    checkpoints.append(initial_checkpoint)

    for checkpoint_index in range(1, checkpoints_count + 1):
        for epoch in range(epochs_per_checkpoint):
            for example in encoded_examples:
                network.train_one(
                    example["inputs"],
                    example["targets"],
                    example["legal_moves_mask"],
                )

            if show_progress:
                completed_epochs = (
                    (checkpoint_index - 1) * epochs_per_checkpoint
                    + epoch
                    + 1
                )
                total_epochs_to_run = checkpoints_count * epochs_per_checkpoint
                print_progress(
                    "Benchmark entraînement",
                    completed_epochs,
                    total_epochs_to_run,
                )

        elapsed_seconds = perf_counter() - start_time

        checkpoint = evaluate_network_checkpoint(
            checkpoint_index=checkpoint_index,
            total_epochs=checkpoint_index * epochs_per_checkpoint,
            elapsed_seconds=elapsed_seconds,
            network=network,
            encoded_examples=encoded_examples,
            evaluation_games_count=evaluation_games_count,
            game_adapter=game_adapter,
        )
        checkpoints.append(checkpoint)

    first_checkpoint = checkpoints[0]
    last_checkpoint = checkpoints[-1]

    return {
        "game": game_adapter.name,
        "started_from_existing_model": initial_model_data is not None,
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "tactical_repeat_count": tactical_repeat_count,
        "examples_count": encoded_dataset["encoded_examples_count"],
        "input_size": encoded_dataset["input_size"],
        "hidden_size": hidden_size,
        "output_size": encoded_dataset["output_size"],
        "checkpoints_count": checkpoints_count,
        "epochs_per_checkpoint": epochs_per_checkpoint,
        "total_epochs": checkpoints_count * epochs_per_checkpoint,
        "learning_rate": learning_rate,
        "evaluation_games_count": evaluation_games_count,
        "checkpoints": checkpoints,
        "initial_training_error": first_checkpoint["training_error"],
        "final_training_error": last_checkpoint["training_error"],
        "training_error_improvement": (
            first_checkpoint["training_error"]
            - last_checkpoint["training_error"]
        ),
        "initial_evaluation_efficiency": first_checkpoint["evaluation_efficiency"],
        "final_evaluation_efficiency": last_checkpoint["evaluation_efficiency"],
        "evaluation_efficiency_improvement": (
            last_checkpoint["evaluation_efficiency"]
            - first_checkpoint["evaluation_efficiency"]
        ),
        "initial_tactical_success_rate": first_checkpoint["tactical_success_rate"],
        "final_tactical_success_rate": last_checkpoint["tactical_success_rate"],
        "tactical_success_rate_improvement": (
            last_checkpoint["tactical_success_rate"]
            - first_checkpoint["tactical_success_rate"]
        ),
        "final_model_data": network.to_dict(),
    }


def format_neural_benchmark_report(benchmark_result):
    lines = []

    lines.append("Benchmark entraînement neuronal")
    lines.append("Jeu : " + str(benchmark_result["game"]))
    lines.append(
        "Départ depuis modèle existant : "
        + str(benchmark_result["started_from_existing_model"])
    )
    lines.append("Exemples : " + str(benchmark_result["examples_count"]))
    lines.append(
        "Répétitions tactiques : "
        + str(benchmark_result["tactical_repeat_count"])
    )
    lines.append("Couche cachée : " + str(benchmark_result["hidden_size"]))
    lines.append(
        "Paliers : "
        + str(benchmark_result["checkpoints_count"])
        + " × "
        + str(benchmark_result["epochs_per_checkpoint"])
        + " époques"
    )
    lines.append(
        "Parties d'évaluation par palier : "
        + str(benchmark_result["evaluation_games_count"])
    )
    lines.append("")

    lines.append(
        "Palier | Époques | Temps (s) | Erreur dataset | Tactique | Efficacité"
    )

    for checkpoint in benchmark_result["checkpoints"]:
        tactical_text = (
            str(checkpoint["tactical_passed_count"])
            + "/"
            + str(checkpoint["tactical_total_count"])
            + " ("
            + str(round(checkpoint["tactical_success_rate"], 2))
            + " %)"
        )

        line = (
            str(checkpoint["checkpoint_index"])
            + " | "
            + str(checkpoint["total_epochs"])
            + " | "
            + str(round(checkpoint["elapsed_seconds"], 2))
            + " | "
            + str(round(checkpoint["training_error"], 6))
            + " | "
            + tactical_text
            + " | "
            + str(round(checkpoint["evaluation_efficiency"], 2))
            + " %"
        )
        lines.append(line)

    lines.append("")
    lines.append(
        "Gain erreur dataset : "
        + str(round(benchmark_result["training_error_improvement"], 6))
    )
    lines.append(
        "Gain efficacité : "
        + str(round(benchmark_result["evaluation_efficiency_improvement"], 2))
        + " points"
    )
    lines.append(
        "Gain tactique : "
        + str(round(benchmark_result["tactical_success_rate_improvement"], 2))
        + " points"
    )

    return "\n".join(lines)