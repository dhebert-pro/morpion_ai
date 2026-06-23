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


def is_checkpoint_better(candidate_checkpoint, current_best_checkpoint):
    """Détermine si un checkpoint est meilleur qu'un autre.

    Priorité :
    1. meilleur taux tactique ;
    2. meilleure efficacité contre l'adversaire aléatoire ;
    3. plus faible erreur sur le dataset.
    """

    if current_best_checkpoint is None:
        return True

    if candidate_checkpoint["tactical_success_rate"] != current_best_checkpoint["tactical_success_rate"]:
        return (
            candidate_checkpoint["tactical_success_rate"]
            > current_best_checkpoint["tactical_success_rate"]
        )

    if candidate_checkpoint["evaluation_efficiency"] != current_best_checkpoint["evaluation_efficiency"]:
        return (
            candidate_checkpoint["evaluation_efficiency"]
            > current_best_checkpoint["evaluation_efficiency"]
        )

    return candidate_checkpoint["training_error"] < current_best_checkpoint["training_error"]


def format_checkpoint_line(checkpoint, is_best_checkpoint=False):
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

    if is_best_checkpoint:
        line += "  <- meilleur"

    return line


def get_checkpoint_table_header():
    return "Palier | Époques | Temps (s) | Erreur dataset | Tactique | Efficacité"


def get_best_checkpoint_from_benchmark_result(benchmark_result):
    """Récupère ou reconstruit le meilleur checkpoint.

    Cette fonction accepte :
    - un résultat complet récent avec best_checkpoint ;
    - un résultat avec une liste checkpoints ;
    - un résultat simplifié utilisé par certains tests.
    """

    best_checkpoint = benchmark_result.get("best_checkpoint")

    if best_checkpoint:
        return best_checkpoint

    checkpoints = benchmark_result.get("checkpoints", [])

    if checkpoints:
        best_checkpoint_index = benchmark_result.get("best_checkpoint_index")

        if best_checkpoint_index is not None:
            for checkpoint in checkpoints:
                if checkpoint["checkpoint_index"] == best_checkpoint_index:
                    return checkpoint

        best_checkpoint = None

        for checkpoint in checkpoints:
            if is_checkpoint_better(checkpoint, best_checkpoint):
                best_checkpoint = checkpoint

        return best_checkpoint

    return {
        "checkpoint_index": benchmark_result.get(
            "best_checkpoint_index",
            benchmark_result.get("checkpoints_count", 0),
        ),
        "total_epochs": benchmark_result.get(
            "best_total_epochs",
            benchmark_result.get("total_epochs", 0),
        ),
        "elapsed_seconds": benchmark_result.get("elapsed_seconds", 0.0),
        "training_error": benchmark_result.get(
            "best_training_error",
            benchmark_result.get(
                "final_training_error",
                benchmark_result.get("final_error", 0.0),
            ),
        ),
        "evaluation_efficiency": benchmark_result.get(
            "best_evaluation_efficiency",
            benchmark_result.get("final_evaluation_efficiency", 0.0),
        ),
        "tactical_passed_count": benchmark_result.get("tactical_passed_count", 0),
        "tactical_total_count": benchmark_result.get("tactical_total_count", 0),
        "tactical_success_rate": benchmark_result.get(
            "best_tactical_success_rate",
            benchmark_result.get("final_tactical_success_rate", 0.0),
        ),
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
    print_checkpoints=False,
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

    Elle conserve aussi le meilleur modèle rencontré.
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

    best_checkpoint = None
    best_model_data = None

    if print_checkpoints:
        print()
        print(get_checkpoint_table_header())

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

    if is_checkpoint_better(initial_checkpoint, best_checkpoint):
        best_checkpoint = initial_checkpoint
        best_model_data = network.to_dict()

    if print_checkpoints:
        print(format_checkpoint_line(initial_checkpoint, is_best_checkpoint=True))

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

        checkpoint_is_best = is_checkpoint_better(
            checkpoint,
            best_checkpoint,
        )

        if checkpoint_is_best:
            best_checkpoint = checkpoint
            best_model_data = network.to_dict()

        if print_checkpoints:
            print(format_checkpoint_line(checkpoint, checkpoint_is_best))

    first_checkpoint = checkpoints[0]
    last_checkpoint = checkpoints[-1]

    base_examples_count = raw_dataset.get(
        "base_examples_count",
        raw_dataset.get("examples_count", 0),
    )
    extra_examples_count = raw_dataset.get(
        "extra_examples_count",
        0,
    )

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "started_from_existing_model": initial_model_data is not None,
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "tactical_repeat_count": tactical_repeat_count,
        "base_examples_count": base_examples_count,
        "extra_examples_count": extra_examples_count,
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
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_index": best_checkpoint["checkpoint_index"],
        "best_total_epochs": best_checkpoint["total_epochs"],
        "best_training_error": best_checkpoint["training_error"],
        "best_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
        "best_tactical_success_rate": best_checkpoint["tactical_success_rate"],
        "best_model_data": best_model_data,
        "final_model_data": network.to_dict(),
        "final_checkpoint_is_best": (
            best_checkpoint["checkpoint_index"] == last_checkpoint["checkpoint_index"]
        ),
    }


def create_training_summary_from_benchmark_result(benchmark_result):
    best_checkpoint = get_best_checkpoint_from_benchmark_result(
        benchmark_result,
    )

    initial_error = benchmark_result.get(
        "initial_training_error",
        benchmark_result.get("initial_error", 0.0),
    )

    return {
        "game": benchmark_result["game"],
        "training_games_count": benchmark_result["training_games_count"],
        "simulations_per_move": benchmark_result["simulations_per_move"],
        "max_examples": benchmark_result["max_examples"],
        "tactical_repeat_count": benchmark_result["tactical_repeat_count"],
        "base_examples_count": benchmark_result["base_examples_count"],
        "extra_examples_count": benchmark_result["extra_examples_count"],
        "examples_count": benchmark_result["examples_count"],
        "scored_moves_count": None,
        "average_legal_moves": None,
        "average_best_score": None,
        "input_size": benchmark_result["input_size"],
        "hidden_size": benchmark_result["hidden_size"],
        "output_size": benchmark_result["output_size"],
        "epochs": best_checkpoint["total_epochs"],
        "learning_rate": benchmark_result["learning_rate"],
        "started_from_existing_model": benchmark_result["started_from_existing_model"],
        "initial_error": initial_error,
        "final_error": best_checkpoint["training_error"],
        "error_improvement": initial_error - best_checkpoint["training_error"],
        "benchmark_checkpoints_count": benchmark_result.get("checkpoints_count", 0),
        "benchmark_epochs_per_checkpoint": benchmark_result.get("epochs_per_checkpoint", 0),
        "benchmark_evaluation_games_count": benchmark_result.get("evaluation_games_count", 0),
        "initial_evaluation_efficiency": benchmark_result.get("initial_evaluation_efficiency", 0.0),
        "final_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
        "evaluation_efficiency_improvement": (
            best_checkpoint["evaluation_efficiency"]
            - benchmark_result.get("initial_evaluation_efficiency", 0.0)
        ),
        "initial_tactical_success_rate": benchmark_result.get("initial_tactical_success_rate", 0.0),
        "final_tactical_success_rate": best_checkpoint["tactical_success_rate"],
        "tactical_success_rate_improvement": (
            best_checkpoint["tactical_success_rate"]
            - benchmark_result.get("initial_tactical_success_rate", 0.0)
        ),
        "best_checkpoint_index": best_checkpoint["checkpoint_index"],
        "best_total_epochs": best_checkpoint["total_epochs"],
        "final_checkpoint_is_best": benchmark_result.get("final_checkpoint_is_best", True),
    }


def create_model_package_from_benchmark_result(
    benchmark_result,
    game_adapter=MORPION_ADAPTER,
):
    best_checkpoint = get_best_checkpoint_from_benchmark_result(
        benchmark_result,
    )

    model_data = benchmark_result.get(
        "best_model_data",
        benchmark_result["final_model_data"],
    )

    return {
        "type": "neural_model_package",
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "model_data": model_data,
        "training_summary": create_training_summary_from_benchmark_result(
            benchmark_result,
        ),
        "benchmark_summary": {
            "checkpoints": benchmark_result.get("checkpoints", []),
            "initial_training_error": benchmark_result.get("initial_training_error", 0.0),
            "final_training_error": benchmark_result.get("final_training_error", best_checkpoint["training_error"]),
            "training_error_improvement": benchmark_result.get("training_error_improvement", 0.0),
            "initial_evaluation_efficiency": benchmark_result.get("initial_evaluation_efficiency", 0.0),
            "final_evaluation_efficiency": benchmark_result.get("final_evaluation_efficiency", best_checkpoint["evaluation_efficiency"]),
            "evaluation_efficiency_improvement": benchmark_result.get("evaluation_efficiency_improvement", 0.0),
            "initial_tactical_success_rate": benchmark_result.get("initial_tactical_success_rate", 0.0),
            "final_tactical_success_rate": benchmark_result.get("final_tactical_success_rate", best_checkpoint["tactical_success_rate"]),
            "tactical_success_rate_improvement": benchmark_result.get("tactical_success_rate_improvement", 0.0),
            "best_checkpoint": best_checkpoint,
            "best_checkpoint_index": best_checkpoint["checkpoint_index"],
            "best_total_epochs": best_checkpoint["total_epochs"],
            "best_training_error": best_checkpoint["training_error"],
            "best_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
            "best_tactical_success_rate": best_checkpoint["tactical_success_rate"],
            "final_checkpoint_is_best": benchmark_result.get("final_checkpoint_is_best", True),
        },
    }


def format_neural_benchmark_report(benchmark_result):
    lines = []

    best_checkpoint = get_best_checkpoint_from_benchmark_result(
        benchmark_result,
    )

    best_checkpoint_index = best_checkpoint["checkpoint_index"]
    best_total_epochs = best_checkpoint["total_epochs"]
    best_training_error = best_checkpoint["training_error"]
    best_evaluation_efficiency = best_checkpoint["evaluation_efficiency"]
    best_tactical_success_rate = best_checkpoint["tactical_success_rate"]

    final_checkpoint_is_best = benchmark_result.get(
        "final_checkpoint_is_best",
        True,
    )

    lines.append("Benchmark entraînement neuronal")
    lines.append("Jeu : " + str(benchmark_result["game"]))
    lines.append(
        "Départ depuis modèle existant : "
        + str(benchmark_result["started_from_existing_model"])
    )
    lines.append("Exemples Monte-Carlo : " + str(benchmark_result.get("base_examples_count", benchmark_result.get("examples_count", 0))))
    lines.append("Exemples tactiques : " + str(benchmark_result.get("extra_examples_count", 0)))
    lines.append("Exemples totaux : " + str(benchmark_result["examples_count"]))
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

    lines.append(get_checkpoint_table_header())

    for checkpoint in benchmark_result.get("checkpoints", []):
        is_best = checkpoint["checkpoint_index"] == best_checkpoint_index
        lines.append(format_checkpoint_line(checkpoint, is_best))

    lines.append("")

    lines.append(
        "Gain erreur dataset : "
        + str(round(benchmark_result.get("training_error_improvement", 0.0), 6))
    )
    lines.append(
        "Gain efficacité : "
        + str(round(benchmark_result.get("evaluation_efficiency_improvement", 0.0), 2))
        + " points"
    )
    lines.append(
        "Gain tactique : "
        + str(round(benchmark_result.get("tactical_success_rate_improvement", 0.0), 2))
        + " points"
    )

    lines.append("")
    lines.append(
        "Gain erreur dataset du dernier palier : "
        + str(round(benchmark_result.get("training_error_improvement", 0.0), 6))
    )
    lines.append(
        "Gain efficacité du dernier palier : "
        + str(round(benchmark_result.get("evaluation_efficiency_improvement", 0.0), 2))
        + " points"
    )
    lines.append(
        "Gain tactique du dernier palier : "
        + str(round(benchmark_result.get("tactical_success_rate_improvement", 0.0), 2))
        + " points"
    )

    lines.append("")
    lines.append(
        "Meilleur palier : "
        + str(best_checkpoint_index)
        + " ("
        + str(best_total_epochs)
        + " époques)"
    )
    lines.append(
        "Erreur du meilleur modèle : "
        + str(round(best_training_error, 6))
    )
    lines.append(
        "Efficacité du meilleur modèle : "
        + str(round(best_evaluation_efficiency, 2))
        + " %"
    )
    lines.append(
        "Tactique du meilleur modèle : "
        + str(round(best_tactical_success_rate, 2))
        + " %"
    )

    if final_checkpoint_is_best:
        lines.append("Modèle retenu : dernier palier, car c'est aussi le meilleur.")
    else:
        lines.append("Modèle retenu : meilleur palier, pas le dernier.")

    return "\n".join(lines)