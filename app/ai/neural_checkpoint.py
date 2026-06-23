from app.ai.neural_training_session import compute_average_error_on_encoded_examples

from app.ai.neural_evaluation import (
    evaluate_neural_model,
    summarize_neural_evaluation_results,
)

from app.ai.tactical_evaluation import (
    run_default_morpion_tactical_evaluation,
    summarize_tactical_evaluation,
)

from app.ai.neural_dataset_split import get_effective_validation_error
from app.ai.neural_checkpoint_reference import (
    evaluate_reference_opponents_if_needed,
    format_reference_efficiency,
    get_reference_selection_efficiency,
    summarize_reference_evaluations,
)
from app.games.morpion.adapter import MORPION_ADAPTER


def evaluate_network_checkpoint(
    checkpoint_index,
    total_epochs,
    elapsed_seconds,
    network,
    training_examples,
    validation_examples,
    evaluation_games_count,
    game_adapter=MORPION_ADAPTER,
    evaluation_seed=None,
    reference_evaluation_games_count=0,
    reference_evaluation_names=None,
):
    model_data = network.to_dict()

    training_error = compute_average_error_on_encoded_examples(
        network,
        training_examples,
    )
    validation_error = compute_average_error_on_encoded_examples(
        network,
        validation_examples,
    )

    evaluation_results = evaluate_neural_model(
        model_data=model_data,
        games_count=evaluation_games_count,
        game_adapter=game_adapter,
        seed=evaluation_seed,
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
    tactical_summary = summarize_tactical_evaluation(tactical_results)
    tactical_failed_results = _get_failed_tactical_results(tactical_results)
    reference_evaluations = evaluate_reference_opponents_if_needed(
        model_data=model_data,
        games_count=reference_evaluation_games_count,
        game_adapter=game_adapter,
        seed=evaluation_seed,
        reference_names=reference_evaluation_names,
    )
    reference_summary = summarize_reference_evaluations(reference_evaluations)

    return {
        "checkpoint_index": checkpoint_index,
        "total_epochs": total_epochs,
        "elapsed_seconds": elapsed_seconds,
        "training_error": training_error,
        "validation_error": validation_error,
        "evaluation_results": evaluation_results,
        "evaluation_seed": evaluation_seed,
        "evaluation_efficiency": evaluation_summary["efficiency"],
        "reference_evaluations": reference_evaluations,
        "reference_worst_efficiency": reference_summary["worst_efficiency"],
        "reference_worst_name": reference_summary["worst_name"],
        "tactical_passed_count": tactical_summary["passed_count"],
        "tactical_total_count": tactical_summary["total_count"],
        "tactical_success_rate": tactical_summary["success_rate"],
        "tactical_failed_results": tactical_failed_results,
    }


def _get_failed_tactical_results(tactical_results):
    failed_results = []

    for result in tactical_results:
        if result["passed"]:
            continue

        failed_results.append({
            "name": result["name"],
            "expected_moves": result["expected_moves"],
            "chosen_move": result["chosen_move"],
            "description": result["description"],
        })

    return failed_results


def is_checkpoint_better(candidate_checkpoint, current_best_checkpoint):
    if current_best_checkpoint is None:
        return True

    if candidate_checkpoint["tactical_success_rate"] != current_best_checkpoint["tactical_success_rate"]:
        return (
            candidate_checkpoint["tactical_success_rate"]
            > current_best_checkpoint["tactical_success_rate"]
        )

    candidate_reference_efficiency = get_reference_selection_efficiency(candidate_checkpoint)
    best_reference_efficiency = get_reference_selection_efficiency(current_best_checkpoint)

    if candidate_reference_efficiency != best_reference_efficiency:
        return candidate_reference_efficiency > best_reference_efficiency

    if candidate_checkpoint["evaluation_efficiency"] != current_best_checkpoint["evaluation_efficiency"]:
        return (
            candidate_checkpoint["evaluation_efficiency"]
            > current_best_checkpoint["evaluation_efficiency"]
        )

    candidate_validation_error = get_effective_validation_error(candidate_checkpoint)
    best_validation_error = get_effective_validation_error(current_best_checkpoint)

    if candidate_validation_error != best_validation_error:
        return candidate_validation_error < best_validation_error

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

    reference_text = format_reference_efficiency(checkpoint)

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
        + reference_text
        + " | "
        + str(round(checkpoint["evaluation_efficiency"], 2))
        + " %"
    )

    if "validation_error" in checkpoint:
        line += " | " + str(round(checkpoint["validation_error"], 6))

    if is_best_checkpoint:
        line += "  <- meilleur"

    return line


def get_checkpoint_table_header():
    return "Palier | Époques | Temps (s) | Erreur dataset | Tactique | Référence | Efficacité | Erreur valid"


def get_best_checkpoint_from_benchmark_result(benchmark_result):
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

    training_error = benchmark_result.get(
        "best_training_error",
        benchmark_result.get(
            "final_training_error",
            benchmark_result.get("final_error", 0.0),
        ),
    )

    validation_error = benchmark_result.get(
        "best_validation_error",
        benchmark_result.get("final_validation_error", training_error),
    )

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
        "training_error": training_error,
        "validation_error": validation_error,
        "evaluation_efficiency": benchmark_result.get(
            "best_evaluation_efficiency",
            benchmark_result.get("final_evaluation_efficiency", 0.0),
        ),
        "reference_worst_efficiency": benchmark_result.get("best_reference_worst_efficiency"),
        "reference_worst_name": benchmark_result.get("best_reference_worst_name"),
        "reference_evaluations": benchmark_result.get("best_reference_evaluations", []),
        "tactical_passed_count": benchmark_result.get("tactical_passed_count", 0),
        "tactical_total_count": benchmark_result.get("tactical_total_count", 0),
        "tactical_success_rate": benchmark_result.get(
            "best_tactical_success_rate",
            benchmark_result.get("final_tactical_success_rate", 0.0),
        ),
    }
