from app.ai.neural_training_session import compute_average_error_on_encoded_examples

from app.ai.neural_evaluation import (
    evaluate_neural_model,
    summarize_neural_evaluation_results,
)

from app.ai.tactical_evaluation import (
    run_default_morpion_tactical_evaluation,
    summarize_tactical_evaluation,
)

from app.ai.neural_checkpoint_reference import (
    evaluate_reference_opponents_if_needed,
    format_reference_summary,
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
        "reference_worst_efficiency_name": reference_summary["worst_efficiency_name"],
        "reference_worst_survival_rate": reference_summary["worst_survival_rate"],
        "reference_worst_survival_name": reference_summary["worst_survival_name"],
        "reference_worst_name": reference_summary["worst_efficiency_name"],
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



def format_checkpoint_line(checkpoint, is_best_checkpoint=False):
    tactical_text = (
        str(checkpoint["tactical_passed_count"])
        + "/"
        + str(checkpoint["tactical_total_count"])
        + " ("
        + str(round(checkpoint["tactical_success_rate"], 2))
        + " %)"
    )

    reference_text = format_reference_summary(checkpoint)

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

# Backward-compatible exports for tests and older imports.
from app.ai.neural_checkpoint_selection import (  # noqa: E402
    get_best_checkpoint_from_benchmark_result,
    is_checkpoint_better,
)
