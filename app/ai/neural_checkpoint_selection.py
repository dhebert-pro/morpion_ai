from app.ai.neural_dataset_split import get_effective_validation_error
from app.ai.neural_checkpoint_reference import (
    get_reference_selection_efficiency,
    get_reference_selection_survival,
)


def is_checkpoint_better(candidate_checkpoint, current_best_checkpoint):
    if current_best_checkpoint is None:
        return True

    if candidate_checkpoint["tactical_success_rate"] != current_best_checkpoint["tactical_success_rate"]:
        return (
            candidate_checkpoint["tactical_success_rate"]
            > current_best_checkpoint["tactical_success_rate"]
        )

    candidate_survival = get_reference_selection_survival(candidate_checkpoint)
    best_survival = get_reference_selection_survival(current_best_checkpoint)

    if candidate_survival != best_survival:
        return candidate_survival > best_survival

    candidate_reference_score = get_reference_selection_efficiency(candidate_checkpoint)
    best_reference_score = get_reference_selection_efficiency(current_best_checkpoint)

    if candidate_reference_score != best_reference_score:
        return candidate_reference_score > best_reference_score

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


def get_best_checkpoint_from_benchmark_result(benchmark_result):
    best_checkpoint = benchmark_result.get("best_checkpoint")

    if best_checkpoint:
        return best_checkpoint

    checkpoints = benchmark_result.get("checkpoints", [])

    if checkpoints:
        return _get_best_from_checkpoints(checkpoints, benchmark_result)

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

    return _build_legacy_checkpoint(benchmark_result, training_error, validation_error)


def _get_best_from_checkpoints(checkpoints, benchmark_result):
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


def _build_legacy_checkpoint(benchmark_result, training_error, validation_error):
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
        "reference_worst_efficiency_name": benchmark_result.get("best_reference_worst_efficiency_name"),
        "reference_worst_survival_rate": benchmark_result.get("best_reference_worst_survival_rate"),
        "reference_worst_survival_name": benchmark_result.get("best_reference_worst_survival_name"),
        "reference_worst_name": benchmark_result.get("best_reference_worst_name"),
        "reference_evaluations": benchmark_result.get("best_reference_evaluations", []),
        "tactical_passed_count": benchmark_result.get("tactical_passed_count", 0),
        "tactical_total_count": benchmark_result.get("tactical_total_count", 0),
        "tactical_success_rate": benchmark_result.get(
            "best_tactical_success_rate",
            benchmark_result.get("final_tactical_success_rate", 0.0),
        ),
    }
