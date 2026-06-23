from app.ai.neural_checkpoint_selection import get_best_checkpoint_from_benchmark_result
from app.games.morpion.adapter import MORPION_ADAPTER


def create_training_summary_from_benchmark_result(benchmark_result):
    best_checkpoint = get_best_checkpoint_from_benchmark_result(benchmark_result)

    initial_error = benchmark_result.get(
        "initial_training_error",
        benchmark_result.get("initial_error", 0.0),
    )
    best_validation_error = best_checkpoint.get("validation_error", best_checkpoint["training_error"])
    initial_validation_error = benchmark_result.get("initial_validation_error", initial_error)

    return {
        "game": benchmark_result["game"],
        "training_games_count": benchmark_result["training_games_count"],
        "simulations_per_move": benchmark_result["simulations_per_move"],
        "max_examples": benchmark_result["max_examples"],
        "available_states_count": benchmark_result.get("available_states_count", 0),
        "tactical_repeat_count": benchmark_result["tactical_repeat_count"],
        "base_examples_count": benchmark_result["base_examples_count"],
        "extra_examples_count": benchmark_result["extra_examples_count"],
        "tactical_examples_count": benchmark_result.get("tactical_examples_count", 0),
        "reference_examples_count": benchmark_result.get("reference_examples_count", 0),
        "reference_training_games_count": benchmark_result.get("reference_training_games_count", 0),
        "reference_training_names": benchmark_result.get("reference_training_names", []),
        "examples_count": benchmark_result["examples_count"],
        "training_examples_count": benchmark_result.get("training_examples_count"),
        "validation_examples_count": benchmark_result.get("validation_examples_count"),
        "validation_ratio": benchmark_result.get("validation_ratio", 0.0),
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
        "initial_validation_error": initial_validation_error,
        "final_validation_error": best_validation_error,
        "validation_error_improvement": initial_validation_error - best_validation_error,
        "benchmark_checkpoints_count": benchmark_result.get("checkpoints_count", 0),
        "benchmark_epochs_per_checkpoint": benchmark_result.get("epochs_per_checkpoint", 0),
        "benchmark_evaluation_games_count": benchmark_result.get("evaluation_games_count", 0),
        "initial_evaluation_efficiency": benchmark_result.get("initial_evaluation_efficiency", 0.0),
        "final_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
        "evaluation_efficiency_improvement": (
            best_checkpoint["evaluation_efficiency"]
            - benchmark_result.get("initial_evaluation_efficiency", 0.0)
        ),
        "final_reference_worst_efficiency": best_checkpoint.get("reference_worst_efficiency"),
        "final_reference_worst_efficiency_name": best_checkpoint.get("reference_worst_efficiency_name"),
        "final_reference_worst_survival_rate": best_checkpoint.get("reference_worst_survival_rate"),
        "final_reference_worst_survival_name": best_checkpoint.get("reference_worst_survival_name"),
        "final_reference_worst_name": best_checkpoint.get("reference_worst_name"),
        "initial_tactical_success_rate": benchmark_result.get("initial_tactical_success_rate", 0.0),
        "final_tactical_success_rate": best_checkpoint["tactical_success_rate"],
        "tactical_success_rate_improvement": (
            best_checkpoint["tactical_success_rate"]
            - benchmark_result.get("initial_tactical_success_rate", 0.0)
        ),
        "best_checkpoint_index": best_checkpoint["checkpoint_index"],
        "best_total_epochs": best_checkpoint["total_epochs"],
        "final_checkpoint_is_best": benchmark_result.get("final_checkpoint_is_best", True),
        "stopped_early": benchmark_result.get("stopped_early", False),
    }


def create_model_package_from_benchmark_result(
    benchmark_result,
    game_adapter=MORPION_ADAPTER,
):
    best_checkpoint = get_best_checkpoint_from_benchmark_result(benchmark_result)
    model_data = benchmark_result.get("best_model_data", benchmark_result["final_model_data"])

    return {
        "type": "neural_model_package",
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "model_data": model_data,
        "training_summary": create_training_summary_from_benchmark_result(benchmark_result),
        "benchmark_summary": create_benchmark_summary(benchmark_result, best_checkpoint),
    }


def create_benchmark_summary(benchmark_result, best_checkpoint):
    return {
        "available_states_count": benchmark_result.get("available_states_count", 0),
        "checkpoints": benchmark_result.get("checkpoints", []),
        "initial_training_error": benchmark_result.get("initial_training_error", 0.0),
        "final_training_error": benchmark_result.get("final_training_error", best_checkpoint["training_error"]),
        "training_error_improvement": benchmark_result.get("training_error_improvement", 0.0),
        "initial_validation_error": benchmark_result.get("initial_validation_error", 0.0),
        "final_validation_error": benchmark_result.get("final_validation_error", best_checkpoint.get("validation_error", 0.0)),
        "validation_error_improvement": benchmark_result.get("validation_error_improvement", 0.0),
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
        "best_validation_error": best_checkpoint.get("validation_error", best_checkpoint["training_error"]),
        "best_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
        "best_reference_worst_efficiency": best_checkpoint.get("reference_worst_efficiency"),
        "best_reference_worst_efficiency_name": best_checkpoint.get("reference_worst_efficiency_name"),
        "best_reference_worst_survival_rate": best_checkpoint.get("reference_worst_survival_rate"),
        "best_reference_worst_survival_name": best_checkpoint.get("reference_worst_survival_name"),
        "best_tactical_success_rate": best_checkpoint["tactical_success_rate"],
        "final_checkpoint_is_best": benchmark_result.get("final_checkpoint_is_best", True),
        "stopped_early": benchmark_result.get("stopped_early", False),
    }
