from app.ai.neural_checkpoint import (
    evaluate_network_checkpoint,
    is_checkpoint_better,
    format_checkpoint_line,
)


def evaluate_and_store_checkpoint(
    benchmark_state,
    checkpoint_index,
    total_epochs,
    elapsed_seconds,
    network,
    training_examples,
    validation_examples,
    evaluation_games_count,
    game_adapter,
    print_checkpoints=False,
):
    checkpoint = evaluate_network_checkpoint(
        checkpoint_index=checkpoint_index,
        total_epochs=total_epochs,
        elapsed_seconds=elapsed_seconds,
        network=network,
        training_examples=training_examples,
        validation_examples=validation_examples,
        evaluation_games_count=evaluation_games_count,
        game_adapter=game_adapter,
    )
    benchmark_state["checkpoints"].append(checkpoint)

    checkpoint_is_best = is_checkpoint_better(
        checkpoint,
        benchmark_state["best_checkpoint"],
    )

    if checkpoint_is_best:
        benchmark_state["best_checkpoint"] = checkpoint
        benchmark_state["best_model_data"] = network.to_dict()

    if print_checkpoints:
        print(format_checkpoint_line(checkpoint, checkpoint_is_best))

    return checkpoint_is_best


def should_stop_early(benchmark_state, early_stop_patience):
    if early_stop_patience is None or early_stop_patience <= 0:
        return False

    return benchmark_state["checkpoints_without_improvement"] >= early_stop_patience


def create_benchmark_result(
    raw_dataset,
    encoded_dataset,
    training_examples,
    validation_examples,
    checkpoints_count,
    epochs_per_checkpoint,
    learning_rate,
    evaluation_games_count,
    training_games_count,
    simulations_per_move,
    max_examples,
    tactical_repeat_count,
    hidden_size,
    initial_model_data,
    benchmark_state,
    final_model_data,
    game_adapter,
    validation_ratio,
    early_stop_patience,
    stopped_early,
):
    checkpoints = benchmark_state["checkpoints"]
    first_checkpoint = checkpoints[0]
    last_checkpoint = checkpoints[-1]
    best_checkpoint = benchmark_state["best_checkpoint"]

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "started_from_existing_model": initial_model_data is not None,
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "tactical_repeat_count": tactical_repeat_count,
        "base_examples_count": _get_base_examples_count(raw_dataset),
        "extra_examples_count": raw_dataset.get("extra_examples_count", 0),
        "examples_count": encoded_dataset["encoded_examples_count"],
        "training_examples_count": len(training_examples),
        "validation_examples_count": len(validation_examples),
        "validation_ratio": validation_ratio,
        "input_size": encoded_dataset["input_size"],
        "hidden_size": hidden_size,
        "output_size": encoded_dataset["output_size"],
        "checkpoints_count": checkpoints_count,
        "epochs_per_checkpoint": epochs_per_checkpoint,
        "total_epochs": last_checkpoint["total_epochs"],
        "learning_rate": learning_rate,
        "evaluation_games_count": evaluation_games_count,
        "early_stop_patience": early_stop_patience,
        "stopped_early": stopped_early,
        "checkpoints": checkpoints,
        "initial_training_error": first_checkpoint["training_error"],
        "final_training_error": last_checkpoint["training_error"],
        "training_error_improvement": first_checkpoint["training_error"] - last_checkpoint["training_error"],
        "initial_validation_error": first_checkpoint.get("validation_error", 0.0),
        "final_validation_error": last_checkpoint.get("validation_error", 0.0),
        "validation_error_improvement": _get_validation_error_improvement(first_checkpoint, last_checkpoint),
        "initial_evaluation_efficiency": first_checkpoint["evaluation_efficiency"],
        "final_evaluation_efficiency": last_checkpoint["evaluation_efficiency"],
        "evaluation_efficiency_improvement": _get_efficiency_improvement(first_checkpoint, last_checkpoint),
        "initial_tactical_success_rate": first_checkpoint["tactical_success_rate"],
        "final_tactical_success_rate": last_checkpoint["tactical_success_rate"],
        "tactical_success_rate_improvement": _get_tactical_improvement(first_checkpoint, last_checkpoint),
        "best_checkpoint": best_checkpoint,
        "best_checkpoint_index": best_checkpoint["checkpoint_index"],
        "best_total_epochs": best_checkpoint["total_epochs"],
        "best_training_error": best_checkpoint["training_error"],
        "best_validation_error": best_checkpoint.get("validation_error", 0.0),
        "best_evaluation_efficiency": best_checkpoint["evaluation_efficiency"],
        "best_tactical_success_rate": best_checkpoint["tactical_success_rate"],
        "best_model_data": benchmark_state["best_model_data"],
        "final_model_data": final_model_data,
        "final_checkpoint_is_best": best_checkpoint["checkpoint_index"] == last_checkpoint["checkpoint_index"],
    }


def _get_base_examples_count(raw_dataset):
    return raw_dataset.get("base_examples_count", raw_dataset.get("examples_count", 0))


def _get_validation_error_improvement(first_checkpoint, last_checkpoint):
    return first_checkpoint.get("validation_error", 0.0) - last_checkpoint.get("validation_error", 0.0)


def _get_efficiency_improvement(first_checkpoint, last_checkpoint):
    return last_checkpoint["evaluation_efficiency"] - first_checkpoint["evaluation_efficiency"]


def _get_tactical_improvement(first_checkpoint, last_checkpoint):
    return last_checkpoint["tactical_success_rate"] - first_checkpoint["tactical_success_rate"]
