import random
from time import perf_counter

from app.utils.progress import print_progress


def train_checkpoint_loop(
    network,
    training_examples,
    validation_examples,
    checkpoints_count,
    epochs_per_checkpoint,
    evaluation_games_count,
    show_progress,
    print_checkpoints,
    early_stop_patience,
    benchmark_state,
    start_time,
    game_adapter,
    evaluation_seed,
    evaluate_and_store_checkpoint,
    should_stop_early,
    reference_evaluation_games_count=0,
    reference_evaluation_names=None,
    seed=0,
):
    training_rng = random.Random(seed + 10)

    for checkpoint_index in range(1, checkpoints_count + 1):
        train_one_checkpoint(
            network,
            training_examples,
            checkpoint_index,
            epochs_per_checkpoint,
            checkpoints_count,
            show_progress,
            training_rng,
        )
        elapsed_seconds = perf_counter() - start_time
        checkpoint_is_best = evaluate_and_store_checkpoint(
            benchmark_state=benchmark_state,
            checkpoint_index=checkpoint_index,
            total_epochs=checkpoint_index * epochs_per_checkpoint,
            elapsed_seconds=elapsed_seconds,
            network=network,
            training_examples=training_examples,
            validation_examples=validation_examples,
            evaluation_games_count=evaluation_games_count,
            game_adapter=game_adapter,
            print_checkpoints=print_checkpoints,
            evaluation_seed=evaluation_seed,
            reference_evaluation_games_count=reference_evaluation_games_count,
            reference_evaluation_names=reference_evaluation_names,
        )

        if checkpoint_is_best:
            benchmark_state["checkpoints_without_improvement"] = 0
        else:
            benchmark_state["checkpoints_without_improvement"] += 1

        if should_stop_early(benchmark_state, early_stop_patience):
            return True

    return False


def train_one_checkpoint(
    network,
    training_examples,
    checkpoint_index,
    epochs_per_checkpoint,
    checkpoints_count,
    show_progress,
    training_rng,
):
    for epoch in range(epochs_per_checkpoint):
        epoch_examples = list(training_examples)
        training_rng.shuffle(epoch_examples)

        for example in epoch_examples:
            network.train_one(
                example["inputs"],
                example["targets"],
                example["legal_moves_mask"],
            )

        if show_progress:
            completed_epochs = (checkpoint_index - 1) * epochs_per_checkpoint + epoch + 1
            print_progress(
                "Benchmark entraînement",
                completed_epochs,
                checkpoints_count * epochs_per_checkpoint,
            )
