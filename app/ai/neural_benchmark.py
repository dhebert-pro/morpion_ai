from time import perf_counter
import random

from app.ai.neural_pipeline import build_augmented_move_score_dataset
from app.ai.neural_encoding import encode_move_score_dataset
from app.ai.neural_training_session import (
    get_examples_from_encoded_dataset,
    create_or_load_neural_network_for_training,
)
from app.ai.neural_dataset_split import split_encoded_examples
from app.ai.neural_checkpoint import (
    is_checkpoint_better,
    format_checkpoint_line,
    get_checkpoint_table_header,
    get_best_checkpoint_from_benchmark_result,
)
from app.ai.neural_benchmark_state import (
    evaluate_and_store_checkpoint as _evaluate_and_store_checkpoint,
    should_stop_early as _should_stop_early,
    create_benchmark_result as _create_benchmark_result,
)
from app.ai.neural_benchmark_package import (
    create_training_summary_from_benchmark_result,
    create_model_package_from_benchmark_result,
)
from app.ai.neural_benchmark_report import format_neural_benchmark_report
from app.utils.progress import print_progress
from app.games.morpion.adapter import MORPION_ADAPTER


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
    validation_ratio=0.0,
    early_stop_patience=None,
    evaluation_seed=None,
):
    """Entraîne par paliers et garde le meilleur modèle rencontré."""

    raw_dataset = build_augmented_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        game_adapter=game_adapter,
        seed=seed,
    )
    encoded_dataset = encode_move_score_dataset(raw_dataset, game_adapter=game_adapter)
    encoded_examples = get_examples_from_encoded_dataset(encoded_dataset)

    split_result = split_encoded_examples(
        encoded_examples,
        validation_ratio=validation_ratio,
        seed=seed,
    )
    training_examples = split_result["training_examples"]
    validation_examples = split_result["validation_examples"]

    network = create_or_load_neural_network_for_training(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
        initial_model_data=initial_model_data,
    )

    effective_evaluation_seed = _get_effective_evaluation_seed(
        seed,
        evaluation_seed,
    )

    benchmark_state = _create_initial_state(print_checkpoints)
    start_time = perf_counter()

    _evaluate_and_store_checkpoint(
        benchmark_state=benchmark_state,
        checkpoint_index=0,
        total_epochs=0,
        elapsed_seconds=0.0,
        network=network,
        training_examples=training_examples,
        validation_examples=validation_examples,
        evaluation_games_count=evaluation_games_count,
        game_adapter=game_adapter,
        print_checkpoints=print_checkpoints,
        evaluation_seed=effective_evaluation_seed,
    )

    stopped_early = _train_checkpoint_loop(
        network=network,
        training_examples=training_examples,
        validation_examples=validation_examples,
        checkpoints_count=checkpoints_count,
        epochs_per_checkpoint=epochs_per_checkpoint,
        evaluation_games_count=evaluation_games_count,
        show_progress=show_progress,
        print_checkpoints=print_checkpoints,
        early_stop_patience=early_stop_patience,
        benchmark_state=benchmark_state,
        start_time=start_time,
        game_adapter=game_adapter,
        evaluation_seed=effective_evaluation_seed,
        seed=seed,
    )

    return _create_benchmark_result(
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        training_examples=training_examples,
        validation_examples=validation_examples,
        checkpoints_count=checkpoints_count,
        epochs_per_checkpoint=epochs_per_checkpoint,
        learning_rate=learning_rate,
        evaluation_games_count=evaluation_games_count,
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        tactical_repeat_count=tactical_repeat_count,
        hidden_size=hidden_size,
        initial_model_data=initial_model_data,
        benchmark_state=benchmark_state,
        final_model_data=network.to_dict(),
        game_adapter=game_adapter,
        validation_ratio=validation_ratio,
        early_stop_patience=early_stop_patience,
        stopped_early=stopped_early,
        evaluation_seed=effective_evaluation_seed,
    )


def _get_effective_evaluation_seed(training_seed, evaluation_seed):
    if evaluation_seed is None:
        return training_seed

    return evaluation_seed


def _create_initial_state(print_checkpoints):
    if print_checkpoints:
        print()
        print(get_checkpoint_table_header())

    return {
        "checkpoints": [],
        "best_checkpoint": None,
        "best_model_data": None,
        "checkpoints_without_improvement": 0,
    }


def _train_checkpoint_loop(
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
    seed=0,
):
    training_rng = random.Random(seed + 10)

    for checkpoint_index in range(1, checkpoints_count + 1):
        _train_one_checkpoint(
            network,
            training_examples,
            checkpoint_index,
            epochs_per_checkpoint,
            checkpoints_count,
            show_progress,
            training_rng,
        )
        elapsed_seconds = perf_counter() - start_time
        checkpoint_is_best = _evaluate_and_store_checkpoint(
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
        )

        if checkpoint_is_best:
            benchmark_state["checkpoints_without_improvement"] = 0
        else:
            benchmark_state["checkpoints_without_improvement"] += 1

        if _should_stop_early(benchmark_state, early_stop_patience):
            return True

    return False


def _train_one_checkpoint(
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
