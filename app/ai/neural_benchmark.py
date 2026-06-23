from time import perf_counter
from app.ai.neural_pipeline import build_augmented_move_score_dataset
from app.ai.neural_encoding import encode_move_score_dataset
from app.ai.neural_training_session import (
    get_examples_from_encoded_dataset,
    create_or_load_neural_network_for_training,
)
from app.ai.neural_dataset_split import split_encoded_examples
from app.ai.neural_checkpoint import get_checkpoint_table_header, is_checkpoint_better
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
from app.ai.neural_benchmark_loop import train_checkpoint_loop
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
    reference_evaluation_games_count=0,
    reference_evaluation_names=None,
    reference_training_games_count=0,
    reference_training_max_examples=0,
    reference_training_names=None,
):
    raw_dataset = build_augmented_move_score_dataset(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        tactical_repeat_count=tactical_repeat_count,
        show_progress=show_progress,
        game_adapter=game_adapter,
        seed=seed,
        reference_training_games_count=reference_training_games_count,
        reference_training_max_examples=reference_training_max_examples,
        reference_training_names=reference_training_names,
    )
    encoded_dataset = encode_move_score_dataset(raw_dataset, game_adapter=game_adapter)
    encoded_examples = get_examples_from_encoded_dataset(encoded_dataset)

    split_result = split_encoded_examples(
        encoded_examples,
        validation_ratio=validation_ratio,
        seed=seed,
        always_train_sources=_get_always_train_sources(reference_training_names),
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
        reference_evaluation_games_count=reference_evaluation_games_count,
        reference_evaluation_names=reference_evaluation_names,
    )

    stopped_early = train_checkpoint_loop(
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
        reference_evaluation_games_count=reference_evaluation_games_count,
        reference_evaluation_names=reference_evaluation_names,
        seed=seed,
        evaluate_and_store_checkpoint=_evaluate_and_store_checkpoint,
        should_stop_early=_should_stop_early,
    )

    return _create_benchmark_result(
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        training_examples=training_examples,
        validation_examples=validation_examples,
        always_train_examples_count=split_result.get("always_train_examples_count", 0),
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
        reference_evaluation_games_count=reference_evaluation_games_count,
        reference_evaluation_names=reference_evaluation_names,
        reference_training_games_count=reference_training_games_count,
        reference_training_max_examples=reference_training_max_examples,
        reference_training_names=reference_training_names,
    )


def _get_always_train_sources(reference_training_names):
    sources = ["tactical_probe"]

    if reference_training_names is None:
        return sources

    for reference_name in reference_training_names:
        sources.append("reference_" + str(reference_name))

    return sources


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


