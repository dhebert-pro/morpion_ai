import random

from app.ai.neural_training_session import (
    compute_average_error_on_encoded_examples,
    create_neural_network_from_encoded_dataset,
)
from app.games.santorini.dataset import build_santorini_move_score_dataset
from app.games.santorini.evaluation_summary import summarize_o_results
from app.games.santorini.neural_dataset import encode_santorini_move_score_dataset
from app.games.santorini.paired_evaluation import evaluate_santorini_neural_vs_random_paired
from app.games.santorini.neural_training import create_santorini_model_package
from app.storage.json_storage import save_json
from app.utils.progress import print_progress


def train_santorini_neural_with_watch(
    model_file,
    dataset_file,
    games_count,
    max_examples,
    simulations_per_move,
    hidden_size,
    checkpoints_count,
    epochs_per_checkpoint,
    learning_rate,
    evaluation_games_count,
    seed=0,
    show_progress=False,
):
    raw_dataset = build_santorini_move_score_dataset(
        games_count=games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        seed=seed,
        show_progress=show_progress,
    )
    encoded_dataset = encode_santorini_move_score_dataset(raw_dataset)
    save_json(raw_dataset, dataset_file)

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    examples = encoded_dataset["examples"]
    training_rng = random.Random(seed + 50)
    rows = []
    best_row = None
    best_model_data = None

    for checkpoint in range(checkpoints_count + 1):
        row = _evaluate_checkpoint(
            network=network,
            examples=examples,
            checkpoint=checkpoint,
            epochs=checkpoint * epochs_per_checkpoint,
            evaluation_games_count=evaluation_games_count,
            seed=seed,
        )
        rows.append(row)

        if _is_better_santorini_checkpoint(row, best_row):
            best_row = row
            best_model_data = network.to_dict()

        if checkpoint < checkpoints_count:
            _train_santorini_checkpoint(
                network=network,
                examples=examples,
                epochs=epochs_per_checkpoint,
                rng=training_rng,
            )

            if show_progress:
                print_progress(
                    "Benchmark Santorini  ",
                    checkpoint + 1,
                    checkpoints_count,
                )

    package = _create_watch_package(
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        best_model_data=best_model_data,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        checkpoints_count=checkpoints_count,
        epochs_per_checkpoint=epochs_per_checkpoint,
        evaluation_games_count=evaluation_games_count,
        rows=rows,
        best_row=best_row,
    )
    save_json(package, model_file)
    return package


def _train_santorini_checkpoint(network, examples, epochs, rng):
    for _ in range(epochs):
        epoch_examples = list(examples)
        rng.shuffle(epoch_examples)

        for example in epoch_examples:
            network.train_one(
                example["inputs"],
                example["targets"],
                example["legal_moves_mask"],
            )


def _evaluate_checkpoint(
    network,
    examples,
    checkpoint,
    epochs,
    evaluation_games_count,
    seed,
):
    error = compute_average_error_on_encoded_examples(network, examples)
    paired = evaluate_santorini_neural_vs_random_paired(
        network,
        games_count=evaluation_games_count,
        seed=seed,
    )
    summary = summarize_o_results(paired["neural_results"])
    baseline_summary = summarize_o_results(paired["baseline_results"])
    delta = summary["efficiency"] - baseline_summary["efficiency"]

    return {
        "checkpoint": checkpoint,
        "epochs": epochs,
        "error": error,
        "wins_x": summary["wins_x"],
        "wins_o": summary["wins_o"],
        "draws": summary["draws"],
        "efficiency": summary["efficiency"],
        "baseline_efficiency": baseline_summary["efficiency"],
        "baseline_wins_x": baseline_summary["wins_x"],
        "baseline_wins_o": baseline_summary["wins_o"],
        "baseline_draws": baseline_summary["draws"],
        "delta_vs_random": delta,
    }


def _is_better_santorini_checkpoint(candidate, current_best):
    if current_best is None:
        return True

    return _checkpoint_key(candidate) > _checkpoint_key(current_best)


def _checkpoint_key(row):
    return (row["delta_vs_random"], row["efficiency"], -row["wins_x"], -row["error"])


def _create_watch_package(
    raw_dataset,
    encoded_dataset,
    best_model_data,
    hidden_size,
    learning_rate,
    checkpoints_count,
    epochs_per_checkpoint,
    evaluation_games_count,
    rows,
    best_row,
):
    fake_training_result = {
        "model_data": best_model_data,
        "input_size": encoded_dataset["input_size"],
        "hidden_size": hidden_size,
        "output_size": encoded_dataset["output_size"],
        "initial_error": rows[0]["error"],
        "final_error": best_row["error"],
    }
    package = create_santorini_model_package(
        raw_dataset=raw_dataset,
        encoded_dataset=encoded_dataset,
        training_result=fake_training_result,
        hidden_size=hidden_size,
        epochs=best_row["epochs"],
        learning_rate=learning_rate,
    )
    package["training_summary"].update(_create_watch_summary_fields(
        checkpoints_count,
        epochs_per_checkpoint,
        evaluation_games_count,
        best_row,
    ))
    package["watch_rows"] = rows
    return package


def _create_watch_summary_fields(
    checkpoints_count, epochs_per_checkpoint, evaluation_games_count, best_row
):
    return {
        "mode": "watch",
        "checkpoints_count": checkpoints_count,
        "epochs_per_checkpoint": epochs_per_checkpoint,
        "evaluation_games_count": evaluation_games_count,
        "best_checkpoint": best_row["checkpoint"],
        "best_delta_vs_random": best_row["delta_vs_random"],
        "best_baseline_efficiency": best_row["baseline_efficiency"],
        "paired_evaluation": True,
        "best_efficiency": best_row["efficiency"],
        "best_wins_x": best_row["wins_x"],
        "best_wins_o": best_row["wins_o"],
        "best_draws": best_row["draws"],
    }


def format_santorini_watch_table(package):
    lines = []
    lines.append("Comparaison appairée : mêmes placements de départ pour modèle et random.")
    lines.append("Palier | Époques | Erreur | Modèle O | Random O | Écart | Résultat")
    best_checkpoint = package["training_summary"].get("best_checkpoint")

    for row in package.get("watch_rows", []):
        marker = "  <- meilleur" if row["checkpoint"] == best_checkpoint else ""
        lines.append(_format_watch_row(row) + marker)

    return "\n".join(lines)


def _format_watch_row(row):
    result = f'X {row["wins_x"]} / O {row["wins_o"]} / N {row["draws"]}'
    return (
        f'{row["checkpoint"]} | {row["epochs"]} | {round(row["error"], 6)} | '
        f'{round(row["efficiency"], 2)} % | '
        f'{round(row["baseline_efficiency"], 2)} % | '
        f'{_format_delta(row["delta_vs_random"])} | {result}'
    )


def _format_delta(value):
    rounded = round(value, 2)
    return "+" + str(rounded) if rounded > 0 else str(rounded)
