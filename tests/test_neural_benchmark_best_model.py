from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork
from app.ai.neural_benchmark import (
    run_neural_training_benchmark,
    create_training_summary_from_benchmark_result,
    create_model_package_from_benchmark_result,
    format_neural_benchmark_report,
)


def assert_contains(text, expected_part):
    if expected_part not in text:
        raise AssertionError("Texte attendu introuvable : " + expected_part)


def test_neural_benchmark_returns_checkpoints_and_best_model():
    result = run_neural_training_benchmark(
        training_games_count=10,
        simulations_per_move=1,
        max_examples=3,
        tactical_repeat_count=0,
        hidden_size=8,
        checkpoints_count=2,
        epochs_per_checkpoint=5,
        learning_rate=0.2,
        evaluation_games_count=3,
        show_progress=False,
        print_checkpoints=False,
        seed=0,
    )

    assert_equal(result["game"], "morpion")
    assert_equal(result["started_from_existing_model"], False)
    assert_true(result["examples_count"] > 0)
    assert_equal(result["hidden_size"], 8)
    assert_equal(result["checkpoints_count"], 2)
    assert_equal(result["epochs_per_checkpoint"], 5)
    assert_equal(result["total_epochs"], 10)
    assert_equal(len(result["checkpoints"]), 3)
    assert_true("best_checkpoint" in result)
    assert_true("best_model_data" in result)

    for checkpoint in result["checkpoints"]:
        assert_true("training_error" in checkpoint)
        assert_true("evaluation_efficiency" in checkpoint)
        assert_true("tactical_success_rate" in checkpoint)


def test_neural_benchmark_can_start_from_existing_model():
    network = SimpleNeuralNetwork(
        input_size=34,
        hidden_size=8,
        output_size=9,
        learning_rate=0.2,
        seed=0,
    )

    result = run_neural_training_benchmark(
        training_games_count=10,
        simulations_per_move=1,
        max_examples=3,
        tactical_repeat_count=0,
        hidden_size=8,
        checkpoints_count=1,
        epochs_per_checkpoint=5,
        learning_rate=0.2,
        evaluation_games_count=3,
        show_progress=False,
        print_checkpoints=False,
        seed=999,
        initial_model_data=network.to_dict(),
    )

    assert_equal(result["started_from_existing_model"], True)
    assert_equal(len(result["checkpoints"]), 2)


def test_training_summary_can_be_created_from_best_benchmark_checkpoint():
    result = create_benchmark_result_fixture()

    summary = create_training_summary_from_benchmark_result(result)

    assert_equal(summary["game"], "morpion")
    assert_equal(summary["examples_count"], 11)
    assert_equal(summary["epochs"], 10)
    assert_equal(summary["started_from_existing_model"], True)
    assert_equal(summary["initial_error"], 0.5)
    assert_equal(summary["final_error"], 0.2)
    assert_equal(summary["error_improvement"], 0.3)


def test_model_package_uses_best_model_from_benchmark_result():
    result = create_benchmark_result_fixture()
    best_network = SimpleNeuralNetwork(18, 8, 9, learning_rate=0.2, seed=1)
    best_model_data = best_network.to_dict()
    best_model_data["output_biases"][0] = 9.0
    result["best_model_data"] = best_model_data
    result["final_model_data"] = SimpleNeuralNetwork(18, 8, 9).to_dict()

    package = create_model_package_from_benchmark_result(result)

    assert_equal(package["type"], "neural_model_package")
    assert_equal(package["model_data"]["output_biases"][0], 9.0)
    assert_equal(package["training_summary"]["epochs"], 10)
    assert_equal(package["benchmark_summary"]["best_checkpoint_index"], 1)


def test_format_neural_benchmark_report_contains_best_checkpoint_information():
    result = create_benchmark_result_fixture()
    result["checkpoints"] = [
        create_checkpoint(0, 0, 0.08, 55.0, 25.0),
        create_checkpoint(1, 40, 0.03, 62.5, 50.0),
        create_checkpoint(2, 80, 0.02, 58.0, 50.0),
    ]

    text = format_neural_benchmark_report(result)

    assert_contains(text, "Benchmark entraînement neuronal")
    assert_contains(text, "Jeu : morpion")
    assert_contains(text, "Exemples totaux : 11")
    assert_contains(text, "<- meilleur")
    assert_contains(text, "Meilleur palier : 1 (10 époques)")
    assert_contains(text, "Modèle retenu : meilleur palier, pas le dernier.")


def create_benchmark_result_fixture():
    return {
        "game": "morpion",
        "trained_player": "O",
        "opponent_player": "X",
        "training_games_count": 10,
        "simulations_per_move": 1,
        "max_examples": 3,
        "tactical_repeat_count": 2,
        "base_examples_count": 3,
        "extra_examples_count": 8,
        "examples_count": 11,
        "input_size": 34,
        "hidden_size": 8,
        "output_size": 9,
        "total_epochs": 20,
        "learning_rate": 0.2,
        "started_from_existing_model": True,
        "checkpoints_count": 2,
        "epochs_per_checkpoint": 10,
        "evaluation_games_count": 3,
        "initial_training_error": 0.5,
        "final_training_error": 0.25,
        "training_error_improvement": 0.25,
        "initial_evaluation_efficiency": 40.0,
        "final_evaluation_efficiency": 50.0,
        "evaluation_efficiency_improvement": 10.0,
        "initial_tactical_success_rate": 25.0,
        "final_tactical_success_rate": 50.0,
        "tactical_success_rate_improvement": 25.0,
        "best_checkpoint": create_checkpoint(1, 10, 0.2, 60.0, 75.0),
        "best_checkpoint_index": 1,
        "best_total_epochs": 10,
        "final_checkpoint_is_best": False,
        "best_model_data": SimpleNeuralNetwork(18, 8, 9).to_dict(),
        "final_model_data": SimpleNeuralNetwork(18, 8, 9).to_dict(),
        "checkpoints": [],
    }


def create_checkpoint(index, epochs, error, efficiency, tactical):
    return {
        "checkpoint_index": index,
        "total_epochs": epochs,
        "elapsed_seconds": 0.0,
        "training_error": error,
        "evaluation_efficiency": efficiency,
        "tactical_passed_count": 2,
        "tactical_total_count": 4,
        "tactical_success_rate": tactical,
    }


TESTS = [
    ("Le benchmark neuronal retourne des checkpoints et le meilleur modèle", test_neural_benchmark_returns_checkpoints_and_best_model),
    ("Le benchmark neuronal peut partir d'un modèle existant", test_neural_benchmark_can_start_from_existing_model),
    ("Un résumé d'entraînement utilise le meilleur checkpoint du benchmark", test_training_summary_can_be_created_from_best_benchmark_checkpoint),
    ("Un package de modèle utilise le meilleur modèle du benchmark", test_model_package_uses_best_model_from_benchmark_result),
    ("Le rapport de benchmark indique le meilleur checkpoint", test_format_neural_benchmark_report_contains_best_checkpoint_information),
]
