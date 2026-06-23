from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_benchmark import (
    run_neural_training_benchmark,
    format_neural_benchmark_report,
)


def assert_contains(text, expected_part):
    if expected_part not in text:
        raise AssertionError(
            "Texte attendu introuvable : "
            + expected_part
            + "\nTexte obtenu :\n"
            + text
        )


def test_neural_benchmark_returns_checkpoints():
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
    assert_equal(result["checkpoints"][0]["total_epochs"], 0)
    assert_equal(result["checkpoints"][1]["total_epochs"], 5)
    assert_equal(result["checkpoints"][2]["total_epochs"], 10)

    for checkpoint in result["checkpoints"]:
        assert_true("training_error" in checkpoint)
        assert_true("evaluation_efficiency" in checkpoint)
        assert_true("tactical_success_rate" in checkpoint)
        assert_true(0.0 <= checkpoint["evaluation_efficiency"] <= 100.0)
        assert_true(0.0 <= checkpoint["tactical_success_rate"] <= 100.0)


def test_neural_benchmark_can_start_from_existing_model():
    network = SimpleNeuralNetwork(
        input_size=18,
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
        seed=999,
        initial_model_data=network.to_dict(),
    )

    assert_equal(result["started_from_existing_model"], True)
    assert_equal(len(result["checkpoints"]), 2)


def test_format_neural_benchmark_report_contains_key_information():
    result = {
        "game": "morpion",
        "started_from_existing_model": True,
        "examples_count": 300,
        "tactical_repeat_count": 25,
        "hidden_size": 18,
        "checkpoints_count": 2,
        "epochs_per_checkpoint": 40,
        "evaluation_games_count": 100,
        "training_error_improvement": 0.05,
        "evaluation_efficiency_improvement": 7.5,
        "tactical_success_rate_improvement": 25.0,
        "checkpoints": [
            {
                "checkpoint_index": 0,
                "total_epochs": 0,
                "elapsed_seconds": 0.0,
                "training_error": 0.08,
                "tactical_passed_count": 1,
                "tactical_total_count": 4,
                "tactical_success_rate": 25.0,
                "evaluation_efficiency": 55.0,
            },
            {
                "checkpoint_index": 1,
                "total_epochs": 40,
                "elapsed_seconds": 1.25,
                "training_error": 0.03,
                "tactical_passed_count": 2,
                "tactical_total_count": 4,
                "tactical_success_rate": 50.0,
                "evaluation_efficiency": 62.5,
            },
        ],
    }

    text = format_neural_benchmark_report(result)

    assert_contains(text, "Benchmark entraînement neuronal")
    assert_contains(text, "Jeu : morpion")
    assert_contains(text, "Départ depuis modèle existant : True")
    assert_contains(text, "Palier | Époques | Temps (s) | Erreur dataset | Tactique | Efficacité")
    assert_contains(text, "Gain erreur dataset : 0.05")
    assert_contains(text, "Gain efficacité : 7.5 points")
    assert_contains(text, "Gain tactique : 25.0 points")


TESTS = [
    ("Le benchmark neuronal retourne des checkpoints", test_neural_benchmark_returns_checkpoints),
    ("Le benchmark neuronal peut partir d'un modèle existant", test_neural_benchmark_can_start_from_existing_model),
    ("Le rapport de benchmark neuronal contient les informations clés", test_format_neural_benchmark_report_contains_key_information),
]