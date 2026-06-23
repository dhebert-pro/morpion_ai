from tempfile import TemporaryDirectory
from pathlib import Path

from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_model_service import (
    train_and_save_neural_model,
    train_and_save_neural_model_from_package,
    load_neural_model_package,
    get_model_data_from_package,
    evaluate_saved_neural_model_package,
)
from app.ai.tactical_evaluation import get_default_morpion_tactical_probes


def test_train_and_save_neural_model_creates_file():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        model_package = train_and_save_neural_model(
            file_path=file_path,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=60,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=0,
        )

        assert_true(file_path.exists())
        assert_equal(model_package["type"], "neural_model_package")
        assert_equal(model_package["game"], "morpion")
        assert_equal(model_package["trained_player"], "O")
        assert_equal(model_package["opponent_player"], "X")
        assert_true("model_data" in model_package)
        assert_true("training_summary" in model_package)

        training_summary = model_package["training_summary"]

        assert_true(training_summary["examples_count"] > 0)
        assert_equal(training_summary["started_from_existing_model"], False)
        assert_true(training_summary["final_error"] < training_summary["initial_error"])


def test_train_and_save_neural_model_can_include_tactical_examples():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        model_package = train_and_save_neural_model(
            file_path=file_path,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=60,
            learning_rate=0.2,
            tactical_repeat_count=2,
            show_progress=False,
            seed=0,
        )

        training_summary = model_package["training_summary"]

        assert_true(file_path.exists())
        assert_equal(training_summary["tactical_repeat_count"], 2)
        expected_extra_count = len(get_default_morpion_tactical_probes()) * 2

        assert_equal(training_summary["extra_examples_count"], expected_extra_count)
        assert_equal(
            training_summary["examples_count"],
            training_summary["base_examples_count"] + training_summary["extra_examples_count"],
        )


def test_load_neural_model_package_loads_saved_model():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        train_and_save_neural_model(
            file_path=file_path,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=60,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=0,
        )

        loaded_package = load_neural_model_package(file_path)

        assert_equal(loaded_package["type"], "neural_model_package")
        assert_equal(loaded_package["game"], "morpion")
        assert_true("model_data" in loaded_package)
        assert_true("training_summary" in loaded_package)


def test_get_model_data_from_package_returns_empty_without_package():
    model_data = get_model_data_from_package({})

    assert_equal(model_data, {})


def test_get_model_data_from_package_extracts_model_data():
    package = {
        "type": "neural_model_package",
        "model_data": {
            "type": "simple_neural_network",
            "input_size": 18,
        },
    }

    model_data = get_model_data_from_package(package)

    assert_equal(model_data["type"], "simple_neural_network")
    assert_equal(model_data["input_size"], 18)


def test_train_and_save_neural_model_from_package_continues_existing_model():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        first_package = train_and_save_neural_model(
            file_path=file_path,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=40,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=0,
        )

        continued_package = train_and_save_neural_model_from_package(
            file_path=file_path,
            existing_model_package=first_package,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=40,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=1,
        )

        assert_true(file_path.exists())
        assert_equal(continued_package["type"], "neural_model_package")
        assert_equal(
            continued_package["training_summary"]["started_from_existing_model"],
            True,
        )
        assert_true(
            continued_package["training_summary"]["final_error"]
            < continued_package["training_summary"]["initial_error"]
        )


def test_train_and_save_neural_model_from_empty_package_starts_from_zero():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        model_package = train_and_save_neural_model_from_package(
            file_path=file_path,
            existing_model_package={},
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=40,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=1,
        )

        assert_true(file_path.exists())
        assert_equal(
            model_package["training_summary"]["started_from_existing_model"],
            False,
        )


def test_evaluate_saved_neural_model_package_counts_games():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"

        model_package = train_and_save_neural_model(
            file_path=file_path,
            training_games_count=15,
            simulations_per_move=1,
            max_examples=5,
            hidden_size=8,
            epochs=60,
            learning_rate=0.2,
            tactical_repeat_count=0,
            show_progress=False,
            seed=1,
        )

        evaluation = evaluate_saved_neural_model_package(
            model_package,
            games_count=5,
        )

        summary = evaluation["summary"]

        assert_equal(summary["total_games"], 5)
        assert_equal(summary["trained_player"], "O")
        assert_equal(summary["opponent_player"], "X")
        assert_true(0.0 <= summary["efficiency"] <= 100.0)


TESTS = [
    ("L'entraînement neuronal sauvegarde un fichier", test_train_and_save_neural_model_creates_file),
    ("L'entraînement neuronal peut inclure des exemples tactiques", test_train_and_save_neural_model_can_include_tactical_examples),
    ("Le modèle neuronal sauvegardé peut être rechargé", test_load_neural_model_package_loads_saved_model),
    ("L'extraction du modèle retourne vide sans package", test_get_model_data_from_package_returns_empty_without_package),
    ("L'extraction du modèle récupère model_data", test_get_model_data_from_package_extracts_model_data),
    ("L'entraînement neuronal continue depuis un package existant", test_train_and_save_neural_model_from_package_continues_existing_model),
    ("L'entraînement neuronal démarre de zéro sans package existant", test_train_and_save_neural_model_from_empty_package_starts_from_zero),
    ("L'évaluation d'un modèle neuronal sauvegardé compte les parties", test_evaluate_saved_neural_model_package_counts_games),
]