from tempfile import TemporaryDirectory
from pathlib import Path

from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_model_service import (
    train_and_save_neural_model,
    load_neural_model_package,
    get_model_data_from_package,
    evaluate_saved_neural_model_package,
)


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
        assert_true(training_summary["final_error"] < training_summary["initial_error"])


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
    ("Le modèle neuronal sauvegardé peut être rechargé", test_load_neural_model_package_loads_saved_model),
    ("L'extraction du modèle retourne vide sans package", test_get_model_data_from_package_returns_empty_without_package),
    ("L'extraction du modèle récupère model_data", test_get_model_data_from_package_extracts_model_data),
    ("L'évaluation d'un modèle neuronal sauvegardé compte les parties", test_evaluate_saved_neural_model_package_counts_games),
]