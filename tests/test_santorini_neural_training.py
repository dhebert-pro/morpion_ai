from pathlib import Path
import tempfile

from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.neural_training import (
    build_train_and_save_santorini_neural_model,
    format_santorini_training_summary,
    get_santorini_model_data,
    load_santorini_model_package,
)
from app.games.santorini.neural_player import (
    evaluate_santorini_neural_vs_random,
    summarize_santorini_evaluation,
)


def test_santorini_training_saves_model_and_dataset():
    with tempfile.TemporaryDirectory() as directory:
        model_file = Path(directory) / "santorini_model.json"
        dataset_file = Path(directory) / "santorini_dataset.json"
        package = build_train_and_save_santorini_neural_model(
            model_file=model_file,
            dataset_file=dataset_file,
            games_count=3,
            max_examples=1,
            simulations_per_move=1,
            hidden_size=4,
            epochs=1,
            learning_rate=0.04,
            seed=0,
        )
        assert_true(model_file.exists())
        assert_true(dataset_file.exists())
        assert_equal(package["game"], "santorini")
        assert_true("model_data" in package)


def test_santorini_model_loader_rejects_other_games():
    model_data = get_santorini_model_data({"game": "morpion", "model_data": {"x": 1}})
    assert_equal(model_data, {})


def test_santorini_saved_model_can_be_evaluated():
    with tempfile.TemporaryDirectory() as directory:
        model_file = Path(directory) / "santorini_model.json"
        dataset_file = Path(directory) / "santorini_dataset.json"
        build_train_and_save_santorini_neural_model(
            model_file=model_file,
            dataset_file=dataset_file,
            games_count=3,
            max_examples=1,
            simulations_per_move=1,
            hidden_size=4,
            epochs=1,
            learning_rate=0.04,
            seed=0,
        )
        package = load_santorini_model_package(model_file)
        model_data = get_santorini_model_data(package)
        results = evaluate_santorini_neural_vs_random(model_data, games_count=2, seed=0)
        summary = summarize_santorini_evaluation(results)
        assert_equal(summary["total_games"], 2)
        assert_true(0 <= summary["efficiency"] <= 100)


def test_santorini_training_summary_is_readable():
    with tempfile.TemporaryDirectory() as directory:
        model_file = Path(directory) / "santorini_model.json"
        dataset_file = Path(directory) / "santorini_dataset.json"
        package = build_train_and_save_santorini_neural_model(
            model_file=model_file,
            dataset_file=dataset_file,
            games_count=3,
            max_examples=1,
            simulations_per_move=1,
            hidden_size=4,
            epochs=1,
            learning_rate=0.04,
            seed=0,
        )
        text = format_santorini_training_summary(package["training_summary"])
        assert_true("Résumé entraînement neuronal Santorini" in text)
        assert_true("Taille entrée" in text)


TESTS = [
    ("Santorini entraînement sauvegarde modèle et dataset", test_santorini_training_saves_model_and_dataset),
    ("Santorini loader refuse les autres jeux", test_santorini_model_loader_rejects_other_games),
    ("Santorini modèle sauvegardé est évaluable", test_santorini_saved_model_can_be_evaluated),
    ("Santorini résumé d'entraînement est lisible", test_santorini_training_summary_is_readable),
]
