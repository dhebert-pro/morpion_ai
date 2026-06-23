from pathlib import Path
import tempfile

from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.neural_watch import (
    format_santorini_watch_table,
    train_santorini_neural_with_watch,
)


def test_santorini_watch_saves_best_checkpoint_model():
    with tempfile.TemporaryDirectory() as directory:
        model_file = Path(directory) / "santorini_model.json"
        dataset_file = Path(directory) / "santorini_dataset.json"
        package = train_santorini_neural_with_watch(
            model_file=model_file,
            dataset_file=dataset_file,
            games_count=3,
            max_examples=1,
            simulations_per_move=1,
            hidden_size=4,
            checkpoints_count=1,
            epochs_per_checkpoint=1,
            learning_rate=0.04,
            evaluation_games_count=2,
            seed=0,
        )
        assert_true(model_file.exists())
        assert_true(dataset_file.exists())
        assert_equal(package["game"], "santorini")
        assert_true("watch_rows" in package)
        assert_equal(len(package["watch_rows"]), 2)
        assert_true("best_checkpoint" in package["training_summary"])


def test_santorini_watch_table_marks_best_checkpoint():
    package = {
        "training_summary": {"best_checkpoint": 1},
        "watch_rows": [
            _row(0, 0, 40.0, 50.0),
            _row(1, 10, 60.0, 50.0),
        ],
    }
    text = format_santorini_watch_table(package)
    assert_true("Palier" in text)
    assert_true("<- meilleur" in text)
    assert_true("+10.0" in text)


def _row(checkpoint, epochs, efficiency, baseline):
    return {
        "checkpoint": checkpoint,
        "epochs": epochs,
        "error": 0.1,
        "wins_x": 0,
        "wins_o": 1,
        "draws": 1,
        "efficiency": efficiency,
        "baseline_efficiency": baseline,
        "delta_vs_random": efficiency - baseline,
    }


TESTS = [
    ("Santorini watch sauvegarde le meilleur checkpoint", test_santorini_watch_saves_best_checkpoint_model),
    ("Santorini watch table marque le meilleur checkpoint", test_santorini_watch_table_marks_best_checkpoint),
]
