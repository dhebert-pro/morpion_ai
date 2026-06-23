from tempfile import TemporaryDirectory
from pathlib import Path

from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_model_service import train_and_save_neural_model_from_package


def test_train_and_save_neural_model_from_incompatible_package_restarts():
    with TemporaryDirectory() as temporary_directory:
        file_path = Path(temporary_directory) / "neural_model.json"
        old_package = {
            "type": "neural_model_package",
            "model_data": {
                "type": "simple_neural_network",
                "input_size": 18,
                "hidden_size": 8,
                "output_size": 9,
            },
        }

        model_package = train_and_save_neural_model_from_package(
            file_path=file_path,
            existing_model_package=old_package,
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


TESTS = [
    ("L'entraînement neuronal redémarre si le package est incompatible", test_train_and_save_neural_model_from_incompatible_package_restarts),
]
