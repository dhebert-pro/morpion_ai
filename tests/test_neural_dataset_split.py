from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_dataset_split import split_encoded_examples
from app.ai.neural_checkpoint import is_checkpoint_better


def create_examples(count):
    examples = []

    for index in range(count):
        examples.append({"state_key": "state_" + str(index)})

    return examples


def test_split_without_validation_keeps_all_examples_for_training():
    examples = create_examples(4)

    result = split_encoded_examples(
        examples,
        validation_ratio=0.0,
        seed=0,
    )

    assert_equal(len(result["training_examples"]), 4)
    assert_equal(len(result["validation_examples"]), 0)


def test_split_with_validation_keeps_at_least_one_training_example():
    examples = create_examples(5)

    result = split_encoded_examples(
        examples,
        validation_ratio=0.4,
        seed=0,
    )

    assert_equal(len(result["training_examples"]), 3)
    assert_equal(len(result["validation_examples"]), 2)


def test_checkpoint_comparison_uses_validation_after_game_metrics():
    current_best = {
        "tactical_success_rate": 100.0,
        "evaluation_efficiency": 80.0,
        "training_error": 0.01,
        "validation_error": 0.20,
    }
    candidate = {
        "tactical_success_rate": 100.0,
        "evaluation_efficiency": 80.0,
        "training_error": 0.05,
        "validation_error": 0.10,
    }

    assert_true(is_checkpoint_better(candidate, current_best))


TESTS = [
    ("Le split sans validation garde tous les exemples", test_split_without_validation_keeps_all_examples_for_training),
    ("Le split de validation conserve un entraînement non vide", test_split_with_validation_keeps_at_least_one_training_example),
    ("La comparaison de checkpoints utilise l'erreur de validation", test_checkpoint_comparison_uses_validation_after_game_metrics),
]
