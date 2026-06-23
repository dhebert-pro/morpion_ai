from tests.test_helpers import assert_equal

from app.ai.neural_benchmark import is_checkpoint_better


def test_checkpoint_comparison_prefers_tactical_success_first():
    current_best = {
        "tactical_success_rate": 75.0,
        "evaluation_efficiency": 90.0,
        "training_error": 0.01,
    }
    candidate = {
        "tactical_success_rate": 100.0,
        "evaluation_efficiency": 60.0,
        "training_error": 0.02,
    }

    assert_equal(is_checkpoint_better(candidate, current_best), True)


def test_checkpoint_comparison_uses_efficiency_when_tactical_is_equal():
    current_best = {
        "tactical_success_rate": 100.0,
        "evaluation_efficiency": 60.0,
        "training_error": 0.01,
    }
    candidate = {
        "tactical_success_rate": 100.0,
        "evaluation_efficiency": 70.0,
        "training_error": 0.02,
    }

    assert_equal(is_checkpoint_better(candidate, current_best), True)


def test_checkpoint_comparison_uses_reference_before_random_efficiency():
    current_best = {
        "tactical_success_rate": 100.0,
        "reference_worst_efficiency": 40.0,
        "evaluation_efficiency": 95.0,
        "training_error": 0.01,
    }
    candidate = {
        "tactical_success_rate": 100.0,
        "reference_worst_efficiency": 65.0,
        "evaluation_efficiency": 80.0,
        "training_error": 0.02,
    }

    assert_equal(is_checkpoint_better(candidate, current_best), True)


def test_checkpoint_comparison_prefers_reference_survival_before_reference_score():
    current_best = {
        "tactical_success_rate": 100.0,
        "reference_worst_survival_rate": 90.0,
        "reference_worst_efficiency": 70.0,
        "evaluation_efficiency": 95.0,
        "training_error": 0.01,
    }
    candidate = {
        "tactical_success_rate": 100.0,
        "reference_worst_survival_rate": 100.0,
        "reference_worst_efficiency": 55.0,
        "evaluation_efficiency": 80.0,
        "training_error": 0.02,
    }

    assert_equal(is_checkpoint_better(candidate, current_best), True)


TESTS = [
    ("La comparaison de checkpoints privilégie d'abord la tactique", test_checkpoint_comparison_prefers_tactical_success_first),
    ("La comparaison de checkpoints utilise l'efficacité si la tactique est égale", test_checkpoint_comparison_uses_efficiency_when_tactical_is_equal),
    ("La comparaison de checkpoints utilise la référence avant le hasard", test_checkpoint_comparison_uses_reference_before_random_efficiency),
    ("La comparaison de checkpoints privilégie la survie référence", test_checkpoint_comparison_prefers_reference_survival_before_reference_score),
]
