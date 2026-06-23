from tests.test_helpers import assert_true

from app.ai.neural_benchmark_diagnostics import build_benchmark_diagnostic_lines


def assert_contains(text, expected_part):
    if expected_part not in text:
        raise AssertionError("Texte attendu introuvable : " + expected_part)


def test_benchmark_diagnostics_detect_overfitting_signal():
    result = {
        "training_error_improvement": 0.2,
        "validation_error_improvement": -0.1,
        "final_training_error": 0.05,
        "final_validation_error": 0.15,
        "final_checkpoint_is_best": False,
        "best_checkpoint_index": 3,
        "checkpoints": [
            {"evaluation_efficiency": 40.0, "tactical_success_rate": 25.0},
            {"evaluation_efficiency": 60.0, "tactical_success_rate": 50.0},
        ],
    }

    lines = build_benchmark_diagnostic_lines(result)
    text = "\n".join(lines)

    assert_contains(text, "Signal d'overfitting")
    assert_contains(text, "palier 3")


def test_benchmark_diagnostics_handles_missing_checkpoints():
    lines = build_benchmark_diagnostic_lines({"checkpoints": []})

    assert_true(lines[0].startswith("Lecture diagnostic"))


TESTS = [
    (
        "Le diagnostic de benchmark détecte l'overfitting",
        test_benchmark_diagnostics_detect_overfitting_signal,
    ),
    (
        "Le diagnostic de benchmark accepte un historique vide",
        test_benchmark_diagnostics_handles_missing_checkpoints,
    ),
]
