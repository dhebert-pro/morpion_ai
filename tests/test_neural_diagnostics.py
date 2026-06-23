from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_diagnostics import (
    run_neural_diagnostic,
    format_neural_diagnostic_report,
)


def test_run_neural_diagnostic_returns_training_and_evaluation():
    diagnostic_result = run_neural_diagnostic(
        training_games_count=15,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=80,
        learning_rate=0.2,
        evaluation_games_count=5,
        show_progress=False,
        seed=0,
    )

    training_summary = diagnostic_result["training_summary"]
    evaluation_summary = diagnostic_result["evaluation_summary"]

    assert_equal(diagnostic_result["game"], "morpion")
    assert_true("model_data" in diagnostic_result)
    assert_true("evaluation_results" in diagnostic_result)

    assert_true(training_summary["examples_count"] > 0)
    assert_true(training_summary["examples_count"] <= 5)
    assert_equal(training_summary["input_size"], 34)
    assert_equal(training_summary["output_size"], 9)
    assert_true(training_summary["final_error"] < training_summary["initial_error"])

    assert_equal(evaluation_summary["total_games"], 5)
    assert_equal(evaluation_summary["trained_player"], "O")
    assert_equal(evaluation_summary["opponent_player"], "X")
    assert_true(0.0 <= evaluation_summary["efficiency"] <= 100.0)


def test_format_neural_diagnostic_report_contains_training_and_evaluation():
    diagnostic_result = {
        "training_summary": {
            "game": "morpion",
            "training_games_count": 20,
            "simulations_per_move": 1,
            "max_examples": 5,
            "examples_count": 5,
            "scored_moves_count": 25,
            "average_legal_moves": 5.0,
            "average_best_score": 0.75,
            "input_size": 34,
            "hidden_size": 8,
            "output_size": 9,
            "epochs": 40,
            "learning_rate": 0.2,
            "initial_error": 0.25,
            "final_error": 0.10,
            "error_improvement": 0.15,
        },
        "evaluation_summary": {
            "total_games": 10,
            "trained_player": "O",
            "opponent_player": "X",
            "trained_player_wins": 6,
            "opponent_player_wins": 2,
            "draws": 2,
            "efficiency": 70.0,
        },
    }

    text = format_neural_diagnostic_report(diagnostic_result)

    assert_true("Diagnostic IA neuronale" in text)
    assert_true("Résumé entraînement neuronal" in text)
    assert_true("Résumé évaluation neuronale" in text)
    assert_true("Erreur initiale : 0.25" in text)
    assert_true("Score d'efficacité : 70.0 %" in text)


TESTS = [
    ("Le diagnostic neuronal retourne entraînement et évaluation", test_run_neural_diagnostic_returns_training_and_evaluation),
    ("Le rapport de diagnostic contient entraînement et évaluation", test_format_neural_diagnostic_report_contains_training_and_evaluation),
]