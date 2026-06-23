from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.evaluation_summary import (
    format_o_evaluation_comparison,
    summarize_o_results,
)
from app.games.santorini.random_baseline import evaluate_santorini_random_o_vs_random


def test_santorini_o_summary_counts_efficiency():
    summary = summarize_o_results({"X": 1, "O": 2, "draw": 1})
    assert_equal(summary["total_games"], 4)
    assert_equal(summary["wins_o"], 2)
    assert_equal(summary["efficiency"], 62.5)


def test_santorini_comparison_shows_delta():
    neural = summarize_o_results({"X": 1, "O": 3, "draw": 0})
    baseline = summarize_o_results({"X": 2, "O": 2, "draw": 0})
    text = format_o_evaluation_comparison(neural, baseline)
    assert_true("Référence random O contre random X" in text)
    assert_true("Écart modèle - random" in text)
    assert_true("+25.0" in text)


def test_santorini_random_baseline_runs():
    results = evaluate_santorini_random_o_vs_random(games_count=2, seed=0)
    total = results["X"] + results["O"] + results["draw"]
    assert_equal(total, 2)


TESTS = [
    ("Santorini résumé O calcule l'efficacité", test_santorini_o_summary_counts_efficiency),
    ("Santorini comparaison affiche l'écart", test_santorini_comparison_shows_delta),
    ("Santorini référence random est évaluable", test_santorini_random_baseline_runs),
]
