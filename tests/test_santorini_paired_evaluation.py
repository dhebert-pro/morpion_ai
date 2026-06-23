from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork
from app.games.santorini.paired_evaluation import (
    create_santorini_starting_games,
    evaluate_santorini_neural_vs_random_paired,
)


def test_starting_games_are_reproducible():
    first = create_santorini_starting_games(3, seed=0)
    second = create_santorini_starting_games(3, seed=0)
    assert_equal(first, second)


def test_paired_evaluation_returns_two_results():
    network = SimpleNeuralNetwork(input_size=225, hidden_size=4, output_size=144, seed=0)
    result = evaluate_santorini_neural_vs_random_paired(network, games_count=2, seed=0)
    assert_true("neural_results" in result)
    assert_true("baseline_results" in result)
    assert_equal(sum(result["neural_results"].values()), 2)
    assert_equal(sum(result["baseline_results"].values()), 2)


TESTS = [
    ("Santorini départs appairés reproductibles", test_starting_games_are_reproducible),
    ("Santorini évaluation appairée retourne modèle et baseline", test_paired_evaluation_returns_two_results),
]
