from tests.test_helpers import assert_equal

from app.ai.neural_network import SimpleNeuralNetwork
from app.games.santorini.encoding import SANTORINI_INPUT_SIZE
from app.games.santorini.action_index import ACTION_OUTPUT_SIZE
from app.games.santorini.reference_evaluation import (
    evaluate_santorini_neural_vs_reference_paired,
    evaluate_santorini_neural_vs_references_paired,
)


def _network():
    return SimpleNeuralNetwork(
        input_size=SANTORINI_INPUT_SIZE,
        hidden_size=4,
        output_size=ACTION_OUTPUT_SIZE,
        learning_rate=0.01,
        seed=0,
    )


def test_reference_evaluation_returns_neural_and_baseline_results():
    result = evaluate_santorini_neural_vs_reference_paired(
        network=_network(),
        opponent_name="random",
        games_count=3,
        seed=0,
        max_turns=30,
    )

    assert_equal(result["opponent"], "random")
    assert_equal(sum(result["neural_results"].values()), 3)
    assert_equal(sum(result["baseline_results"].values()), 3)


def test_multiple_reference_evaluations_keep_order():
    results = evaluate_santorini_neural_vs_references_paired(
        network=_network(),
        opponent_names=["random", "climber"],
        games_count=2,
        seed=1,
        max_turns=20,
    )

    assert_equal([result["opponent"] for result in results], ["random", "climber"])


TESTS = [
    ("Santorini évaluation référence retourne modèle et baseline", test_reference_evaluation_returns_neural_and_baseline_results),
    ("Santorini évaluations référence gardent l'ordre", test_multiple_reference_evaluations_keep_order),
]
