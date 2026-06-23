from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork
from app.games.santorini.action_index import ACTION_OUTPUT_SIZE
from app.games.santorini.encoding import SANTORINI_INPUT_SIZE
from app.games.santorini.reference_diagnostics import (
    diagnose_santorini_neural_vs_reference,
    format_santorini_reference_diagnostic,
)


def _network():
    return SimpleNeuralNetwork(
        input_size=SANTORINI_INPUT_SIZE,
        hidden_size=4,
        output_size=ACTION_OUTPUT_SIZE,
        learning_rate=0.01,
        seed=0,
    )


def test_santorini_reference_diagnostic_counts_games():
    report = diagnose_santorini_neural_vs_reference(
        network=_network(),
        opponent_name="random",
        games_count=2,
        seed=0,
        max_turns=20,
        max_details=1,
    )

    assert_equal(report["opponent"], "random")
    assert_equal(report["games_count"], 2)
    assert_true(report["losses_count"] >= 0)
    assert_true(len(report["details"]) <= 1)


def test_santorini_reference_diagnostic_format_is_readable():
    report = {
        "opponent": "climber",
        "games_count": 3,
        "losses_count": 0,
        "details": [],
    }

    text = format_santorini_reference_diagnostic(report)

    assert_true("Diagnostic Santorini" in text)
    assert_true("Défaites trouvées : 0" in text)


TESTS = [
    (
        "Santorini diagnostic référence compte les parties",
        test_santorini_reference_diagnostic_counts_games,
    ),
    (
        "Santorini diagnostic référence est lisible",
        test_santorini_reference_diagnostic_format_is_readable,
    ),
]
