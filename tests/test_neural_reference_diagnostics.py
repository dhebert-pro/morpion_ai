from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork
from app.ai.neural_reference_diagnostics import (
    collect_reference_loss_diagnostics,
    format_board,
    format_reference_loss_diagnostics_report,
    play_traced_neural_against_reference_game,
)
from app.ai.reference_opponents import REFERENCE_TACTICAL


def create_simple_model():
    network = SimpleNeuralNetwork(
        input_size=34,
        hidden_size=3,
        output_size=9,
        learning_rate=0.1,
        seed=0,
    )
    network.input_hidden_weights = [
        [0.0 for _ in range(network.hidden_size)]
        for _ in range(network.input_size)
    ]
    network.hidden_biases = [0.0 for _ in range(network.hidden_size)]
    network.hidden_output_weights = [
        [0.0 for _ in range(network.output_size)]
        for _ in range(network.hidden_size)
    ]
    network.output_biases = [
        0.0, 0.1, 0.2,
        0.3, 1.5, 0.5,
        0.6, 0.7, 0.8,
    ]
    return network.to_dict()


def test_traced_reference_game_contains_moves_and_result():
    trace = play_traced_neural_against_reference_game(
        model_data=create_simple_model(),
        reference_name=REFERENCE_TACTICAL,
    )

    assert_true(trace["result"] in {"X", "O", "draw"})
    assert_true(len(trace["moves"]) > 0)
    assert_equal(len(trace["final_board"]), 9)


def test_loss_diagnostics_counts_games():
    diagnostics = collect_reference_loss_diagnostics(
        model_data=create_simple_model(),
        reference_name=REFERENCE_TACTICAL,
        games_count=5,
        max_losses=2,
        seed=0,
    )

    assert_equal(diagnostics["games_count"], 5)
    assert_true(diagnostics["shown_losses_count"] <= 2)
    assert_equal(diagnostics["reference_name"], REFERENCE_TACTICAL)


def test_format_board_returns_three_rows():
    rows = format_board("XO..O..X.")

    assert_equal(rows, ["  X O .", "  . O .", "  . X ."])


def test_loss_diagnostics_report_contains_summary():
    diagnostics = {
        "reference_name": "tactical",
        "games_count": 10,
        "losses_count": 0,
        "shown_losses_count": 0,
        "losses": [],
        "trained_player": "O",
        "opponent_player": "X",
    }

    text = format_reference_loss_diagnostics_report(diagnostics)

    assert_true("Diagnostic des défaites" in text)
    assert_true("Aucune défaite" in text)


TESTS = [
    ("Une partie diagnostiquée contient résultat et coups", test_traced_reference_game_contains_moves_and_result),
    ("Le diagnostic des défaites compte les parties", test_loss_diagnostics_counts_games),
    ("Le format de plateau retourne trois lignes", test_format_board_returns_three_rows),
    ("Le rapport diagnostic contient un résumé", test_loss_diagnostics_report_contains_summary),
]
