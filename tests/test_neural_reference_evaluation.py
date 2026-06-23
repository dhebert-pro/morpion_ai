from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork
from app.ai.neural_reference_evaluation import (
    evaluate_neural_model_against_reference,
    evaluate_neural_model_against_references,
    format_neural_reference_evaluation_report,
    play_neural_against_reference_game,
)
from app.ai.reference_opponents import (
    REFERENCE_GREEDY,
    REFERENCE_RANDOM,
    REFERENCE_TACTICAL,
    choose_reference_opponent_move,
    score_result_for_player,
)
from app.games.morpion.rules import create_new_game


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


def test_score_result_for_player_is_generic():
    assert_equal(score_result_for_player("O", "O"), 1.0)
    assert_equal(score_result_for_player("draw", "O"), 0.5)
    assert_equal(score_result_for_player("ongoing", "O"), 0.5)
    assert_equal(score_result_for_player("X", "O"), 0.0)


def test_reference_random_move_is_legal():
    game = create_new_game()

    move = choose_reference_opponent_move(
        game=game,
        player="X",
        opponent="O",
        reference_name=REFERENCE_RANDOM,
    )

    assert_true(move in range(9))


def test_reference_tactical_blocks_immediate_loss():
    game = create_new_game()
    game["board"] = ["O", "O", None, "X", None, None, None, None, None]

    move = choose_reference_opponent_move(
        game=game,
        player="X",
        opponent="O",
        reference_name=REFERENCE_TACTICAL,
    )

    assert_equal(move, 2)


def test_reference_greedy_takes_immediate_win():
    game = create_new_game()
    game["board"] = ["X", "X", None, "O", None, None, None, None, None]

    move = choose_reference_opponent_move(
        game=game,
        player="X",
        opponent="O",
        reference_name=REFERENCE_GREEDY,
    )

    assert_equal(move, 2)


def test_play_neural_against_reference_game_returns_valid_result():
    result = play_neural_against_reference_game(
        model_data=create_simple_model(),
        reference_name=REFERENCE_TACTICAL,
    )

    assert_true(result in {"X", "O", "draw"})


def test_reference_evaluation_counts_all_games():
    evaluation = evaluate_neural_model_against_reference(
        model_data=create_simple_model(),
        reference_name=REFERENCE_RANDOM,
        games_count=10,
        seed=0,
    )

    total = (
        evaluation["trained_player_wins"]
        + evaluation["opponent_player_wins"]
        + evaluation["draws"]
    )

    assert_equal(total, 10)
    assert_equal(evaluation["reference_name"], REFERENCE_RANDOM)


def test_reference_evaluation_runs_default_opponents():
    evaluations = evaluate_neural_model_against_references(
        model_data=create_simple_model(),
        games_count=3,
        seed=0,
    )

    assert_equal(len(evaluations), 3)


def test_format_reference_evaluation_report_contains_all_blocks():
    evaluations = [
        {
            "reference_name": "random",
            "total_games": 10,
            "trained_player": "O",
            "opponent_player": "X",
            "trained_player_wins": 7,
            "opponent_player_wins": 1,
            "draws": 2,
            "win_rate": 70.0,
            "draw_rate": 20.0,
            "loss_rate": 10.0,
            "survival_rate": 90.0,
            "efficiency": 80.0,
        },
        {
            "reference_name": "tactical",
            "total_games": 10,
            "trained_player": "O",
            "opponent_player": "X",
            "trained_player_wins": 4,
            "opponent_player_wins": 2,
            "draws": 4,
            "win_rate": 40.0,
            "draw_rate": 40.0,
            "loss_rate": 20.0,
            "survival_rate": 80.0,
            "efficiency": 60.0,
        },
    ]

    text = format_neural_reference_evaluation_report(evaluations)

    assert_true("adversaires de référence" in text)
    assert_true("Adversaire : random" in text)
    assert_true("Adversaire : tactical" in text)
    assert_true("Taux de survie : 90.0 %" in text)
    assert_true("Taux de victoire : 70.0 %" in text)
    assert_true("Score d'efficacité : 80.0 %" in text)


TESTS = [
    ("Le score de référence est générique", test_score_result_for_player_is_generic),
    ("L'adversaire aléatoire choisit un coup légal", test_reference_random_move_is_legal),
    ("L'adversaire tactique bloque une menace", test_reference_tactical_blocks_immediate_loss),
    ("L'adversaire glouton prend une victoire immédiate", test_reference_greedy_takes_immediate_win),
    ("Une partie contre référence retourne un résultat valide", test_play_neural_against_reference_game_returns_valid_result),
    ("L'évaluation référence compte toutes les parties", test_reference_evaluation_counts_all_games),
    ("L'évaluation référence lance les adversaires par défaut", test_reference_evaluation_runs_default_opponents),
    ("Le rapport référence affiche les blocs", test_format_reference_evaluation_report_contains_all_blocks),
]
