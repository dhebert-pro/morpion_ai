from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_evaluation import (
    play_neural_automatic_game,
    evaluate_neural_model,
    summarize_neural_evaluation_results,
    format_neural_evaluation_summary,
)


def create_deterministic_model():
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

    network.hidden_biases = [
        0.0 for _ in range(network.hidden_size)
    ]

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


def test_play_neural_automatic_game_returns_valid_result():
    model_data = create_deterministic_model()

    result = play_neural_automatic_game(model_data)

    assert_true(result in {"X", "O", "draw"})


def test_evaluate_neural_model_counts_all_games():
    model_data = create_deterministic_model()

    results = evaluate_neural_model(
        model_data=model_data,
        games_count=10,
    )

    assert_equal(set(results.keys()), {"X", "O", "draw"})
    assert_equal(results["X"] + results["O"] + results["draw"], 10)


def test_seeded_neural_evaluation_is_repeatable():
    model_data = create_deterministic_model()

    first_results = evaluate_neural_model(
        model_data=model_data,
        games_count=20,
        seed=123,
    )
    second_results = evaluate_neural_model(
        model_data=model_data,
        games_count=20,
        seed=123,
    )

    assert_equal(first_results, second_results)


def test_summarize_neural_evaluation_results_computes_efficiency():
    results = {
        "X": 2,
        "O": 6,
        "draw": 2,
    }

    summary = summarize_neural_evaluation_results(
        results,
        trained_player="O",
        opponent_player="X",
    )

    assert_equal(summary["total_games"], 10)
    assert_equal(summary["trained_player"], "O")
    assert_equal(summary["opponent_player"], "X")
    assert_equal(summary["trained_player_wins"], 6)
    assert_equal(summary["opponent_player_wins"], 2)
    assert_equal(summary["draws"], 2)
    assert_equal(summary["efficiency"], 70.0)


def test_format_neural_evaluation_summary_contains_key_information():
    summary = {
        "total_games": 10,
        "trained_player": "O",
        "opponent_player": "X",
        "trained_player_wins": 6,
        "opponent_player_wins": 2,
        "draws": 2,
        "efficiency": 70.0,
    }

    text = format_neural_evaluation_summary(summary)

    assert_true("Résumé évaluation neuronale" in text)
    assert_true("Parties jouées : 10" in text)
    assert_true("Victoires de X : 2" in text)
    assert_true("Victoires de O : 6" in text)
    assert_true("Matchs nuls : 2" in text)
    assert_true("Score d'efficacité : 70.0 %" in text)


TESTS = [
    ("Une partie neuronale automatique retourne un résultat valide", test_play_neural_automatic_game_returns_valid_result),
    ("L'évaluation neuronale compte toutes les parties", test_evaluate_neural_model_counts_all_games),
    ("L'évaluation neuronale avec graine est reproductible", test_seeded_neural_evaluation_is_repeatable),
    ("Le résumé d'évaluation neuronale calcule l'efficacité", test_summarize_neural_evaluation_results_computes_efficiency),
    ("Le résumé textuel d'évaluation neuronale contient les informations clés", test_format_neural_evaluation_summary_contains_key_information),
]