from tests.test_helpers import assert_equal

from app.ai.strategies import choose_random_move

from app.ai.evaluation import (
    play_automatic_game,
    evaluate_model,
    compute_o_efficiency,
)


def test_play_automatic_game_returns_final_result():
    result = play_automatic_game(choose_random_move, choose_random_move)

    assert result in ["X", "O", "draw"], f"Résultat inattendu : {result}"


def test_play_automatic_game_rejects_illegal_strategy():
    def illegal_strategy(game):
        return 99

    try:
        play_automatic_game(illegal_strategy, choose_random_move)
        raise AssertionError("La partie aurait dû refuser le coup illégal.")
    except ValueError:
        pass


def test_evaluate_model_returns_expected_keys():
    results = evaluate_model({}, games_count=10)

    assert_equal(set(results.keys()), {"X", "O", "draw"})


def test_evaluate_model_counts_requested_games():
    results = evaluate_model({}, games_count=10)
    total = results["X"] + results["O"] + results["draw"]

    assert_equal(total, 10)


def test_compute_o_efficiency_full_wins():
    results = {
        "X": 0,
        "O": 10,
        "draw": 0
    }

    assert_equal(compute_o_efficiency(results), 100.0)


def test_compute_o_efficiency_full_losses():
    results = {
        "X": 10,
        "O": 0,
        "draw": 0
    }

    assert_equal(compute_o_efficiency(results), 0.0)


def test_compute_o_efficiency_with_draws():
    results = {
        "X": 0,
        "O": 1,
        "draw": 1
    }

    assert_equal(compute_o_efficiency(results), 75.0)


TESTS = [
    ("Une partie automatique renvoie un résultat final", test_play_automatic_game_returns_final_result),
    ("Une stratégie automatique illégale est refusée", test_play_automatic_game_rejects_illegal_strategy),
    ("L'évaluation du modèle renvoie les bonnes clés", test_evaluate_model_returns_expected_keys),
    ("L'évaluation compte le bon nombre de parties", test_evaluate_model_counts_requested_games),
    ("Score d'efficacité avec uniquement des victoires", test_compute_o_efficiency_full_wins),
    ("Score d'efficacité avec uniquement des défaites", test_compute_o_efficiency_full_losses),
    ("Score d'efficacité avec des matchs nuls", test_compute_o_efficiency_with_draws),
]
