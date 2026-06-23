from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_strategy import (
    predict_legal_move_scores,
    choose_best_predicted_move,
    choose_neural_move,
)

from app.games.morpion.rules import create_new_game


def create_game_with_board(board):
    game = create_new_game()
    game["board"] = board.copy()
    return game


def create_deterministic_model(output_biases):
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

    network.output_biases = output_biases.copy()

    return network.to_dict()


def test_predict_legal_move_scores_returns_only_legal_moves():
    game = create_game_with_board([
        "O", None, None,
        "X", None, None,
        None, None, "X",
    ])

    model_data = create_deterministic_model([
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
    ])

    scores = predict_legal_move_scores(game, model_data)

    assert_equal(set(scores.keys()), {1, 2, 4, 5, 6, 7})

    for score in scores.values():
        assert_true(0.0 <= score <= 1.0)


def test_choose_best_predicted_move_returns_highest_score():
    move_scores = {
        1: 0.25,
        2: 0.75,
        5: 0.50,
    }

    best_move = choose_best_predicted_move(move_scores)

    assert_equal(best_move, 2)


def test_choose_best_predicted_move_returns_none_without_scores():
    best_move = choose_best_predicted_move({})

    assert_equal(best_move, None)


def test_choose_neural_move_ignores_illegal_highest_prediction():
    game = create_game_with_board([
        None, None, None,
        None, "X", None,
        None, None, "O",
    ])

    model_data = create_deterministic_model([
        0.0, 0.1, 0.2,
        0.3, 5.0, 0.5,
        2.0, 0.7, 4.0,
    ])

    move = choose_neural_move(game, model_data)

    assert_equal(move, 6)


def test_choose_neural_move_returns_none_on_finished_board():
    game = create_game_with_board([
        "O", "X", "O",
        "O", "X", "X",
        "X", "O", "O",
    ])

    model_data = create_deterministic_model([
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
    ])

    move = choose_neural_move(game, model_data)

    assert_equal(move, None)


def test_choose_neural_move_uses_fallback_without_model():
    game = create_game_with_board([
        "O", None, None,
        None, "X", None,
        None, None, None,
    ])

    def fallback_strategy(current_game):
        return 1

    move = choose_neural_move(
        game=game,
        model_data={},
        fallback_strategy=fallback_strategy,
    )

    assert_equal(move, 1)


def test_choose_neural_move_is_pure_by_default_on_immediate_win():
    game = create_game_with_board([
        "X", None, None,
        "O", "O", None,
        "X", None, None,
    ])

    model_data = create_deterministic_model([
        0.0, 5.0, 4.0,
        0.0, 0.0, 0.1,
        0.0, 3.0, 2.0,
    ])

    move = choose_neural_move(game, model_data)

    assert_equal(move, 1)


def test_choose_neural_move_can_use_guard_as_explicit_diagnostic_mode():
    game = create_game_with_board([
        "X", None, None,
        "O", "O", None,
        "X", None, None,
    ])

    model_data = create_deterministic_model([
        0.0, 5.0, 4.0,
        0.0, 0.0, 0.1,
        0.0, 3.0, 2.0,
    ])

    move = choose_neural_move(
        game,
        model_data,
        use_tactical_guard=True,
    )

    assert_equal(move, 5)


def test_choose_neural_move_is_pure_by_default_on_immediate_loss():
    game = create_game_with_board([
        "O", None, None,
        "X", "X", None,
        "O", None, None,
    ])

    model_data = create_deterministic_model([
        0.0, 5.0, 4.0,
        0.0, 0.0, 0.1,
        0.0, 3.0, 2.0,
    ])

    move = choose_neural_move(game, model_data)

    assert_equal(move, 1)


TESTS = [
    ("La stratégie neuronale prédit seulement les coups légaux", test_predict_legal_move_scores_returns_only_legal_moves),
    ("La stratégie choisit le score prédit le plus élevé", test_choose_best_predicted_move_returns_highest_score),
    ("La stratégie retourne None sans score disponible", test_choose_best_predicted_move_returns_none_without_scores),
    ("La stratégie neuronale ignore les prédictions illégales", test_choose_neural_move_ignores_illegal_highest_prediction),
    ("La stratégie neuronale retourne None sur plateau terminé", test_choose_neural_move_returns_none_on_finished_board),
    ("La stratégie neuronale utilise un fallback sans modèle", test_choose_neural_move_uses_fallback_without_model),
    ("La stratégie neuronale pure ne force pas une victoire immédiate", test_choose_neural_move_is_pure_by_default_on_immediate_win),
    ("La garde morpion reste disponible en mode explicite", test_choose_neural_move_can_use_guard_as_explicit_diagnostic_mode),
    ("La stratégie neuronale pure ne force pas un blocage immédiat", test_choose_neural_move_is_pure_by_default_on_immediate_loss),
]
