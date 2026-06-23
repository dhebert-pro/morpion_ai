from tests.test_helpers import assert_equal

from app.games.morpion.rules import create_new_game, get_legal_moves

from app.ai.training import (
    simulate_random_game_from,
    evaluate_o_move,
    collect_training_states,
    train_model,
)


def test_simulate_random_game_from_returns_final_result():
    game = create_new_game()
    result = simulate_random_game_from(game, "X")

    assert result in ["X", "O", "draw"], f"Résultat inattendu : {result}"


def test_evaluate_o_move_returns_score_between_0_and_1():
    game = create_new_game()
    game["board"][4] = "X"

    score = evaluate_o_move(game, 0, simulations_count=5)

    assert score >= 0.0, f"Score trop petit : {score}"
    assert score <= 1.0, f"Score trop grand : {score}"


def test_collect_training_states_contains_only_o_turn_states():
    states = collect_training_states(training_games_count=20)

    for state_key in states:
        x_count = state_key.count("X")
        o_count = state_key.count("O")

        assert_equal(x_count, o_count + 1)


def test_train_model_returns_legal_moves():
    model = train_model(training_games_count=20, simulations_per_move=5)

    for state_key in model:
        game = create_new_game()

        for index in range(9):
            symbol = state_key[index]

            if symbol == ".":
                game["board"][index] = None
            else:
                game["board"][index] = symbol

        move = model[state_key]

        assert move in get_legal_moves(game), f"Coup illégal dans le modèle : {move} pour {state_key}"


TESTS = [
    ("Simuler une partie aléatoire jusqu'à la fin", test_simulate_random_game_from_returns_final_result),
    ("Évaluer un coup de O donne un score entre 0 et 1", test_evaluate_o_move_returns_score_between_0_and_1),
    ("Collecter uniquement des états où O doit jouer", test_collect_training_states_contains_only_o_turn_states),
    ("Le modèle entraîné ne contient que des coups légaux", test_train_model_returns_legal_moves),
]
