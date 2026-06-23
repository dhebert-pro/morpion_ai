from tests.test_helpers import assert_equal

from app.ai.game_adapter import GameAdapter

from app.ai.training import (
    collect_training_states,
    train_model,
)

from app.ai.evaluation import (
    evaluate_model,
    compute_player_efficiency,
)


def create_tiny_game_adapter():
    def create_new_game():
        return {"moves": []}

    def copy_game(game):
        return {"moves": game["moves"].copy()}

    def encode_game_state(game):
        parts = []

        for player, move in game["moves"]:
            parts.append(player + str(move))

        return ";".join(parts)

    def get_legal_moves(game):
        if len(game["moves"]) >= 2:
            return []

        return [0, 1]

    def is_valid_move(game, move):
        return move in get_legal_moves(game)

    def apply_move(game, move, player):
        game["moves"].append((player, move))

    def switch_player(player):
        if player == "A":
            return "B"

        return "A"

    def get_game_result(game):
        if len(game["moves"]) < 2:
            return "ongoing"

        player, move = game["moves"][1]

        if player == "B" and move == 1:
            return "B"

        return "A"

    def get_winner(game):
        result = get_game_result(game)

        if result == "ongoing":
            return None

        return result

    def get_score_for_trained_player(result):
        if result == "B":
            return 1.0

        if result == "draw":
            return 0.5

        return 0.0

    def move_to_index(move):
        return move

    def index_to_move(index):
        return index

    return GameAdapter(
        name="tiny_game",
        first_player="A",
        trained_player="B",
        opponent_player="A",
        create_new_game=create_new_game,
        copy_game=copy_game,
        encode_game_state=encode_game_state,
        get_legal_moves=get_legal_moves,
        is_valid_move=is_valid_move,
        apply_move=apply_move,
        switch_player=switch_player,
        get_game_result=get_game_result,
        get_winner=get_winner,
        get_score_for_trained_player=get_score_for_trained_player,
        output_size=2,
        move_to_index=move_to_index,
        index_to_move=index_to_move,
    )


def test_training_collects_states_with_custom_adapter():
    adapter = create_tiny_game_adapter()
    states = collect_training_states(10, game_adapter=adapter)

    assert len(states) > 0, "L'adaptateur devrait permettre de collecter des états."

    for game in states.values():
        assert_equal(len(game["moves"]), 1)


def test_training_learns_with_custom_adapter():
    adapter = create_tiny_game_adapter()
    model = train_model(
        training_games_count=20,
        simulations_per_move=2,
        game_adapter=adapter,
    )

    assert len(model) > 0, "Le modèle devrait apprendre au moins un état."

    for move in model.values():
        assert_equal(move, 1)


def test_evaluation_uses_custom_adapter_player_names():
    adapter = create_tiny_game_adapter()
    results = evaluate_model({}, games_count=5, game_adapter=adapter)

    assert_equal(set(results.keys()), {"A", "B", "draw"})
    assert_equal(results["A"] + results["B"] + results["draw"], 5)


def test_compute_player_efficiency_is_not_limited_to_o():
    results = {
        "A": 1,
        "B": 2,
        "draw": 1,
    }

    assert_equal(compute_player_efficiency(results, "B"), 62.5)


TESTS = [
    ("L'entraînement collecte des états avec un adaptateur non morpion", test_training_collects_states_with_custom_adapter),
    ("L'entraînement apprend avec un adaptateur non morpion", test_training_learns_with_custom_adapter),
    ("L'évaluation utilise les noms de joueurs de l'adaptateur", test_evaluation_uses_custom_adapter_player_names),
    ("Le score d'efficacité n'est pas limité au joueur O", test_compute_player_efficiency_is_not_limited_to_o),
]