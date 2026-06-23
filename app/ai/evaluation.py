from app.config import EVALUATION_GAMES_COUNT

from app.ai.strategies import (
    choose_random_move,
    choose_model_move,
)

from app.games.morpion.rules import (
    create_new_game,
    switch_player,
    is_valid_move,
    get_game_result,
)


def play_automatic_game(x_strategy, o_strategy):
    game = create_new_game()
    current_player = "X"

    while get_game_result(game) == "ongoing":
        if current_player == "X":
            move = x_strategy(game)
        else:
            move = o_strategy(game)

        if not is_valid_move(game, move):
            raise ValueError("La stratégie a choisi un coup illégal : " + str(move))

        game["board"][move] = current_player
        current_player = switch_player(current_player)

    return get_game_result(game)


def evaluate_model(model, games_count=EVALUATION_GAMES_COUNT):
    results = {
        "X": 0,
        "O": 0,
        "draw": 0
    }

    def o_strategy(game):
        return choose_model_move(game, model)

    for _ in range(games_count):
        result = play_automatic_game(choose_random_move, o_strategy)
        results[result] += 1

    return results


def compute_o_efficiency(results):
    total = results["X"] + results["O"] + results["draw"]

    if total == 0:
        return 0.0

    score = results["O"] + 0.5 * results["draw"]

    return score / total * 100


def print_evaluation_results(results):
    total = results["X"] + results["O"] + results["draw"]
    efficiency = compute_o_efficiency(results)

    print("Résultats sur", total, "parties :")
    print("Victoires de X :", results["X"])
    print("Victoires de O :", results["O"])
    print("Matchs nuls    :", results["draw"])
    print("Score d'efficacité de O :", round(efficiency, 2), "%")