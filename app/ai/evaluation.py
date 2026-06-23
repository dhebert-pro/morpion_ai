from app.config import EVALUATION_GAMES_COUNT

from app.ai.strategies import (
    choose_random_move,
    choose_model_move,
    choose_fallback_move,
)

from app.games.morpion.adapter import MORPION_ADAPTER


def play_automatic_game(x_strategy, o_strategy, game_adapter=MORPION_ADAPTER):
    game = game_adapter.create_new_game()
    current_player = game_adapter.first_player

    strategies_by_player = {
        game_adapter.opponent_player: x_strategy,
        game_adapter.trained_player: o_strategy,
    }

    while game_adapter.get_game_result(game) == "ongoing":
        strategy = strategies_by_player[current_player]
        move = strategy(game)

        if not game_adapter.is_valid_move(game, move):
            raise ValueError("La stratégie a choisi un coup illégal : " + str(move))

        game_adapter.apply_move(game, move, current_player)
        current_player = game_adapter.switch_player(current_player)

    return game_adapter.get_game_result(game)


def evaluate_model(model, games_count=EVALUATION_GAMES_COUNT, game_adapter=MORPION_ADAPTER):
    results = {
        game_adapter.opponent_player: 0,
        game_adapter.trained_player: 0,
        "draw": 0,
    }

    def opponent_strategy(game):
        return choose_random_move(game, game_adapter)

    def trained_strategy(game):
        return choose_model_move(
            game,
            model,
            fallback_strategy=choose_fallback_move,
            game_adapter=game_adapter,
        )

    for _ in range(games_count):
        result = play_automatic_game(opponent_strategy, trained_strategy, game_adapter)
        results[result] += 1

    return results


def compute_player_efficiency(results, player):
    total = 0

    for result_count in results.values():
        total += result_count

    if total == 0:
        return 0.0

    score = results[player] + 0.5 * results["draw"]

    return score / total * 100


def compute_o_efficiency(results):
    return compute_player_efficiency(results, "O")


def print_evaluation_results(
    results,
    trained_player="O",
    opponent_player="X",
):
    total = 0

    for result_count in results.values():
        total += result_count

    efficiency = compute_player_efficiency(results, trained_player)

    print("Résultats sur", total, "parties :")
    print("Victoires de", opponent_player, ":", results[opponent_player])
    print("Victoires de", trained_player, ":", results[trained_player])
    print("Matchs nuls    :", results["draw"])
    print("Score d'efficacité de", trained_player, ":", round(efficiency, 2), "%")