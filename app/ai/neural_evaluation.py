import random

from app.ai.evaluation import compute_player_efficiency
from app.ai.neural_strategy import choose_neural_move
from app.ai.strategies import choose_random_move

from app.games.morpion.adapter import MORPION_ADAPTER


def play_neural_automatic_game(
    model_data,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    """Joue une partie automatique.

    L'adversaire joue des coups aléatoires.
    Le joueur entraîné joue avec le modèle neuronal.

    Cette fonction ne sauvegarde rien et n'affiche rien.
    Elle sert uniquement à mesurer la qualité du modèle.
    """

    game = game_adapter.create_new_game()
    current_player = game_adapter.first_player

    while game_adapter.get_game_result(game) == "ongoing":
        if current_player == game_adapter.trained_player:
            move = choose_neural_move(
                game=game,
                model_data=model_data,
                game_adapter=game_adapter,
                fallback_strategy=lambda current_game: choose_random_move(
                    current_game,
                    game_adapter,
                    rng=rng,
                ),
            )
        else:
            move = choose_random_move(
                game,
                game_adapter,
                rng=rng,
            )

        if not game_adapter.is_valid_move(game, move):
            raise ValueError("Coup illégal choisi : " + str(move))

        game_adapter.apply_move(
            game,
            move,
            current_player,
        )
        current_player = game_adapter.switch_player(current_player)

    return game_adapter.get_game_result(game)


def evaluate_neural_model(
    model_data,
    games_count,
    game_adapter=MORPION_ADAPTER,
    seed=None,
):
    """Évalue un modèle neuronal contre un adversaire aléatoire."""

    results = {
        game_adapter.opponent_player: 0,
        game_adapter.trained_player: 0,
        "draw": 0,
    }

    rng = None

    if seed is not None:
        rng = random.Random(seed)

    for _ in range(games_count):
        result = play_neural_automatic_game(
            model_data,
            game_adapter,
            rng=rng,
        )
        results[result] += 1

    return results


def summarize_neural_evaluation_results(
    results,
    trained_player="O",
    opponent_player="X",
):
    total_games = 0

    for result_count in results.values():
        total_games += result_count

    efficiency = compute_player_efficiency(
        results,
        trained_player,
    )

    return {
        "total_games": total_games,
        "trained_player": trained_player,
        "opponent_player": opponent_player,
        "trained_player_wins": results[trained_player],
        "opponent_player_wins": results[opponent_player],
        "draws": results["draw"],
        "efficiency": efficiency,
    }


def format_neural_evaluation_summary(summary):
    lines = []

    lines.append("Résumé évaluation neuronale")
    lines.append("Parties jouées : " + str(summary["total_games"]))
    lines.append(
        "Victoires de "
        + str(summary["opponent_player"])
        + " : "
        + str(summary["opponent_player_wins"])
    )
    lines.append(
        "Victoires de "
        + str(summary["trained_player"])
        + " : "
        + str(summary["trained_player_wins"])
    )
    lines.append("Matchs nuls : " + str(summary["draws"]))
    lines.append("Score d'efficacité : " + str(round(summary["efficiency"], 2)) + " %")

    return "\n".join(lines)
