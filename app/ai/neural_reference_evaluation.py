import random

from app.ai.evaluation import compute_player_efficiency
from app.ai.neural_strategy import choose_neural_move
from app.ai.reference_opponents import (
    REFERENCE_RANDOM,
    choose_reference_opponent_move,
    get_default_reference_opponents,
)
from app.games.morpion.adapter import MORPION_ADAPTER


def play_neural_against_reference_game(
    model_data,
    reference_name=REFERENCE_RANDOM,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    game = game_adapter.create_new_game()
    current_player = game_adapter.first_player

    while game_adapter.get_game_result(game) == "ongoing":
        if current_player == game_adapter.trained_player:
            move = choose_neural_move(
                game=game,
                model_data=model_data,
                game_adapter=game_adapter,
            )
        else:
            move = choose_reference_opponent_move(
                game=game,
                player=game_adapter.opponent_player,
                opponent=game_adapter.trained_player,
                reference_name=reference_name,
                game_adapter=game_adapter,
                rng=rng,
            )

        if not game_adapter.is_valid_move(game, move):
            raise ValueError("Coup illégal choisi : " + str(move))

        game_adapter.apply_move(game, move, current_player)
        current_player = game_adapter.switch_player(current_player)

    return game_adapter.get_game_result(game)


def evaluate_neural_model_against_reference(
    model_data,
    reference_name,
    games_count,
    game_adapter=MORPION_ADAPTER,
    seed=None,
):
    results = {
        game_adapter.opponent_player: 0,
        game_adapter.trained_player: 0,
        "draw": 0,
    }
    rng = None

    if seed is not None:
        rng = random.Random(seed)

    for _ in range(games_count):
        result = play_neural_against_reference_game(
            model_data=model_data,
            reference_name=reference_name,
            game_adapter=game_adapter,
            rng=rng,
        )
        results[result] += 1

    return summarize_reference_evaluation_results(
        results=results,
        reference_name=reference_name,
        game_adapter=game_adapter,
    )


def evaluate_neural_model_against_references(
    model_data,
    games_count,
    game_adapter=MORPION_ADAPTER,
    seed=None,
    reference_names=None,
):
    if reference_names is None:
        reference_names = get_default_reference_opponents()

    evaluations = []

    for index, reference_name in enumerate(reference_names):
        reference_seed = None

        if seed is not None:
            reference_seed = seed + index

        evaluations.append(
            evaluate_neural_model_against_reference(
                model_data=model_data,
                reference_name=reference_name,
                games_count=games_count,
                game_adapter=game_adapter,
                seed=reference_seed,
            )
        )

    return evaluations


def summarize_reference_evaluation_results(
    results,
    reference_name,
    game_adapter=MORPION_ADAPTER,
):
    total_games = sum(results.values())
    efficiency = compute_player_efficiency(
        results,
        game_adapter.trained_player,
    )

    return {
        "reference_name": reference_name,
        "total_games": total_games,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "trained_player_wins": results[game_adapter.trained_player],
        "opponent_player_wins": results[game_adapter.opponent_player],
        "draws": results["draw"],
        "efficiency": efficiency,
    }


def format_neural_reference_evaluation_report(evaluations):
    lines = []
    lines.append("Évaluation contre adversaires de référence")
    lines.append("Objectif : mesurer la robustesse sans adversaire parfait spécifique au jeu.")

    for evaluation in evaluations:
        lines.append("")
        lines.append("Adversaire : " + str(evaluation["reference_name"]))
        lines.append("Parties jouées : " + str(evaluation["total_games"]))
        lines.append(
            "Victoires de "
            + str(evaluation["opponent_player"])
            + " : "
            + str(evaluation["opponent_player_wins"])
        )
        lines.append(
            "Victoires de "
            + str(evaluation["trained_player"])
            + " : "
            + str(evaluation["trained_player_wins"])
        )
        lines.append("Matchs nuls : " + str(evaluation["draws"]))
        lines.append(
            "Score d'efficacité : "
            + str(round(evaluation["efficiency"], 2))
            + " %"
        )

    return "\n".join(lines)
