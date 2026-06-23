import random

from app.games.santorini.neural_player import (
    TRAINED_PLAYER,
    create_random_started_santorini_game,
    choose_santorini_neural_action_from_network,
)
from app.games.santorini.reference_agents import choose_reference_action
from app.games.santorini.rules import apply_action, copy_game, get_game_result, switch_player

OPPONENT_PLAYER = "X"


def evaluate_santorini_neural_vs_references_paired(
    network,
    opponent_names,
    games_count,
    seed=0,
    max_turns=160,
):
    return [
        evaluate_santorini_neural_vs_reference_paired(
            network=network,
            opponent_name=name,
            games_count=games_count,
            seed=seed,
            max_turns=max_turns,
        )
        for name in opponent_names
    ]


def evaluate_santorini_neural_vs_reference_paired(
    network,
    opponent_name,
    games_count,
    seed=0,
    max_turns=160,
):
    starts = _create_starts(games_count, seed)
    neural_results = {"X": 0, "O": 0, "draw": 0}
    baseline_results = {"X": 0, "O": 0, "draw": 0}

    for index, start in enumerate(starts):
        neural_rng = random.Random(seed + 30_000 + index)
        baseline_rng = random.Random(seed + 40_000 + index)
        _add_result(
            neural_results,
            _play_started_game(start, network, opponent_name, neural_rng, max_turns),
        )
        _add_result(
            baseline_results,
            _play_started_game(start, None, opponent_name, baseline_rng, max_turns),
        )

    return {
        "opponent": opponent_name,
        "neural_results": neural_results,
        "baseline_results": baseline_results,
    }


def _create_starts(games_count, seed):
    rng = random.Random(seed + 2_000)
    return [create_random_started_santorini_game(rng) for _ in range(games_count)]


def _play_started_game(start_game, o_network, opponent_name, rng, max_turns):
    game = copy_game(start_game)
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        action = _choose_action(game, o_network, opponent_name, rng)

        if action is None:
            game["winner"] = switch_player(game["current_player"])
            game["phase"] = "finished"
            break

        apply_action(game, action)
        turn_count += 1

    result = get_game_result(game)
    return "draw" if result == "ongoing" else result


def _choose_action(game, o_network, opponent_name, rng):
    if game["current_player"] == TRAINED_PLAYER and o_network is not None:
        return choose_santorini_neural_action_from_network(game, o_network, rng)

    if game["current_player"] == OPPONENT_PLAYER:
        return choose_reference_action(game, opponent_name, rng)

    return choose_reference_action(game, "random", rng)


def _add_result(results, result):
    if result in results:
        results[result] += 1
    else:
        results["draw"] += 1
