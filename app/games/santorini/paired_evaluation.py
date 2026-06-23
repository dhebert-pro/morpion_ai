import random

from app.games.santorini.agents import choose_random_action
from app.games.santorini.neural_player import (
    create_random_started_santorini_game,
    choose_santorini_neural_action_from_network,
)
from app.games.santorini.rules import (
    apply_action,
    copy_game,
    get_game_result,
    switch_player,
)

TRAINED_PLAYER = "O"
OPPONENT_PLAYER = "X"


def evaluate_santorini_neural_vs_random_paired(network, games_count, seed=0, max_turns=160):
    starts = create_santorini_starting_games(games_count, seed=seed)
    neural_results = {"X": 0, "O": 0, "draw": 0}
    baseline_results = {"X": 0, "O": 0, "draw": 0}

    for index, start in enumerate(starts):
        neural_rng = random.Random(seed + 10_000 + index)
        baseline_rng = random.Random(seed + 20_000 + index)
        _add_result(
            neural_results,
            play_started_santorini_game(start, network, neural_rng, max_turns=max_turns),
        )
        _add_result(
            baseline_results,
            play_started_santorini_game(start, None, baseline_rng, max_turns=max_turns),
        )

    return {
        "neural_results": neural_results,
        "baseline_results": baseline_results,
    }


def create_santorini_starting_games(games_count, seed=0):
    rng = random.Random(seed + 1_000)
    return [create_random_started_santorini_game(rng) for _ in range(games_count)]


def play_started_santorini_game(start_game, o_network, rng, max_turns=160):
    game = copy_game(start_game)
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        action = _choose_action(game, o_network, rng)

        if action is None:
            game["winner"] = switch_player(game["current_player"])
            game["phase"] = "finished"
            break

        apply_action(game, action)
        turn_count += 1

    result = get_game_result(game)
    return "draw" if result == "ongoing" else result


def _choose_action(game, o_network, rng):
    if game["current_player"] == TRAINED_PLAYER and o_network is not None:
        return choose_santorini_neural_action_from_network(game, o_network, rng)

    return choose_random_action(game, rng)


def _add_result(results, result):
    if result in results:
        results[result] += 1
    else:
        results["draw"] += 1
