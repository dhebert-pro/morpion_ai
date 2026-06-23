import random

from app.games.santorini.agents import choose_random_action
from app.games.santorini.neural_player import create_random_started_santorini_game
from app.games.santorini.rules import apply_action, get_game_result, switch_player


def evaluate_santorini_random_o_vs_random(games_count, seed=0, max_turns=160):
    rng = random.Random(seed)
    results = {"X": 0, "O": 0, "draw": 0}

    for _ in range(games_count):
        result = play_santorini_random_o_vs_random(rng, max_turns=max_turns)

        if result in results:
            results[result] += 1
        else:
            results["draw"] += 1

    return results


def play_santorini_random_o_vs_random(rng, max_turns=160):
    game = create_random_started_santorini_game(rng)
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        action = choose_random_action(game, rng)

        if action is None:
            game["winner"] = switch_player(game["current_player"])
            game["phase"] = "finished"
            break

        apply_action(game, action)
        turn_count += 1

    return get_game_result(game)
