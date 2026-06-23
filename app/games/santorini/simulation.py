from app.games.santorini.action_format import format_action
from app.games.santorini.agents import create_rng, choose_random_action, choose_random_placement
from app.games.santorini.coordinates import index_to_cell
from app.games.santorini.rules import apply_action, create_new_game, get_game_result, place_worker


def play_random_game(seed=None, max_turns=200, keep_log=False):
    rng = create_rng(seed)
    game = create_new_game()
    log = []
    turn_count = 0

    while game["phase"] == "placement":
        player = game["current_player"]
        cell = choose_random_placement(game, rng)

        if cell is None:
            break

        place_worker(game, cell)

        if keep_log:
            log.append(player + " place " + index_to_cell(cell))

    while game["phase"] == "play" and turn_count < max_turns:
        player = game["current_player"]
        action = choose_random_action(game, rng)

        if action is None:
            game["phase"] = "finished"
            game["winner"] = "O" if player == "X" else "X"
            break

        apply_action(game, action)
        turn_count += 1

        if keep_log:
            log.append(player + " " + format_action(action))

    return {
        "winner": get_game_result(game),
        "turns": turn_count,
        "finished": game["phase"] == "finished",
        "log": log,
        "game": game,
    }


def run_random_matches(count, seed=0):
    results = {"X": 0, "O": 0, "ongoing": 0}
    total_turns = 0

    for index in range(count):
        result = play_random_game(seed=seed + index)
        results[result["winner"]] += 1
        total_turns += result["turns"]

    average_turns = 0.0
    if count > 0:
        average_turns = total_turns / count

    return {
        "games": count,
        "wins_x": results["X"],
        "wins_o": results["O"],
        "unfinished": results["ongoing"],
        "average_turns": average_turns,
    }
