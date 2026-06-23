from app.games.santorini.action_format import format_action
from app.games.santorini.agents import choose_random_action
from app.games.santorini.encoding import encode_santorini_state, encode_santorini_state_key
from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.rules import apply_action, copy_game, get_game_result, switch_player


TRAINED_PLAYER = "O"


def create_santorini_move_score_example(game, simulations_per_move, rng):
    move_scores = []

    for action in get_indexed_legal_actions(game, TRAINED_PLAYER):
        score = evaluate_santorini_action(game, action, simulations_per_move, rng)
        move_scores.append(_format_score(action, score))

    best_move = choose_best_move_score(move_scores)

    return {
        "state_key": encode_santorini_state_key(game),
        "inputs": encode_santorini_state(game, TRAINED_PLAYER),
        "player": TRAINED_PLAYER,
        "legal_moves_count": len(move_scores),
        "move_scores": move_scores,
        "best_output_index": best_move["output_index"] if best_move else None,
        "best_action": best_move["action"] if best_move else None,
        "best_score": best_move["score"] if best_move else None,
    }


def evaluate_santorini_action(game, action, simulations_per_move, rng):
    if simulations_per_move <= 0:
        return 0.0

    total = 0.0

    for _ in range(simulations_per_move):
        simulation = copy_game(game)
        apply_action(simulation, action)
        result = get_game_result(simulation)

        if result == "ongoing":
            result = play_random_rollout(simulation, rng)

        total += score_result_for_o(result)

    return total / simulations_per_move


def play_random_rollout(game, rng, max_turns=160):
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        action = choose_random_action(game, rng)

        if action is None:
            current_player = game["current_player"]
            game["winner"] = switch_player(current_player)
            game["phase"] = "finished"
            break

        apply_action(game, action)
        turn_count += 1

    return get_game_result(game)


def score_result_for_o(result):
    if result == "O":
        return 1.0
    if result == "X":
        return 0.0
    return 0.5


def choose_best_move_score(move_scores):
    if not move_scores:
        return None

    return max(move_scores, key=lambda item: item["score"])


def _format_score(action, score):
    return {
        "output_index": action["output_index"],
        "action": format_action(action),
        "score": round(score, 4),
    }
