from app.ai.neural_network import SimpleNeuralNetwork
from app.games.santorini.agents import choose_random_action, choose_random_placement
from app.games.santorini.encoding import encode_santorini_state
from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.evaluation_summary import summarize_o_results, format_o_evaluation_summary
from app.games.santorini.rules import (
    apply_action,
    create_new_game,
    get_game_result,
    place_worker,
    switch_player,
)

TRAINED_PLAYER = "O"
OPPONENT_PLAYER = "X"


def choose_santorini_neural_action(game, model_data, rng=None, use_tactical_guard=False):
    if not model_data:
        return choose_random_action(game, rng)

    network = SimpleNeuralNetwork.from_dict(model_data)
    return choose_santorini_neural_action_from_network(
        game,
        network,
        rng,
        use_tactical_guard=use_tactical_guard,
    )


def choose_santorini_neural_action_from_network(
    game,
    network,
    rng=None,
    use_tactical_guard=False,
):
    legal_actions = get_indexed_legal_actions(game, TRAINED_PLAYER)

    if not legal_actions:
        return None

    if network is None:
        return choose_random_action(game, rng)

    inputs = encode_santorini_state(game, TRAINED_PLAYER)
    predictions = network.predict(inputs)

    best_action = None
    best_score = None

    candidate_actions = legal_actions

    if use_tactical_guard:
        from app.games.santorini.tactical_guard import filter_santorini_tactical_actions

        candidate_actions = filter_santorini_tactical_actions(game, legal_actions)

    for action in candidate_actions:
        score = predictions[action["output_index"]]

        if best_score is None or score > best_score:
            best_score = score
            best_action = action

    return best_action


def create_random_started_santorini_game(rng):
    game = create_new_game()

    while game["phase"] == "placement":
        placement = choose_random_placement(game, rng)

        if placement is None:
            break

        place_worker(game, placement)

    return game


def play_santorini_neural_vs_random(model_data, rng, max_turns=160):
    network = SimpleNeuralNetwork.from_dict(model_data) if model_data else None
    return play_santorini_neural_network_vs_random(network, rng, max_turns)


def play_santorini_neural_network_vs_random(network, rng, max_turns=160):
    game = create_random_started_santorini_game(rng)
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        if game["current_player"] == TRAINED_PLAYER:
            action = choose_santorini_neural_action_from_network(game, network, rng)
        else:
            action = choose_random_action(game, rng)

        if action is None:
            game["winner"] = switch_player(game["current_player"])
            game["phase"] = "finished"
            break

        apply_action(game, action)
        turn_count += 1

    return get_game_result(game)


def evaluate_santorini_neural_vs_random(model_data, games_count, seed=0):
    if not model_data:
        return {"X": 0, "O": 0, "draw": games_count}

    network = SimpleNeuralNetwork.from_dict(model_data)
    return evaluate_santorini_neural_network_vs_random(network, games_count, seed)


def evaluate_santorini_neural_network_vs_random(network, games_count, seed=0):
    import random

    rng = random.Random(seed)
    results = {"X": 0, "O": 0, "draw": 0}

    for _ in range(games_count):
        result = play_santorini_neural_network_vs_random(network, rng)

        if result in results:
            results[result] += 1
        else:
            results["draw"] += 1

    return results


def summarize_santorini_evaluation(results):
    return summarize_o_results(results)


def format_santorini_evaluation_summary(summary):
    return format_o_evaluation_summary(
        "Résumé évaluation neuronale Santorini",
        summary,
    )
