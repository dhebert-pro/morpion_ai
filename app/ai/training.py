from app.config import (
    TRAINING_GAMES_COUNT,
    SIMULATIONS_PER_MOVE,
)

from app.utils.progress import print_progress

from app.ai.strategies import choose_random_move

from app.games.morpion.rules import (
    copy_game,
    encode_game_state,
    switch_player,
    get_legal_moves,
    get_game_result,
    get_score_for_o,
    create_new_game,
)


def simulate_random_game_from(game, player_to_move):
    simulated_game = copy_game(game)
    current_player = player_to_move

    while get_game_result(simulated_game) == "ongoing":
        move = choose_random_move(simulated_game)
        simulated_game["board"][move] = current_player
        current_player = switch_player(current_player)

    return get_game_result(simulated_game)


def evaluate_o_move(game, move, simulations_count):
    total_score = 0

    for _ in range(simulations_count):
        simulated_game = copy_game(game)
        simulated_game["board"][move] = "O"

        result = get_game_result(simulated_game)

        if result == "ongoing":
            result = simulate_random_game_from(simulated_game, "X")

        total_score += get_score_for_o(result)

    return total_score / simulations_count


def collect_training_states(training_games_count, show_progress=False):
    states_to_learn = {}

    for game_number in range(training_games_count):
        game = create_new_game()
        current_player = "X"

        while get_game_result(game) == "ongoing":
            if current_player == "O":
                state_key = encode_game_state(game)
                states_to_learn[state_key] = copy_game(game)

            move = choose_random_move(game)
            game["board"][move] = current_player
            current_player = switch_player(current_player)

        if show_progress:
            print_progress("Collecte des états     ", game_number + 1, training_games_count)

    return states_to_learn


def train_model(
    training_games_count=TRAINING_GAMES_COUNT,
    simulations_per_move=SIMULATIONS_PER_MOVE,
    show_progress=False
):
    states_to_learn = collect_training_states(training_games_count, show_progress)
    model = {}

    state_items = list(states_to_learn.items())
    total_states = len(state_items)

    if show_progress and total_states == 0:
        print("Aucun état à apprendre.")

    for state_number, item in enumerate(state_items, start=1):
        state_key = item[0]
        game = item[1]
        legal_moves = get_legal_moves(game)

        best_move = None
        best_score = -1

        for move in legal_moves:
            score = evaluate_o_move(game, move, simulations_per_move)

            if score > best_score:
                best_score = score
                best_move = move

        model[state_key] = best_move

        if show_progress:
            print_progress("Apprentissage du modèle", state_number, total_states)

    return model