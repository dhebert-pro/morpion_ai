from app.config import (
    TRAINING_GAMES_COUNT,
    SIMULATIONS_PER_MOVE,
)

from app.utils.progress import print_progress

from app.ai.strategies import choose_random_move

from app.games.morpion.adapter import MORPION_ADAPTER


def simulate_random_game_from(game, player_to_move, game_adapter=MORPION_ADAPTER, rng=None):
    simulated_game = game_adapter.copy_game(game)
    current_player = player_to_move

    while game_adapter.get_game_result(simulated_game) == "ongoing":
        move = choose_random_move(simulated_game, game_adapter, rng=rng)
        game_adapter.apply_move(simulated_game, move, current_player)
        current_player = game_adapter.switch_player(current_player)

    return game_adapter.get_game_result(simulated_game)


def evaluate_player_move(
    game,
    move,
    player,
    simulations_count,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    total_score = 0

    for _ in range(simulations_count):
        simulated_game = game_adapter.copy_game(game)
        game_adapter.apply_move(simulated_game, move, player)

        result = game_adapter.get_game_result(simulated_game)

        if result == "ongoing":
            next_player = game_adapter.switch_player(player)
            result = simulate_random_game_from(
                simulated_game,
                next_player,
                game_adapter,
                rng=rng,
            )

        total_score += game_adapter.get_score_for_trained_player(result)

    return total_score / simulations_count


def evaluate_o_move(game, move, simulations_count, game_adapter=MORPION_ADAPTER, rng=None):
    return evaluate_player_move(
        game,
        move,
        game_adapter.trained_player,
        simulations_count,
        game_adapter,
        rng=rng,
    )


def collect_training_states(
    training_games_count,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    states_to_learn = {}

    for game_number in range(training_games_count):
        game = game_adapter.create_new_game()
        current_player = game_adapter.first_player

        while game_adapter.get_game_result(game) == "ongoing":
            if current_player == game_adapter.trained_player:
                state_key = game_adapter.encode_game_state(game)
                states_to_learn[state_key] = game_adapter.copy_game(game)

            move = choose_random_move(game, game_adapter, rng=rng)
            game_adapter.apply_move(game, move, current_player)
            current_player = game_adapter.switch_player(current_player)

        if show_progress:
            print_progress("Collecte des états     ", game_number + 1, training_games_count)

    return states_to_learn


def train_model(
    training_games_count=TRAINING_GAMES_COUNT,
    simulations_per_move=SIMULATIONS_PER_MOVE,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    states_to_learn = collect_training_states(
        training_games_count,
        show_progress,
        game_adapter,
        rng=rng,
    )
    model = {}

    state_items = list(states_to_learn.items())
    total_states = len(state_items)

    if show_progress and total_states == 0:
        print("Aucun état à apprendre.")

    for state_number, item in enumerate(state_items, start=1):
        state_key = item[0]
        game = item[1]
        legal_moves = game_adapter.get_legal_moves(game)

        best_move = None
        best_score = -1

        for move in legal_moves:
            score = evaluate_player_move(
                game,
                move,
                game_adapter.trained_player,
                simulations_per_move,
                game_adapter,
                rng=rng,
            )

            if score > best_score:
                best_score = score
                best_move = move

        model[state_key] = best_move

        if show_progress:
            print_progress("Apprentissage du modèle", state_number, total_states)

    return model
