import random

from app.ai.move_scoring import choose_best_scored_move
from app.ai.reference_opponents import choose_reference_opponent_move
from app.ai.strategies import choose_random_move
from app.ai.training_dataset import select_balanced_state_items
from app.games.morpion.adapter import MORPION_ADAPTER
from app.utils.progress import print_progress


REFERENCE_EXAMPLE_SOURCE_PREFIX = "reference_"

def build_reference_move_score_dataset(
    training_games_count,
    simulations_per_move,
    max_examples=None,
    reference_names=None,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
    seed=0,
):
    if reference_names is None:
        reference_names = []

    if training_games_count <= 0 or len(reference_names) == 0:
        return _empty_reference_dataset(game_adapter, reference_names)

    examples = []
    available_states_count = 0
    total_slots = _get_total_slots(max_examples, reference_names)

    for name_index, reference_name in enumerate(reference_names):
        reference_seed = seed + name_index * 1000
        states = collect_reference_training_states(
            training_games_count=training_games_count,
            reference_name=reference_name,
            game_adapter=game_adapter,
            rng=random.Random(reference_seed),
        )
        available_states_count += len(states)
        selected_items = select_balanced_state_items(
            list(states.items()),
            max_examples=total_slots,
            seed=reference_seed + 1,
        )
        scoring_rng = random.Random(reference_seed + 2)
        total_states = len(selected_items)

        for index, item in enumerate(selected_items, start=1):
            example = create_reference_move_score_example(
                game=item[1],
                simulations_per_move=simulations_per_move,
                reference_name=reference_name,
                game_adapter=game_adapter,
                rng=scoring_rng,
            )
            examples.append(example)

            if show_progress:
                label = "Dataset réf. " + str(reference_name)
                print_progress(label[:22].ljust(22), index, total_states)

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "source": "reference_training",
        "reference_names": list(reference_names),
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "available_states_count": available_states_count,
        "reference_examples_count": len(examples),
        "examples_count": len(examples),
        "examples": examples,
    }

def collect_reference_training_states(
    training_games_count,
    reference_name,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    states_to_learn = {}

    for _ in range(training_games_count):
        game = game_adapter.create_new_game()
        current_player = game_adapter.first_player

        while game_adapter.get_game_result(game) == "ongoing":
            if current_player == game_adapter.trained_player:
                state_key = game_adapter.encode_game_state(game)
                states_to_learn[state_key] = game_adapter.copy_game(game)
                move = choose_random_move(game, game_adapter, rng=rng)
            else:
                move = choose_reference_opponent_move(
                    game=game,
                    player=game_adapter.opponent_player,
                    opponent=game_adapter.trained_player,
                    reference_name=reference_name,
                    game_adapter=game_adapter,
                    rng=rng,
                )

            game_adapter.apply_move(game, move, current_player)
            current_player = game_adapter.switch_player(current_player)

    return states_to_learn

def create_reference_move_score_example(
    game,
    simulations_per_move,
    reference_name,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    move_scores = score_legal_moves_against_reference(
        game=game,
        simulations_per_move=simulations_per_move,
        reference_name=reference_name,
        game_adapter=game_adapter,
        rng=rng,
    )
    formatted_scores = []

    for move, score in move_scores.items():
        formatted_scores.append({
            "move": move,
            "score": score,
        })

    return {
        "state_key": game_adapter.encode_game_state(game),
        "player": game_adapter.trained_player,
        "move_scores": formatted_scores,
        "best_move": choose_best_scored_move(move_scores),
        "source": REFERENCE_EXAMPLE_SOURCE_PREFIX + str(reference_name),
        "reference_name": reference_name,
    }

def score_legal_moves_against_reference(
    game,
    simulations_per_move,
    reference_name,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    move_scores = {}

    for move in game_adapter.get_legal_moves(game):
        move_scores[move] = evaluate_move_against_reference(
            game=game,
            move=move,
            simulations_count=simulations_per_move,
            reference_name=reference_name,
            game_adapter=game_adapter,
            rng=rng,
        )

    return move_scores

def evaluate_move_against_reference(
    game,
    move,
    simulations_count,
    reference_name,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    total_score = 0.0

    for _ in range(simulations_count):
        simulated_game = game_adapter.copy_game(game)
        game_adapter.apply_move(
            simulated_game,
            move,
            game_adapter.trained_player,
        )
        result = game_adapter.get_game_result(simulated_game)

        if result == "ongoing":
            result = simulate_reference_rollout_from(
                game=simulated_game,
                current_player=game_adapter.opponent_player,
                reference_name=reference_name,
                game_adapter=game_adapter,
                rng=rng,
            )

        total_score += game_adapter.get_score_for_trained_player(result)

    return total_score / simulations_count

def simulate_reference_rollout_from(
    game,
    current_player,
    reference_name,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    simulated_game = game_adapter.copy_game(game)

    while game_adapter.get_game_result(simulated_game) == "ongoing":
        if current_player == game_adapter.opponent_player:
            move = choose_reference_opponent_move(
                game=simulated_game,
                player=game_adapter.opponent_player,
                opponent=game_adapter.trained_player,
                reference_name=reference_name,
                game_adapter=game_adapter,
                rng=rng,
            )
        else:
            move = choose_reference_opponent_move(
                game=simulated_game,
                player=game_adapter.trained_player,
                opponent=game_adapter.opponent_player,
                reference_name="tactical",
                game_adapter=game_adapter,
                rng=rng,
            )

        game_adapter.apply_move(simulated_game, move, current_player)
        current_player = game_adapter.switch_player(current_player)

    return game_adapter.get_game_result(simulated_game)

def _get_total_slots(max_examples, reference_names):
    if max_examples is None:
        return None

    names_count = max(1, len(reference_names))
    return max(1, max_examples // names_count)

def _empty_reference_dataset(game_adapter, reference_names):
    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "source": "reference_training",
        "reference_names": list(reference_names),
        "available_states_count": 0,
        "reference_examples_count": 0,
        "examples_count": 0,
        "examples": [],
    }
