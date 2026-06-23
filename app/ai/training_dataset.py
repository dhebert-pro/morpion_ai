import random

from app.ai.move_scoring import create_move_score_example
from app.ai.training import collect_training_states
from app.games.morpion.adapter import MORPION_ADAPTER
from app.utils.progress import print_progress


DEFAULT_DATASET_SEED = 0


def build_move_score_dataset(
    training_games_count,
    simulations_per_move,
    max_examples=None,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
    seed=DEFAULT_DATASET_SEED,
):
    collection_rng = random.Random(seed)
    scoring_rng = random.Random(seed + 2)

    states_to_learn = collect_training_states(
        training_games_count,
        show_progress=False,
        game_adapter=game_adapter,
        rng=collection_rng,
    )

    state_items = select_balanced_state_items(
        list(states_to_learn.items()),
        max_examples=max_examples,
        seed=seed + 1,
    )

    examples = []
    total_states = len(state_items)

    for index, item in enumerate(state_items, start=1):
        game = item[1]

        example = create_move_score_example(
            game,
            simulations_per_move,
            game_adapter,
            rng=scoring_rng,
        )
        examples.append(example)

        if show_progress:
            print_progress("Création dataset       ", index, total_states)

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "training_games_count": training_games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "dataset_seed": seed,
        "available_states_count": len(states_to_learn),
        "examples_count": len(examples),
        "examples": examples,
    }


def select_balanced_state_items(state_items, max_examples=None, seed=DEFAULT_DATASET_SEED):
    sorted_items = sorted(state_items, key=lambda item: item[0])

    if max_examples is None or len(sorted_items) <= max_examples:
        return sorted_items

    groups = _group_state_items_by_filled_cells(sorted_items)
    generator = random.Random(seed)

    for group_items in groups.values():
        generator.shuffle(group_items)

    selected_items = _pick_round_robin_from_groups(groups, max_examples)
    generator.shuffle(selected_items)

    return selected_items


def summarize_move_score_dataset(dataset):
    examples = dataset.get("examples", [])

    scored_moves_count = 0
    best_score_total = 0.0
    examples_with_scores = 0

    for example in examples:
        move_scores = example.get("move_scores", [])
        scored_moves_count += len(move_scores)

        if len(move_scores) > 0:
            best_score = max(score_data["score"] for score_data in move_scores)
            best_score_total += best_score
            examples_with_scores += 1

    if len(examples) == 0:
        average_legal_moves = 0.0
    else:
        average_legal_moves = scored_moves_count / len(examples)

    if examples_with_scores == 0:
        average_best_score = 0.0
    else:
        average_best_score = best_score_total / examples_with_scores

    return {
        "game": dataset.get("game"),
        "examples_count": len(examples),
        "available_states_count": dataset.get("available_states_count", len(examples)),
        "scored_moves_count": scored_moves_count,
        "average_legal_moves": round(average_legal_moves, 2),
        "average_best_score": round(average_best_score, 3),
    }


def _group_state_items_by_filled_cells(state_items):
    groups = {}

    for item in state_items:
        state_key = item[0]
        filled_cells = _count_filled_cells(state_key)

        if filled_cells not in groups:
            groups[filled_cells] = []

        groups[filled_cells].append(item)

    return groups


def _count_filled_cells(state_key):
    filled_cells = 0

    for cell in state_key:
        if cell != ".":
            filled_cells += 1

    return filled_cells


def _pick_round_robin_from_groups(groups, max_examples):
    selected_items = []
    group_keys = sorted(groups.keys())
    group_positions = {group_key: 0 for group_key in group_keys}

    while len(selected_items) < max_examples:
        added_on_this_pass = False

        for group_key in group_keys:
            group_items = groups[group_key]
            position = group_positions[group_key]

            if position >= len(group_items):
                continue

            selected_items.append(group_items[position])
            group_positions[group_key] = position + 1
            added_on_this_pass = True

            if len(selected_items) >= max_examples:
                break

        if not added_on_this_pass:
            break

    return selected_items
