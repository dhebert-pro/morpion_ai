import random

from app.games.santorini.dataset_collection import (
    collect_santorini_training_states,
    select_santorini_state_items,
)
from app.games.santorini.move_scoring import create_santorini_move_score_example
from app.utils.progress import print_progress


DEFAULT_SANTORINI_DATASET_SEED = 0


def build_santorini_move_score_dataset(
    games_count,
    simulations_per_move,
    max_examples=None,
    seed=DEFAULT_SANTORINI_DATASET_SEED,
    show_progress=False,
):
    states = collect_santorini_training_states(games_count, seed=seed)
    selected_items = select_santorini_state_items(
        states,
        max_examples=max_examples,
        seed=seed + 1,
    )
    rng = random.Random(seed + 2)
    examples = []
    total = len(selected_items)

    for index, item in enumerate(selected_items, start=1):
        example = create_santorini_move_score_example(
            item[1],
            simulations_per_move,
            rng,
        )
        examples.append(example)

        if show_progress:
            print_progress("Dataset Santorini    ", index, total)

    return {
        "game": "santorini",
        "trained_player": "O",
        "training_games_count": games_count,
        "simulations_per_move": simulations_per_move,
        "max_examples": max_examples,
        "dataset_seed": seed,
        "available_states_count": len(states),
        "examples_count": len(examples),
        "examples": examples,
    }


def summarize_santorini_dataset(dataset):
    examples = dataset.get("examples", [])
    scored_moves_count = sum(len(example.get("move_scores", [])) for example in examples)
    best_scores = [example["best_score"] for example in examples if example["best_score"] is not None]
    spreads = [_score_spread(example) for example in examples]
    decisive_count = len([spread for spread in spreads if spread >= 0.25])

    if examples:
        average_legal_moves = scored_moves_count / len(examples)
    else:
        average_legal_moves = 0.0

    if best_scores:
        average_best_score = sum(best_scores) / len(best_scores)
    else:
        average_best_score = 0.0

    if spreads:
        average_score_spread = sum(spreads) / len(spreads)
    else:
        average_score_spread = 0.0

    return {
        "game": dataset.get("game"),
        "examples_count": len(examples),
        "available_states_count": dataset.get("available_states_count", 0),
        "scored_moves_count": scored_moves_count,
        "average_legal_moves": round(average_legal_moves, 2),
        "average_best_score": round(average_best_score, 3),
        "average_score_spread": round(average_score_spread, 3),
        "decisive_examples_count": decisive_count,
    }


def _score_spread(example):
    scores = [move_score["score"] for move_score in example.get("move_scores", [])]

    if not scores:
        return 0.0

    return max(scores) - min(scores)
