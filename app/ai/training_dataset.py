from app.ai.move_scoring import create_move_score_example
from app.ai.training import collect_training_states
from app.games.morpion.adapter import MORPION_ADAPTER
from app.utils.progress import print_progress


def build_move_score_dataset(
    training_games_count,
    simulations_per_move,
    max_examples=None,
    show_progress=False,
    game_adapter=MORPION_ADAPTER,
):
    states_to_learn = collect_training_states(
        training_games_count,
        show_progress=False,
        game_adapter=game_adapter,
    )

    state_items = sorted(states_to_learn.items(), key=lambda item: item[0])

    if max_examples is not None:
        state_items = state_items[:max_examples]

    examples = []
    total_states = len(state_items)

    for index, item in enumerate(state_items, start=1):
        game = item[1]

        example = create_move_score_example(
            game,
            simulations_per_move,
            game_adapter,
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
        "examples_count": len(examples),
        "examples": examples,
    }


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
        "scored_moves_count": scored_moves_count,
        "average_legal_moves": round(average_legal_moves, 2),
        "average_best_score": round(average_best_score, 3),
    }