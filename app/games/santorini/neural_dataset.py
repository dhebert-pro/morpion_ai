from app.games.santorini.action_index import ACTION_OUTPUT_SIZE
from app.games.santorini.encoding import SANTORINI_INPUT_SIZE


def encode_santorini_move_score_dataset(raw_dataset):
    examples = []

    for example in raw_dataset.get("examples", []):
        examples.append(encode_santorini_move_score_example(example))

    return {
        "game": "santorini",
        "trained_player": raw_dataset.get("trained_player", "O"),
        "input_size": SANTORINI_INPUT_SIZE,
        "output_size": ACTION_OUTPUT_SIZE,
        "source_examples_count": raw_dataset.get("examples_count", len(examples)),
        "encoded_examples_count": len(examples),
        "examples": examples,
    }


def encode_santorini_move_score_example(example):
    targets = [0.0 for _ in range(ACTION_OUTPUT_SIZE)]
    legal_moves_mask = [0.0 for _ in range(ACTION_OUTPUT_SIZE)]
    legal_move_indexes = []

    for move_score in example.get("move_scores", []):
        output_index = move_score["output_index"]
        _validate_output_index(output_index)
        targets[output_index] = float(move_score["score"])
        legal_moves_mask[output_index] = 1.0
        legal_move_indexes.append(output_index)

    return {
        "state_key": example["state_key"],
        "inputs": example["inputs"],
        "targets": targets,
        "legal_moves_mask": legal_moves_mask,
        "legal_move_indexes": legal_move_indexes,
        "best_move_index": example.get("best_output_index"),
        "best_action": example.get("best_action"),
        "best_score": example.get("best_score"),
    }


def _validate_output_index(output_index):
    if output_index < 0 or output_index >= ACTION_OUTPUT_SIZE:
        raise ValueError(
            "Index de sortie Santorini invalide : " + str(output_index)
        )
