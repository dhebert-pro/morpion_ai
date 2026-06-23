from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_encoding import (
    encode_state_key_as_neural_input,
    encode_move_scores_as_target_vectors,
    encode_move_score_example,
    encode_move_score_dataset,
)
from app.ai.morpion_line_features import encode_line_threats_as_neural_input


def test_encode_state_key_as_neural_input_uses_two_planes():
    vector = encode_state_key_as_neural_input(
        state_key="O.X......",
        trained_player="O",
        opponent_player="X",
    )

    expected_prefix = [
        1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]

    assert_equal(vector[:18], expected_prefix)
    assert_equal(len(vector), 34)


def test_encode_line_threats_detects_immediate_threats():
    threats = encode_line_threats_as_neural_input(
        state_key="OO.XX....",
        player="O",
    )

    assert_equal(threats[0], 1.0)
    assert_equal(sum(threats), 1.0)


def test_encode_move_scores_as_target_vectors_creates_targets_and_mask():
    move_scores = [
        {"move": 2, "score": 1.0},
        {"move": 5, "score": 0.35},
        {"move": 6, "score": 0.4},
    ]

    targets, legal_moves_mask = encode_move_scores_as_target_vectors(
        move_scores,
        output_size=9,
    )

    assert_equal(targets, [0.0, 0.0, 1.0, 0.0, 0.0, 0.35, 0.4, 0.0, 0.0])
    assert_equal(legal_moves_mask, [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])


def test_encode_move_score_example_keeps_debug_information():
    example = {
        "state_key": "OO.XX....",
        "player": "O",
        "move_scores": [
            {"move": 2, "score": 1.0},
            {"move": 5, "score": 0.35},
            {"move": 6, "score": 0.4},
        ],
        "best_move": 2,
        "source": "tactical_probe",
        "probe_name": "test_probe",
        "description": "Position de test",
        "best_moves": [2],
    }

    encoded_example = encode_move_score_example(example)

    assert_equal(encoded_example["state_key"], "OO.XX....")
    assert_equal(len(encoded_example["inputs"]), 34)
    assert_equal(len(encoded_example["targets"]), 9)
    assert_equal(len(encoded_example["legal_moves_mask"]), 9)
    assert_equal(encoded_example["legal_moves"], [2, 5, 6])
    assert_equal(encoded_example["legal_move_indexes"], [2, 5, 6])
    assert_equal(encoded_example["best_move"], 2)
    assert_equal(encoded_example["best_move_index"], 2)
    assert_equal(encoded_example["targets"][2], 1.0)
    assert_equal(encoded_example["legal_moves_mask"][2], 1.0)
    assert_equal(encoded_example["source"], "tactical_probe")
    assert_equal(encoded_example["probe_name"], "test_probe")
    assert_equal(encoded_example["description"], "Position de test")
    assert_equal(encoded_example["best_moves"], [2])


def test_encode_move_score_dataset_creates_numeric_dataset():
    dataset = {
        "game": "morpion",
        "trained_player": "O",
        "opponent_player": "X",
        "examples_count": 2,
        "examples": [
            {
                "state_key": "OO.XX....",
                "player": "O",
                "move_scores": [
                    {"move": 2, "score": 1.0},
                    {"move": 5, "score": 0.35},
                ],
                "best_move": 2,
            },
            {
                "state_key": "OXXOOX...",
                "player": "O",
                "move_scores": [
                    {"move": 6, "score": 0.75},
                    {"move": 7, "score": 0.5},
                    {"move": 8, "score": 0.25},
                ],
                "best_move": 6,
            },
        ],
    }

    encoded_dataset = encode_move_score_dataset(dataset)

    assert_equal(encoded_dataset["game"], "morpion")
    assert_equal(encoded_dataset["trained_player"], "O")
    assert_equal(encoded_dataset["opponent_player"], "X")
    assert_equal(encoded_dataset["input_size"], 34)
    assert_equal(encoded_dataset["output_size"], 9)
    assert_equal(encoded_dataset["source_examples_count"], 2)
    assert_equal(encoded_dataset["encoded_examples_count"], 2)

    first_example = encoded_dataset["examples"][0]

    assert_true("inputs" in first_example)
    assert_true("targets" in first_example)
    assert_true("legal_moves_mask" in first_example)


TESTS = [
    ("L'encodage d'un état utilise deux plans numériques", test_encode_state_key_as_neural_input_uses_two_planes),
    ("L'encodage détecte les menaces immédiates", test_encode_line_threats_detects_immediate_threats),
    ("L'encodage des scores crée targets et mask", test_encode_move_scores_as_target_vectors_creates_targets_and_mask),
    ("L'encodage d'un exemple conserve les informations de debug", test_encode_move_score_example_keeps_debug_information),
    ("L'encodage du dataset crée un dataset numérique", test_encode_move_score_dataset_creates_numeric_dataset),
]