from tests.test_helpers import assert_equal, assert_true

from app.ai.training_dataset import (
    build_move_score_dataset,
    summarize_move_score_dataset,
)


def test_build_move_score_dataset_creates_examples():
    dataset = build_move_score_dataset(
        training_games_count=10,
        simulations_per_move=1,
        max_examples=5,
        show_progress=False,
    )

    assert_equal(dataset["game"], "morpion")
    assert_equal(dataset["trained_player"], "O")
    assert_true(dataset["examples_count"] > 0)
    assert_true(dataset["examples_count"] <= 5)

    first_example = dataset["examples"][0]

    assert_true("state_key" in first_example)
    assert_true("player" in first_example)
    assert_true("move_scores" in first_example)
    assert_true("best_move" in first_example)


def test_build_move_score_dataset_respects_max_examples():
    dataset = build_move_score_dataset(
        training_games_count=50,
        simulations_per_move=1,
        max_examples=3,
        show_progress=False,
    )

    assert_true(dataset["examples_count"] <= 3)


def test_summarize_move_score_dataset_counts_examples_and_moves():
    dataset = {
        "game": "morpion",
        "examples": [
            {
                "move_scores": [
                    {"move": 0, "score": 0.2},
                    {"move": 1, "score": 0.8},
                ],
            },
            {
                "move_scores": [
                    {"move": 4, "score": 1.0},
                ],
            },
        ],
    }

    summary = summarize_move_score_dataset(dataset)

    assert_equal(summary["game"], "morpion")
    assert_equal(summary["examples_count"], 2)
    assert_equal(summary["scored_moves_count"], 3)
    assert_equal(summary["average_legal_moves"], 1.5)
    assert_equal(summary["average_best_score"], 0.9)


TESTS = [
    ("Le dataset de scoring crée des exemples exploitables", test_build_move_score_dataset_creates_examples),
    ("Le dataset respecte le nombre maximal d'exemples", test_build_move_score_dataset_respects_max_examples),
    ("Le résumé du dataset compte les exemples et les coups scorés", test_summarize_move_score_dataset_counts_examples_and_moves),
]