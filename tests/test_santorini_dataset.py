from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.dataset import (
    build_santorini_move_score_dataset,
    summarize_santorini_dataset,
)
from app.games.santorini.dataset_collection import collect_santorini_training_states
from app.games.santorini.dataset_report import format_santorini_dataset_report


def test_collect_santorini_training_states_returns_o_turns():
    states = collect_santorini_training_states(games_count=2, seed=0)
    assert_true(len(states) > 0)

    first_game = next(iter(states.values()))
    assert_equal(first_game["current_player"], "O")
    assert_equal(first_game["phase"], "play")


def test_build_santorini_dataset_scores_indexed_moves():
    dataset = build_santorini_move_score_dataset(
        games_count=3,
        simulations_per_move=1,
        max_examples=2,
        seed=0,
    )
    assert_equal(dataset["game"], "santorini")
    assert_equal(dataset["examples_count"], 2)

    example = dataset["examples"][0]
    assert_true(example["legal_moves_count"] > 0)
    assert_true(example["best_output_index"] is not None)
    assert_true(0 <= example["best_score"] <= 1)
    assert_true("action" in example["move_scores"][0])


def test_santorini_dataset_summary_counts_moves():
    dataset = build_santorini_move_score_dataset(
        games_count=3,
        simulations_per_move=1,
        max_examples=2,
        seed=0,
    )
    summary = summarize_santorini_dataset(dataset)
    assert_equal(summary["examples_count"], 2)
    assert_true(summary["scored_moves_count"] >= 2)
    assert_true(summary["average_legal_moves"] > 0)
    assert_true("average_score_spread" in summary)
    assert_true("decisive_examples_count" in summary)


def test_santorini_dataset_report_is_readable():
    dataset = build_santorini_move_score_dataset(
        games_count=3,
        simulations_per_move=1,
        max_examples=1,
        seed=0,
    )
    report = format_santorini_dataset_report(dataset, examples_limit=1, moves_limit=2)
    assert_true("Dataset Santorini Monte-Carlo" in report)
    assert_true("Exemple 1" in report)
    assert_true("sortie réseau" in report)
    assert_true("Écart moyen meilleur-pire coup" in report)


TESTS = [
    ("Santorini collecte des états d'entraînement", test_collect_santorini_training_states_returns_o_turns),
    ("Santorini dataset score des coups indexés", test_build_santorini_dataset_scores_indexed_moves),
    ("Santorini dataset résume les coups", test_santorini_dataset_summary_counts_moves),
    ("Santorini dataset produit un rapport lisible", test_santorini_dataset_report_is_readable),
]
