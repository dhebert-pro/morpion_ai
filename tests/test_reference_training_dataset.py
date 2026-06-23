from tests.test_helpers import assert_equal, assert_true

from app.ai.reference_training_dataset import (
    build_reference_move_score_dataset,
    collect_reference_training_states,
    create_reference_move_score_example,
)
from app.ai.tactical_training import merge_move_score_datasets
from app.ai.training_dataset import build_move_score_dataset
from app.games.morpion.rules import create_new_game


def test_collect_reference_training_states_returns_o_positions():
    states = collect_reference_training_states(
        training_games_count=5,
        reference_name="tactical",
        rng=None,
    )

    assert_true(len(states) > 0)

    for state_key in states:
        assert_equal(state_key.count("X"), state_key.count("O") + 1)


def test_reference_example_scores_legal_moves():
    game = create_new_game()
    game["board"] = ["X", None, None, None, "O", None, None, None, None]

    example = create_reference_move_score_example(
        game=game,
        simulations_per_move=2,
        reference_name="tactical",
    )

    assert_equal(example["source"], "reference_tactical")
    assert_true(len(example["move_scores"]) > 0)
    assert_true(example["best_move"] is not None)


def test_build_reference_dataset_respects_max_examples():
    dataset = build_reference_move_score_dataset(
        training_games_count=10,
        simulations_per_move=1,
        max_examples=4,
        reference_names=["tactical"],
        seed=0,
    )

    assert_equal(dataset["source"], "reference_training")
    assert_equal(dataset["reference_examples_count"], len(dataset["examples"]))
    assert_true(len(dataset["examples"]) <= 4)


def test_merge_preserves_base_tactical_and_reference_counts():
    base_dataset = build_move_score_dataset(
        training_games_count=3,
        simulations_per_move=1,
        max_examples=2,
        seed=0,
    )
    reference_dataset = build_reference_move_score_dataset(
        training_games_count=3,
        simulations_per_move=1,
        max_examples=2,
        reference_names=["tactical"],
        seed=0,
    )

    merged = merge_move_score_datasets(base_dataset, reference_dataset)

    assert_equal(merged["base_examples_count"], len(base_dataset["examples"]))
    assert_equal(merged["reference_examples_count"], len(reference_dataset["examples"]))
    assert_equal(merged["examples_count"], len(merged["examples"]))


TESTS = [
    ("La collecte référence retourne des positions de O", test_collect_reference_training_states_returns_o_positions),
    ("Un exemple référence score les coups légaux", test_reference_example_scores_legal_moves),
    ("Le dataset référence respecte max_examples", test_build_reference_dataset_respects_max_examples),
    ("La fusion préserve les compteurs référence", test_merge_preserves_base_tactical_and_reference_counts),
]
