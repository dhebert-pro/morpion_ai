from tests.test_helpers import assert_equal, assert_true

from app.ai.tactical_evaluation import (
    create_tactical_probe,
    get_default_morpion_tactical_probes,
)

from app.ai.tactical_training import (
    create_move_score_example_from_tactical_probe,
    create_tactical_move_score_dataset,
    create_default_morpion_tactical_dataset,
    merge_move_score_datasets,
)


def test_create_move_score_example_from_tactical_probe_scores_expected_move():
    probe = create_tactical_probe(
        name="win_top_row",
        board=[
            "O", "O", None,
            "X", "X", None,
            None, None, None,
        ],
        expected_move=2,
        description="O doit gagner immédiatement.",
    )

    example = create_move_score_example_from_tactical_probe(probe)

    assert_equal(example["state_key"], "OO.XX....")
    assert_equal(example["player"], "O")
    assert_equal(example["best_move"], 2)
    assert_equal(example["source"], "tactical_probe")
    assert_equal(example["probe_name"], "win_top_row")

    scores_by_move = {}

    for score_data in example["move_scores"]:
        scores_by_move[score_data["move"]] = score_data["score"]

    assert_equal(scores_by_move[2], 1.0)

    for move, score in scores_by_move.items():
        if move != 2:
            assert_equal(score, 0.0)


def test_create_move_score_example_rejects_illegal_expected_move():
    probe = create_tactical_probe(
        name="illegal_probe",
        board=[
            "O", "O", "X",
            "X", None, None,
            None, None, None,
        ],
        expected_move=2,
        description="Le coup 2 est déjà occupé.",
    )

    error_was_raised = False

    try:
        create_move_score_example_from_tactical_probe(probe)
    except ValueError:
        error_was_raised = True

    assert_equal(error_was_raised, True)


def test_create_tactical_move_score_dataset_repeats_examples():
    probes = [
        create_tactical_probe(
            name="probe_1",
            board=[
                "O", "O", None,
                "X", "X", None,
                None, None, None,
            ],
            expected_move=2,
            description="O doit jouer 2.",
        ),
        create_tactical_probe(
            name="probe_2",
            board=[
                "X", "X", None,
                "O", None, None,
                None, "O", None,
            ],
            expected_move=2,
            description="O doit bloquer en 2.",
        ),
    ]

    dataset = create_tactical_move_score_dataset(
        probes=probes,
        repeat_count=3,
    )

    assert_equal(dataset["game"], "morpion")
    assert_equal(dataset["trained_player"], "O")
    assert_equal(dataset["opponent_player"], "X")
    assert_equal(dataset["source"], "tactical_probes")
    assert_equal(dataset["repeat_count"], 3)
    assert_equal(dataset["examples_count"], 6)
    assert_equal(len(dataset["examples"]), 6)


def test_create_default_morpion_tactical_dataset_uses_default_probes():
    probes = get_default_morpion_tactical_probes()

    dataset = create_default_morpion_tactical_dataset(
        repeat_count=2,
    )

    assert_equal(dataset["examples_count"], len(probes) * 2)

    for example in dataset["examples"]:
        assert_equal(example["source"], "tactical_probe")
        assert_true("probe_name" in example)
        assert_true("description" in example)


def test_merge_move_score_datasets_concatenates_examples():
    base_dataset = {
        "game": "morpion",
        "trained_player": "O",
        "opponent_player": "X",
        "training_games_count": 10,
        "simulations_per_move": 1,
        "examples_count": 2,
        "examples": [
            {
                "state_key": "base_1",
                "move_scores": [],
            },
            {
                "state_key": "base_2",
                "move_scores": [],
            },
        ],
    }

    extra_dataset = {
        "game": "morpion",
        "trained_player": "O",
        "opponent_player": "X",
        "examples_count": 1,
        "examples": [
            {
                "state_key": "extra_1",
                "move_scores": [],
            },
        ],
    }

    merged_dataset = merge_move_score_datasets(
        base_dataset,
        extra_dataset,
    )

    assert_equal(merged_dataset["game"], "morpion")
    assert_equal(merged_dataset["trained_player"], "O")
    assert_equal(merged_dataset["opponent_player"], "X")
    assert_equal(merged_dataset["base_examples_count"], 2)
    assert_equal(merged_dataset["extra_examples_count"], 1)
    assert_equal(merged_dataset["examples_count"], 3)
    assert_equal(merged_dataset["examples"][0]["state_key"], "base_1")
    assert_equal(merged_dataset["examples"][1]["state_key"], "base_2")
    assert_equal(merged_dataset["examples"][2]["state_key"], "extra_1")


TESTS = [
    ("Un exemple tactique donne 1 au coup attendu", test_create_move_score_example_from_tactical_probe_scores_expected_move),
    ("Un exemple tactique refuse un coup attendu illégal", test_create_move_score_example_rejects_illegal_expected_move),
    ("Le dataset tactique répète les exemples", test_create_tactical_move_score_dataset_repeats_examples),
    ("Le dataset tactique par défaut utilise les probes du morpion", test_create_default_morpion_tactical_dataset_uses_default_probes),
    ("La fusion de datasets concatène les exemples", test_merge_move_score_datasets_concatenates_examples),
]