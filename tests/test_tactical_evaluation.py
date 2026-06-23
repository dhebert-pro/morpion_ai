from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.tactical_evaluation import (
    create_game_from_board,
    create_tactical_probe,
    get_default_morpion_tactical_probes,
    run_tactical_probe,
    run_tactical_evaluation,
    summarize_tactical_evaluation,
    format_tactical_evaluation_report,
)


def create_deterministic_model(output_biases):
    network = SimpleNeuralNetwork(
        input_size=18,
        hidden_size=3,
        output_size=9,
        learning_rate=0.1,
        seed=0,
    )

    network.input_hidden_weights = [
        [0.0 for _ in range(network.hidden_size)]
        for _ in range(network.input_size)
    ]

    network.hidden_biases = [
        0.0 for _ in range(network.hidden_size)
    ]

    network.hidden_output_weights = [
        [0.0 for _ in range(network.output_size)]
        for _ in range(network.hidden_size)
    ]

    network.output_biases = output_biases.copy()

    return network.to_dict()


def test_create_game_from_board_copies_board():
    board = [
        "O", "O", None,
        "X", "X", None,
        None, None, None,
    ]

    game = create_game_from_board(board)

    assert_equal(game["board"], board)

    board[0] = None

    assert_equal(game["board"][0], "O")


def test_default_morpion_tactical_probes_have_required_fields():
    probes = get_default_morpion_tactical_probes()

    assert_true(len(probes) >= 4)

    for probe in probes:
        assert_true("name" in probe)
        assert_true("board" in probe)
        assert_true("expected_move" in probe)
        assert_true("expected_moves" in probe)
        assert_true("description" in probe)
        assert_equal(len(probe["board"]), 9)


def test_create_tactical_probe_accepts_multiple_expected_moves():
    probe = create_tactical_probe(
        name="multiple_good_moves",
        board=[
            None, None, None,
            None, "X", None,
            None, None, None,
        ],
        expected_moves=[0, 2, 6, 8],
        description="O peut jouer un coin.",
    )

    assert_equal(probe["expected_move"], 0)
    assert_equal(probe["expected_moves"], [0, 2, 6, 8])


def test_run_tactical_probe_accepts_any_expected_move():
    probe = create_tactical_probe(
        name="corner_answer",
        board=[
            None, None, None,
            None, "X", None,
            None, None, None,
        ],
        expected_moves=[0, 2, 6, 8],
        description="O peut jouer un coin.",
    )

    model_data = create_deterministic_model([
        0.0, 0.1, 0.2,
        0.3, 0.4, 0.5,
        3.0, 0.7, 0.8,
    ])

    result = run_tactical_probe(
        probe,
        model_data,
    )

    assert_equal(result["chosen_move"], 6)
    assert_equal(result["passed"], True)


def test_run_tactical_probe_detects_success():
    probe = create_tactical_probe(
        name="test_win",
        board=[
            "O", "O", None,
            "X", "X", None,
            None, None, None,
        ],
        expected_move=2,
        description="O doit jouer 2.",
    )

    model_data = create_deterministic_model([
        0.0, 0.1, 3.0,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
    ])

    result = run_tactical_probe(
        probe,
        model_data,
    )

    assert_equal(result["name"], "test_win")
    assert_equal(result["expected_move"], 2)
    assert_equal(result["chosen_move"], 2)
    assert_equal(result["passed"], True)


def test_run_tactical_evaluation_runs_all_probes():
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

    model_data = create_deterministic_model([
        0.0, 0.1, 3.0,
        0.3, 0.4, 0.5,
        0.6, 0.7, 0.8,
    ])

    results = run_tactical_evaluation(
        probes,
        model_data,
    )

    assert_equal(len(results), 2)
    assert_equal(results[0]["passed"], True)
    assert_equal(results[1]["passed"], True)


def test_summarize_tactical_evaluation_counts_successes():
    results = [
        {
            "passed": True,
        },
        {
            "passed": False,
        },
        {
            "passed": True,
        },
    ]

    summary = summarize_tactical_evaluation(results)

    assert_equal(summary["total_count"], 3)
    assert_equal(summary["passed_count"], 2)
    assert_equal(summary["failed_count"], 1)
    assert_true(abs(summary["success_rate"] - 66.6666666667) < 0.000001)


def test_format_tactical_evaluation_report_contains_details():
    results = [
        {
            "name": "win_top_row",
            "description": "O doit gagner.",
            "expected_move": 2,
            "chosen_move": 2,
            "passed": True,
        },
        {
            "name": "block_diagonal",
            "description": "O doit bloquer.",
            "expected_move": 8,
            "chosen_move": 6,
            "passed": False,
        },
    ]

    text = format_tactical_evaluation_report(results)

    assert_true("Résumé évaluation tactique" in text)
    assert_true("Tests réussis : 1 / 2" in text)
    assert_true("OK - win_top_row" in text)
    assert_true("NON - block_diagonal" in text)
    assert_true("attendu : 8" in text)
    assert_true("choisi : 6" in text)


TESTS = [
    ("La création d'un jeu depuis un plateau copie le plateau", test_create_game_from_board_copies_board),
    ("Les probes tactiques par défaut ont les champs nécessaires", test_default_morpion_tactical_probes_have_required_fields),
    ("Une probe tactique accepte plusieurs coups attendus", test_create_tactical_probe_accepts_multiple_expected_moves),
    ("Un test tactique accepte n'importe quel bon coup", test_run_tactical_probe_accepts_any_expected_move),
    ("Un test tactique détecte une réussite", test_run_tactical_probe_detects_success),
    ("L'évaluation tactique exécute toutes les probes", test_run_tactical_evaluation_runs_all_probes),
    ("Le résumé tactique compte les réussites", test_summarize_tactical_evaluation_counts_successes),
    ("Le rapport tactique contient les détails", test_format_tactical_evaluation_report_contains_details),
]