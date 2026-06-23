from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_training_session import (
    get_examples_from_encoded_dataset,
    create_neural_network_from_encoded_dataset,
    create_neural_network_from_model_data,
    validate_network_matches_encoded_dataset,
    compute_average_error_on_encoded_examples,
    train_network_on_encoded_dataset,
)


def create_small_encoded_dataset():
    return {
        "game": "test_game",
        "trained_player": "B",
        "opponent_player": "A",
        "input_size": 4,
        "output_size": 3,
        "source_examples_count": 2,
        "encoded_examples_count": 2,
        "examples": [
            {
                "state_key": "state_1",
                "inputs": [1.0, 0.0, 0.0, 0.0],
                "targets": [1.0, 0.0, 0.0],
                "legal_moves_mask": [1.0, 1.0, 0.0],
                "legal_moves": [0, 1],
                "legal_move_indexes": [0, 1],
                "best_move": 0,
                "best_move_index": 0,
            },
            {
                "state_key": "state_2",
                "inputs": [0.0, 1.0, 0.0, 0.0],
                "targets": [0.0, 1.0, 0.0],
                "legal_moves_mask": [1.0, 1.0, 0.0],
                "legal_moves": [0, 1],
                "best_move": 1,
                "best_move_index": 1,
            },
        ],
    }


def test_get_examples_from_encoded_dataset_returns_examples():
    dataset = create_small_encoded_dataset()

    examples = get_examples_from_encoded_dataset(dataset)

    assert_equal(len(examples), 2)
    assert_equal(examples[0]["state_key"], "state_1")
    assert_equal(examples[1]["state_key"], "state_2")


def test_create_neural_network_from_encoded_dataset_uses_dataset_sizes():
    dataset = create_small_encoded_dataset()

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        learning_rate=0.2,
        seed=0,
    )

    assert_true(isinstance(network, SimpleNeuralNetwork))
    assert_equal(network.input_size, 4)
    assert_equal(network.hidden_size, 6)
    assert_equal(network.output_size, 3)
    assert_equal(network.learning_rate, 0.2)


def test_create_neural_network_from_model_data_loads_existing_weights():
    dataset = create_small_encoded_dataset()

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        learning_rate=0.2,
        seed=0,
    )

    model_data = network.to_dict()

    loaded_network = create_neural_network_from_model_data(
        model_data=model_data,
        learning_rate=0.15,
    )

    assert_equal(loaded_network.input_size, 4)
    assert_equal(loaded_network.hidden_size, 6)
    assert_equal(loaded_network.output_size, 3)
    assert_equal(loaded_network.learning_rate, 0.15)


def test_validate_network_matches_encoded_dataset_accepts_matching_network():
    dataset = create_small_encoded_dataset()

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        learning_rate=0.2,
        seed=0,
    )

    validate_network_matches_encoded_dataset(
        network=network,
        encoded_dataset=dataset,
        expected_hidden_size=6,
    )

    assert_equal(True, True)


def test_validate_network_matches_encoded_dataset_rejects_wrong_hidden_size():
    dataset = create_small_encoded_dataset()

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        learning_rate=0.2,
        seed=0,
    )

    error_was_raised = False

    try:
        validate_network_matches_encoded_dataset(
            network=network,
            encoded_dataset=dataset,
            expected_hidden_size=5,
        )
    except ValueError:
        error_was_raised = True

    assert_equal(error_was_raised, True)


def test_compute_average_error_returns_zero_without_examples():
    dataset = create_small_encoded_dataset()

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        learning_rate=0.2,
        seed=0,
    )

    error = compute_average_error_on_encoded_examples(
        network,
        [],
    )

    assert_equal(error, 0.0)


def test_train_network_on_encoded_dataset_reduces_error():
    dataset = create_small_encoded_dataset()

    training_result = train_network_on_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        epochs=600,
        learning_rate=0.25,
        show_progress=False,
        seed=0,
    )

    assert_equal(training_result["examples_count"], 2)
    assert_equal(training_result["input_size"], 4)
    assert_equal(training_result["hidden_size"], 6)
    assert_equal(training_result["output_size"], 3)
    assert_equal(training_result["epochs"], 600)
    assert_equal(training_result["learning_rate"], 0.25)
    assert_equal(training_result["started_from_existing_model"], False)

    assert_true(training_result["final_error"] < training_result["initial_error"])
    assert_true(training_result["final_error"] < 0.05)


def test_train_network_on_encoded_dataset_returns_serializable_model_data():
    dataset = create_small_encoded_dataset()

    training_result = train_network_on_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        epochs=10,
        learning_rate=0.25,
        show_progress=False,
        seed=0,
    )

    model_data = training_result["model_data"]

    assert_equal(model_data["type"], "simple_neural_network")
    assert_equal(model_data["input_size"], 4)
    assert_equal(model_data["hidden_size"], 6)
    assert_equal(model_data["output_size"], 3)

    loaded_network = SimpleNeuralNetwork.from_dict(model_data)
    predictions = loaded_network.predict([1.0, 0.0, 0.0, 0.0])

    assert_equal(len(predictions), 3)


def test_train_network_on_encoded_dataset_can_continue_from_existing_model():
    dataset = create_small_encoded_dataset()

    first_training = train_network_on_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        epochs=80,
        learning_rate=0.25,
        show_progress=False,
        seed=0,
    )

    continued_training = train_network_on_encoded_dataset(
        encoded_dataset=dataset,
        hidden_size=6,
        epochs=300,
        learning_rate=0.25,
        show_progress=False,
        seed=999,
        initial_model_data=first_training["model_data"],
    )

    assert_equal(continued_training["started_from_existing_model"], True)
    assert_true(continued_training["initial_error"] <= first_training["final_error"] + 0.000000001)
    assert_true(continued_training["final_error"] < continued_training["initial_error"])


def test_train_network_on_encoded_dataset_rejects_incompatible_existing_model():
    dataset = create_small_encoded_dataset()

    incompatible_network = SimpleNeuralNetwork(
        input_size=4,
        hidden_size=5,
        output_size=3,
        learning_rate=0.25,
        seed=0,
    )

    error_was_raised = False

    try:
        train_network_on_encoded_dataset(
            encoded_dataset=dataset,
            hidden_size=6,
            epochs=10,
            learning_rate=0.25,
            show_progress=False,
            seed=0,
            initial_model_data=incompatible_network.to_dict(),
        )
    except ValueError:
        error_was_raised = True

    assert_equal(error_was_raised, True)


TESTS = [
    ("La session récupère les exemples d'un dataset encodé", test_get_examples_from_encoded_dataset_returns_examples),
    ("La session crée un réseau aux bonnes dimensions", test_create_neural_network_from_encoded_dataset_uses_dataset_sizes),
    ("La session recharge un réseau depuis model_data", test_create_neural_network_from_model_data_loads_existing_weights),
    ("La validation accepte un réseau compatible", test_validate_network_matches_encoded_dataset_accepts_matching_network),
    ("La validation refuse une mauvaise taille cachée", test_validate_network_matches_encoded_dataset_rejects_wrong_hidden_size),
    ("L'erreur moyenne vaut 0 sans exemple", test_compute_average_error_returns_zero_without_examples),
    ("L'entraînement sur dataset encodé réduit l'erreur", test_train_network_on_encoded_dataset_reduces_error),
    ("L'entraînement retourne un modèle sérialisable", test_train_network_on_encoded_dataset_returns_serializable_model_data),
    ("L'entraînement peut reprendre depuis un modèle existant", test_train_network_on_encoded_dataset_can_continue_from_existing_model),
    ("L'entraînement refuse un modèle existant incompatible", test_train_network_on_encoded_dataset_rejects_incompatible_existing_model),
]