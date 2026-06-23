from tests.test_helpers import assert_equal, assert_true
from tests.neural_training_session_fixtures import create_small_encoded_dataset

from app.ai.neural_network import SimpleNeuralNetwork
from app.ai.neural_training_session import (
    get_examples_from_encoded_dataset,
    create_neural_network_from_encoded_dataset,
    create_neural_network_from_model_data,
    validate_network_matches_encoded_dataset,
    compute_average_error_on_encoded_examples,
)


def test_get_examples_from_encoded_dataset_returns_examples():
    dataset = create_small_encoded_dataset()
    examples = get_examples_from_encoded_dataset(dataset)

    assert_equal(len(examples), 2)
    assert_equal(examples[0]["state_key"], "state_1")
    assert_equal(examples[1]["state_key"], "state_2")


def test_create_neural_network_from_encoded_dataset_uses_dataset_sizes():
    dataset = create_small_encoded_dataset()
    network = create_neural_network_from_encoded_dataset(dataset, 6, 0.2, seed=0)

    assert_true(isinstance(network, SimpleNeuralNetwork))
    assert_equal(network.input_size, 4)
    assert_equal(network.hidden_size, 6)
    assert_equal(network.output_size, 3)
    assert_equal(network.learning_rate, 0.2)


def test_create_neural_network_from_model_data_loads_existing_weights():
    dataset = create_small_encoded_dataset()
    network = create_neural_network_from_encoded_dataset(dataset, 6, 0.2, seed=0)

    loaded_network = create_neural_network_from_model_data(
        model_data=network.to_dict(),
        learning_rate=0.15,
    )

    assert_equal(loaded_network.input_size, 4)
    assert_equal(loaded_network.hidden_size, 6)
    assert_equal(loaded_network.output_size, 3)
    assert_equal(loaded_network.learning_rate, 0.15)


def test_validate_network_matches_encoded_dataset_accepts_matching_network():
    dataset = create_small_encoded_dataset()
    network = create_neural_network_from_encoded_dataset(dataset, 6, 0.2, seed=0)

    validate_network_matches_encoded_dataset(
        network=network,
        encoded_dataset=dataset,
        expected_hidden_size=6,
    )

    assert_equal(True, True)


def test_validate_network_matches_encoded_dataset_rejects_wrong_hidden_size():
    dataset = create_small_encoded_dataset()
    network = create_neural_network_from_encoded_dataset(dataset, 6, 0.2, seed=0)
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
    network = create_neural_network_from_encoded_dataset(dataset, 6, 0.2, seed=0)

    error = compute_average_error_on_encoded_examples(network, [])

    assert_equal(error, 0.0)


TESTS = [
    ("La session récupère les exemples d'un dataset encodé", test_get_examples_from_encoded_dataset_returns_examples),
    ("La session crée un réseau aux bonnes dimensions", test_create_neural_network_from_encoded_dataset_uses_dataset_sizes),
    ("La session recharge un réseau depuis model_data", test_create_neural_network_from_model_data_loads_existing_weights),
    ("La validation accepte un réseau compatible", test_validate_network_matches_encoded_dataset_accepts_matching_network),
    ("La validation refuse une mauvaise taille cachée", test_validate_network_matches_encoded_dataset_rejects_wrong_hidden_size),
    ("L'erreur moyenne vaut 0 sans exemple", test_compute_average_error_returns_zero_without_examples),
]
