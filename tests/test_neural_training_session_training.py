from tests.test_helpers import assert_equal, assert_true
from tests.neural_training_session_fixtures import create_small_encoded_dataset

from app.ai.neural_network import SimpleNeuralNetwork
from app.ai.neural_training_session import train_network_on_encoded_dataset


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
    assert_equal(len(SimpleNeuralNetwork.from_dict(model_data).predict([1.0, 0.0, 0.0, 0.0])), 3)


def test_train_network_on_encoded_dataset_can_continue_from_existing_model():
    dataset = create_small_encoded_dataset()
    first_training = train_network_on_encoded_dataset(dataset, 6, 80, 0.25, False, seed=0)

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
    incompatible_network = SimpleNeuralNetwork(4, 5, 3, learning_rate=0.25, seed=0)
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
    ("L'entraînement sur dataset encodé réduit l'erreur", test_train_network_on_encoded_dataset_reduces_error),
    ("L'entraînement retourne un modèle sérialisable", test_train_network_on_encoded_dataset_returns_serializable_model_data),
    ("L'entraînement peut reprendre depuis un modèle existant", test_train_network_on_encoded_dataset_can_continue_from_existing_model),
    ("L'entraînement refuse un modèle existant incompatible", test_train_network_on_encoded_dataset_rejects_incompatible_existing_model),
]
