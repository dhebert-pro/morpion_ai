from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import (
    SimpleNeuralNetwork,
    compute_masked_mean_squared_error,
)


def test_network_predicts_one_score_per_output():
    network = SimpleNeuralNetwork(
        input_size=4,
        hidden_size=5,
        output_size=3,
        seed=0,
    )

    predictions = network.predict([1.0, 0.0, 1.0, 0.0])

    assert_equal(len(predictions), 3)

    for prediction in predictions:
        assert_true(0.0 <= prediction <= 1.0)


def test_masked_error_ignores_illegal_outputs():
    predictions = [0.9, 0.1, 0.8]
    targets = [0.0, 0.0, 0.0]
    legal_moves_mask = [0.0, 1.0, 0.0]

    error = compute_masked_mean_squared_error(
        predictions,
        targets,
        legal_moves_mask,
    )

    assert_true(abs(error - 0.01) < 0.000000001)


def test_network_training_reduces_error_on_one_example():
    network = SimpleNeuralNetwork(
        input_size=4,
        hidden_size=6,
        output_size=3,
        learning_rate=0.2,
        seed=1,
    )

    inputs = [1.0, 0.0, 1.0, 0.0]
    targets = [0.0, 0.0, 1.0]
    legal_moves_mask = [0.0, 0.0, 1.0]

    initial_error = network.compute_error(
        inputs,
        targets,
        legal_moves_mask,
    )

    for _ in range(300):
        network.train_one(
            inputs,
            targets,
            legal_moves_mask,
        )

    final_error = network.compute_error(
        inputs,
        targets,
        legal_moves_mask,
    )
    final_predictions = network.predict(inputs)

    assert_true(final_error < initial_error)
    assert_true(final_predictions[2] > 0.8)


def test_network_serialization_preserves_predictions():
    network = SimpleNeuralNetwork(
        input_size=4,
        hidden_size=5,
        output_size=3,
        learning_rate=0.1,
        seed=2,
    )

    inputs = [0.0, 1.0, 0.0, 1.0]
    predictions_before = network.predict(inputs)

    data = network.to_dict()
    loaded_network = SimpleNeuralNetwork.from_dict(data)

    predictions_after = loaded_network.predict(inputs)

    assert_equal(len(predictions_before), len(predictions_after))

    for index in range(len(predictions_before)):
        difference = abs(predictions_before[index] - predictions_after[index])
        assert_true(difference < 0.000000001)


TESTS = [
    ("Le réseau prédit un score par sortie", test_network_predicts_one_score_per_output),
    ("L'erreur masquée ignore les coups illégaux", test_masked_error_ignores_illegal_outputs),
    ("L'entraînement réduit l'erreur sur un exemple", test_network_training_reduces_error_on_one_example),
    ("La sérialisation conserve les prédictions", test_network_serialization_preserves_predictions),
]