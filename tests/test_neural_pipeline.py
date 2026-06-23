from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_pipeline import (
    train_neural_model_in_memory,
    format_neural_training_summary,
)


def test_train_neural_model_in_memory_runs_complete_pipeline():
    result = train_neural_model_in_memory(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=80,
        learning_rate=0.2,
        show_progress=False,
        seed=0,
    )

    summary = result["summary"]

    assert_equal(result["game"], "morpion")
    assert_equal(result["trained_player"], "O")
    assert_equal(result["opponent_player"], "X")

    assert_true("raw_dataset" in result)
    assert_true("encoded_dataset" in result)
    assert_true("model_data" in result)
    assert_true("summary" in result)

    assert_true(summary["examples_count"] > 0)
    assert_true(summary["examples_count"] <= 5)
    assert_equal(summary["input_size"], 18)
    assert_equal(summary["hidden_size"], 8)
    assert_equal(summary["output_size"], 9)
    assert_equal(summary["epochs"], 80)
    assert_equal(summary["learning_rate"], 0.2)

    assert_true(summary["final_error"] < summary["initial_error"])
    assert_true(summary["error_improvement"] > 0.0)


def test_train_neural_model_in_memory_returns_usable_model_data():
    result = train_neural_model_in_memory(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=40,
        learning_rate=0.2,
        show_progress=False,
        seed=1,
    )

    model_data = result["model_data"]

    network = SimpleNeuralNetwork.from_dict(model_data)
    first_example = result["encoded_dataset"]["examples"][0]

    predictions = network.predict(first_example["inputs"])

    assert_equal(len(predictions), 9)

    for prediction in predictions:
        assert_true(0.0 <= prediction <= 1.0)


def test_format_neural_training_summary_contains_key_information():
    summary = {
        "game": "morpion",
        "training_games_count": 20,
        "simulations_per_move": 1,
        "max_examples": 5,
        "examples_count": 5,
        "scored_moves_count": 25,
        "average_legal_moves": 5.0,
        "average_best_score": 0.75,
        "input_size": 18,
        "hidden_size": 8,
        "output_size": 9,
        "epochs": 40,
        "learning_rate": 0.2,
        "initial_error": 0.25,
        "final_error": 0.10,
        "error_improvement": 0.15,
    }

    text = format_neural_training_summary(summary)

    assert_true("Résumé entraînement neuronal" in text)
    assert_true("Jeu : morpion" in text)
    assert_true("Exemples : 5" in text)
    assert_true("Erreur initiale : 0.25" in text)
    assert_true("Erreur finale : 0.1" in text)
    assert_true("Amélioration erreur : 0.15" in text)


TESTS = [
    ("Le pipeline neuronal complet s'exécute en mémoire", test_train_neural_model_in_memory_runs_complete_pipeline),
    ("Le pipeline retourne un modèle neuronal utilisable", test_train_neural_model_in_memory_returns_usable_model_data),
    ("Le résumé d'entraînement neuronal contient les informations clés", test_format_neural_training_summary_contains_key_information),
]