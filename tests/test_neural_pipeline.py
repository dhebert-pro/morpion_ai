from tests.test_helpers import assert_equal, assert_true

from app.ai.neural_network import SimpleNeuralNetwork

from app.ai.neural_pipeline import (
    build_augmented_move_score_dataset,
    train_neural_model_in_memory,
    format_neural_training_summary,
)


def assert_contains(text, expected_part):
    if expected_part not in text:
        raise AssertionError(
            "Texte attendu introuvable : "
            + expected_part
            + "\nTexte obtenu :\n"
            + text
        )


def test_build_augmented_dataset_without_tactics_keeps_base_count():
    dataset = build_augmented_move_score_dataset(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        tactical_repeat_count=0,
        show_progress=False,
    )

    assert_equal(dataset["game"], "morpion")
    assert_true(dataset["examples_count"] > 0)
    assert_true(dataset["examples_count"] <= 5)
    assert_equal(dataset["base_examples_count"], dataset["examples_count"])
    assert_equal(dataset["extra_examples_count"], 0)
    assert_equal(dataset["tactical_repeat_count"], 0)


def test_build_augmented_dataset_with_tactics_adds_tactical_examples():
    dataset = build_augmented_move_score_dataset(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        tactical_repeat_count=2,
        show_progress=False,
    )

    assert_equal(dataset["game"], "morpion")
    assert_true(dataset["base_examples_count"] > 0)
    assert_true(dataset["base_examples_count"] <= 5)
    assert_equal(dataset["extra_examples_count"], 8)
    assert_equal(dataset["tactical_repeat_count"], 2)
    assert_equal(
        dataset["examples_count"],
        dataset["base_examples_count"] + dataset["extra_examples_count"],
    )


def test_train_neural_model_in_memory_runs_complete_pipeline():
    result = train_neural_model_in_memory(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=80,
        learning_rate=0.2,
        tactical_repeat_count=0,
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
    assert_equal(summary["tactical_repeat_count"], 0)
    assert_equal(summary["extra_examples_count"], 0)
    assert_equal(summary["input_size"], 18)
    assert_equal(summary["hidden_size"], 8)
    assert_equal(summary["output_size"], 9)
    assert_equal(summary["epochs"], 80)
    assert_equal(summary["learning_rate"], 0.2)

    assert_true(summary["final_error"] < summary["initial_error"])
    assert_true(summary["error_improvement"] > 0.0)


def test_train_neural_model_in_memory_can_use_tactical_examples():
    result = train_neural_model_in_memory(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=80,
        learning_rate=0.2,
        tactical_repeat_count=2,
        show_progress=False,
        seed=0,
    )

    summary = result["summary"]

    assert_equal(summary["tactical_repeat_count"], 2)
    assert_equal(summary["extra_examples_count"], 8)
    assert_equal(
        summary["examples_count"],
        summary["base_examples_count"] + summary["extra_examples_count"],
    )
    assert_true(summary["final_error"] < summary["initial_error"])


def test_train_neural_model_in_memory_returns_usable_model_data():
    result = train_neural_model_in_memory(
        training_games_count=20,
        simulations_per_move=1,
        max_examples=5,
        hidden_size=8,
        epochs=40,
        learning_rate=0.2,
        tactical_repeat_count=0,
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
        "tactical_repeat_count": 2,
        "base_examples_count": 5,
        "extra_examples_count": 8,
        "examples_count": 13,
        "scored_moves_count": 25,
        "average_legal_moves": 5.0,
        "average_best_score": 0.75,
        "input_size": 18,
        "hidden_size": 8,
        "output_size": 9,
        "epochs": 40,
        "learning_rate": 0.2,
        "started_from_existing_model": False,
        "initial_error": 0.25,
        "final_error": 0.10,
        "error_improvement": 0.15,
    }

    text = format_neural_training_summary(summary)

    assert_contains(text, "Résumé entraînement neuronal")
    assert_contains(text, "Jeu : morpion")
    assert_contains(text, "Exemples Monte-Carlo : 5")
    assert_contains(text, "Répétitions tactiques : 2")
    assert_contains(text, "Exemples tactiques ajoutés : 8")
    assert_contains(text, "Exemples totaux : 13")
    assert_contains(text, "Erreur initiale : 0.25")
    assert_contains(text, "Erreur finale : 0.1")
    assert_contains(text, "Amélioration erreur : 0.15")


TESTS = [
    ("Le dataset augmenté sans tactique garde seulement les exemples de base", test_build_augmented_dataset_without_tactics_keeps_base_count),
    ("Le dataset augmenté avec tactique ajoute les exemples tactiques", test_build_augmented_dataset_with_tactics_adds_tactical_examples),
    ("Le pipeline neuronal complet s'exécute en mémoire", test_train_neural_model_in_memory_runs_complete_pipeline),
    ("Le pipeline neuronal peut utiliser des exemples tactiques", test_train_neural_model_in_memory_can_use_tactical_examples),
    ("Le pipeline retourne un modèle neuronal utilisable", test_train_neural_model_in_memory_returns_usable_model_data),
    ("Le résumé d'entraînement neuronal contient les informations clés", test_format_neural_training_summary_contains_key_information),
]