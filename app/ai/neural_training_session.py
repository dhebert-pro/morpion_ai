import random

from app.ai.neural_network import SimpleNeuralNetwork
from app.utils.progress import print_progress


def get_examples_from_encoded_dataset(encoded_dataset):
    return encoded_dataset.get("examples", [])


def create_neural_network_from_encoded_dataset(
    encoded_dataset,
    hidden_size,
    learning_rate,
    seed=0,
):
    input_size = encoded_dataset["input_size"]
    output_size = encoded_dataset["output_size"]

    return SimpleNeuralNetwork(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        learning_rate=learning_rate,
        seed=seed,
    )


def create_neural_network_from_model_data(
    model_data,
    learning_rate=None,
):
    network = SimpleNeuralNetwork.from_dict(model_data)

    if learning_rate is not None:
        network.learning_rate = learning_rate

    return network


def validate_network_matches_encoded_dataset(
    network,
    encoded_dataset,
    expected_hidden_size=None,
):
    if network.input_size != encoded_dataset["input_size"]:
        raise ValueError(
            "Le modèle neuronal a une taille d'entrée incompatible : "
            + str(network.input_size)
            + " au lieu de "
            + str(encoded_dataset["input_size"])
        )

    if network.output_size != encoded_dataset["output_size"]:
        raise ValueError(
            "Le modèle neuronal a une taille de sortie incompatible : "
            + str(network.output_size)
            + " au lieu de "
            + str(encoded_dataset["output_size"])
        )

    if expected_hidden_size is not None and network.hidden_size != expected_hidden_size:
        raise ValueError(
            "Le modèle neuronal a une taille de couche cachée incompatible : "
            + str(network.hidden_size)
            + " au lieu de "
            + str(expected_hidden_size)
        )


def create_or_load_neural_network_for_training(
    encoded_dataset,
    hidden_size,
    learning_rate,
    seed=0,
    initial_model_data=None,
):
    if initial_model_data is None:
        return create_neural_network_from_encoded_dataset(
            encoded_dataset=encoded_dataset,
            hidden_size=hidden_size,
            learning_rate=learning_rate,
            seed=seed,
        )

    network = create_neural_network_from_model_data(
        model_data=initial_model_data,
        learning_rate=learning_rate,
    )

    validate_network_matches_encoded_dataset(
        network=network,
        encoded_dataset=encoded_dataset,
        expected_hidden_size=hidden_size,
    )

    return network


def compute_average_error_on_encoded_examples(network, encoded_examples):
    if len(encoded_examples) == 0:
        return 0.0

    total_error = 0.0

    for example in encoded_examples:
        total_error += network.compute_error(
            example["inputs"],
            example["targets"],
            example["legal_moves_mask"],
        )

    return total_error / len(encoded_examples)


def train_network_on_encoded_dataset(
    encoded_dataset,
    hidden_size,
    epochs,
    learning_rate,
    show_progress=False,
    seed=0,
    initial_model_data=None,
):
    """Entraîne un réseau sur un dataset déjà encodé.

    Cette fonction ne sauvegarde rien.

    Si initial_model_data est None :
    - un nouveau réseau est créé ;
    - l'entraînement part de zéro.

    Si initial_model_data est fourni :
    - les poids existants sont rechargés ;
    - l'entraînement continue à partir de ce modèle ;
    - les dimensions doivent correspondre au dataset et à hidden_size.
    """

    examples = get_examples_from_encoded_dataset(encoded_dataset)

    network = create_or_load_neural_network_for_training(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
        initial_model_data=initial_model_data,
    )

    initial_error = compute_average_error_on_encoded_examples(
        network,
        examples,
    )

    training_rng = random.Random(seed + 10)

    for epoch in range(epochs):
        epoch_examples = list(examples)
        training_rng.shuffle(epoch_examples)

        for example in epoch_examples:
            network.train_one(
                example["inputs"],
                example["targets"],
                example["legal_moves_mask"],
            )

        if show_progress:
            print_progress("Entraînement réseau   ", epoch + 1, epochs)

    final_error = compute_average_error_on_encoded_examples(
        network,
        examples,
    )

    return {
        "network": network,
        "model_data": network.to_dict(),
        "examples_count": len(examples),
        "input_size": encoded_dataset["input_size"],
        "hidden_size": hidden_size,
        "output_size": encoded_dataset["output_size"],
        "epochs": epochs,
        "learning_rate": learning_rate,
        "started_from_existing_model": initial_model_data is not None,
        "initial_error": initial_error,
        "final_error": final_error,
    }