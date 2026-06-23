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
):
    """Entraîne un réseau sur un dataset déjà encodé.

    Cette fonction ne sauvegarde rien.
    Elle sert à vérifier que le réseau peut apprendre à approximer les scores
    produits par le professeur Monte-Carlo.

    Elle retourne :
    - le réseau entraîné ;
    - le modèle sérialisable ;
    - l'erreur avant entraînement ;
    - l'erreur après entraînement.
    """

    examples = get_examples_from_encoded_dataset(encoded_dataset)

    network = create_neural_network_from_encoded_dataset(
        encoded_dataset=encoded_dataset,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
    )

    initial_error = compute_average_error_on_encoded_examples(
        network,
        examples,
    )

    for epoch in range(epochs):
        for example in examples:
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
        "initial_error": initial_error,
        "final_error": final_error,
    }