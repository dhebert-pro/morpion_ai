import math
import random

from app.ai.neural_math import (
    sigmoid,
    sigmoid_derivative_from_output,
    tanh_derivative_from_output,
    compute_masked_mean_squared_error,
)


class SimpleNeuralNetwork:
    """Petit réseau de neurones sans dépendance externe.

    Il sert à approximer les scores de coups produits par le professeur
    Monte-Carlo.

    Structure :
    - entrée : vecteur numérique de l'état ;
    - couche cachée : tanh ;
    - sortie : sigmoid, donc scores entre 0 et 1.

    Le mask permet d'apprendre seulement sur les coups légaux d'un état.
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        learning_rate=0.08,
        seed=0,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        generator = random.Random(seed)

        self.input_hidden_weights = self._create_weight_matrix(
            input_size,
            hidden_size,
            generator,
        )
        self.hidden_biases = [0.0 for _ in range(hidden_size)]

        self.hidden_output_weights = self._create_weight_matrix(
            hidden_size,
            output_size,
            generator,
        )
        self.output_biases = [0.0 for _ in range(output_size)]

    def _create_weight_matrix(self, rows, columns, generator):
        matrix = []

        for _ in range(rows):
            row = []

            for _ in range(columns):
                row.append(generator.uniform(-0.25, 0.25))

            matrix.append(row)

        return matrix

    def predict(self, inputs):
        self._validate_inputs(inputs)

        hidden_outputs = self._compute_hidden_outputs(inputs)
        final_outputs = self._compute_final_outputs(hidden_outputs)

        return final_outputs

    def train_one(self, inputs, targets, legal_moves_mask):
        self._validate_inputs(inputs)
        self._validate_outputs(targets, "targets")
        self._validate_outputs(legal_moves_mask, "legal_moves_mask")

        hidden_outputs = self._compute_hidden_outputs(inputs)
        final_outputs = self._compute_final_outputs(hidden_outputs)

        output_deltas = []

        for output_index in range(self.output_size):
            if legal_moves_mask[output_index] == 0:
                output_deltas.append(0.0)
                continue

            error = targets[output_index] - final_outputs[output_index]
            delta = error * sigmoid_derivative_from_output(final_outputs[output_index])
            output_deltas.append(delta)

        hidden_deltas = []

        for hidden_index in range(self.hidden_size):
            error = 0.0

            for output_index in range(self.output_size):
                error += (
                    output_deltas[output_index]
                    * self.hidden_output_weights[hidden_index][output_index]
                )

            hidden_deltas.append(
                error * tanh_derivative_from_output(hidden_outputs[hidden_index])
            )

        for hidden_index in range(self.hidden_size):
            for output_index in range(self.output_size):
                self.hidden_output_weights[hidden_index][output_index] += (
                    self.learning_rate
                    * output_deltas[output_index]
                    * hidden_outputs[hidden_index]
                )

        for output_index in range(self.output_size):
            self.output_biases[output_index] += (
                self.learning_rate
                * output_deltas[output_index]
            )

        for input_index in range(self.input_size):
            for hidden_index in range(self.hidden_size):
                self.input_hidden_weights[input_index][hidden_index] += (
                    self.learning_rate
                    * hidden_deltas[hidden_index]
                    * inputs[input_index]
                )

        for hidden_index in range(self.hidden_size):
            self.hidden_biases[hidden_index] += (
                self.learning_rate
                * hidden_deltas[hidden_index]
            )

    def train_many(self, examples, epochs):
        for _ in range(epochs):
            for example in examples:
                self.train_one(
                    example["inputs"],
                    example["targets"],
                    example["legal_moves_mask"],
                )

    def compute_error(self, inputs, targets, legal_moves_mask):
        predictions = self.predict(inputs)

        return compute_masked_mean_squared_error(
            predictions,
            targets,
            legal_moves_mask,
        )

    def _compute_hidden_outputs(self, inputs):
        hidden_outputs = []

        for hidden_index in range(self.hidden_size):
            total = self.hidden_biases[hidden_index]

            for input_index in range(self.input_size):
                total += (
                    inputs[input_index]
                    * self.input_hidden_weights[input_index][hidden_index]
                )

            hidden_outputs.append(math.tanh(total))

        return hidden_outputs

    def _compute_final_outputs(self, hidden_outputs):
        final_outputs = []

        for output_index in range(self.output_size):
            total = self.output_biases[output_index]

            for hidden_index in range(self.hidden_size):
                total += (
                    hidden_outputs[hidden_index]
                    * self.hidden_output_weights[hidden_index][output_index]
                )

            final_outputs.append(sigmoid(total))

        return final_outputs

    def _validate_inputs(self, inputs):
        if len(inputs) != self.input_size:
            raise ValueError(
                "Taille d'entrée invalide : "
                + str(len(inputs))
                + " au lieu de "
                + str(self.input_size)
            )

    def _validate_outputs(self, outputs, name):
        if len(outputs) != self.output_size:
            raise ValueError(
                "Taille invalide pour "
                + name
                + " : "
                + str(len(outputs))
                + " au lieu de "
                + str(self.output_size)
            )

    def to_dict(self):
        return {
            "type": "simple_neural_network",
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "output_size": self.output_size,
            "learning_rate": self.learning_rate,
            "input_hidden_weights": self.input_hidden_weights,
            "hidden_biases": self.hidden_biases,
            "hidden_output_weights": self.hidden_output_weights,
            "output_biases": self.output_biases,
        }

    @classmethod
    def from_dict(cls, data):
        network = cls(
            input_size=data["input_size"],
            hidden_size=data["hidden_size"],
            output_size=data["output_size"],
            learning_rate=data.get("learning_rate", 0.08),
        )

        network.input_hidden_weights = data["input_hidden_weights"]
        network.hidden_biases = data["hidden_biases"]
        network.hidden_output_weights = data["hidden_output_weights"]
        network.output_biases = data["output_biases"]

        return network