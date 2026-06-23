import random


def split_encoded_examples(
    encoded_examples,
    validation_ratio=0.0,
    seed=0,
):
    """Sépare les exemples en apprentissage et validation.

    La validation sert à repérer quand le réseau mémorise le dataset sans
    réellement mieux généraliser.
    """

    examples = list(encoded_examples)

    if len(examples) <= 1 or validation_ratio <= 0:
        return {
            "training_examples": examples,
            "validation_examples": [],
        }

    shuffled_examples = list(examples)
    generator = random.Random(seed)
    generator.shuffle(shuffled_examples)

    validation_count = int(round(len(shuffled_examples) * validation_ratio))
    validation_count = max(1, validation_count)
    validation_count = min(validation_count, len(shuffled_examples) - 1)

    validation_examples = shuffled_examples[:validation_count]
    training_examples = shuffled_examples[validation_count:]

    return {
        "training_examples": training_examples,
        "validation_examples": validation_examples,
    }


def get_effective_validation_error(checkpoint):
    return checkpoint.get(
        "validation_error",
        checkpoint.get("training_error", 0.0),
    )
