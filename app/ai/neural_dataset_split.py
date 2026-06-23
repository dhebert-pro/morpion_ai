import random


def split_encoded_examples(
    encoded_examples,
    validation_ratio=0.0,
    seed=0,
):
    """Sépare les exemples en apprentissage et validation.

    Les doublons d'un même état restent dans le même bloc. C'est important
    pour les exemples tactiques répétés : sinon la validation verrait parfois
    exactement les mêmes positions que l'apprentissage.
    """

    examples = list(encoded_examples)

    if len(examples) <= 1 or validation_ratio <= 0:
        return {
            "training_examples": examples,
            "validation_examples": [],
        }

    grouped_examples = _group_examples_by_state_key(examples)

    if len(grouped_examples) <= 1:
        return {
            "training_examples": examples,
            "validation_examples": [],
        }

    validation_groups = _select_validation_groups(
        grouped_examples,
        len(examples),
        validation_ratio,
        seed,
    )

    validation_examples = []
    training_examples = []

    for group_index, group_examples in enumerate(grouped_examples):
        if group_index in validation_groups:
            validation_examples += group_examples
        else:
            training_examples += group_examples

    return {
        "training_examples": training_examples,
        "validation_examples": validation_examples,
    }


def get_effective_validation_error(checkpoint):
    return checkpoint.get(
        "validation_error",
        checkpoint.get("training_error", 0.0),
    )


def _group_examples_by_state_key(examples):
    groups_by_key = {}

    for index, example in enumerate(examples):
        state_key = example.get("state_key", "__example_" + str(index))

        if state_key not in groups_by_key:
            groups_by_key[state_key] = []

        groups_by_key[state_key].append(example)

    return list(groups_by_key.values())


def _select_validation_groups(grouped_examples, examples_count, validation_ratio, seed):
    generator = random.Random(seed)
    group_indexes = list(range(len(grouped_examples)))
    generator.shuffle(group_indexes)

    target_validation_count = int(round(examples_count * validation_ratio))
    target_validation_count = max(1, target_validation_count)
    target_validation_count = min(target_validation_count, examples_count - 1)

    selected_groups = set()
    selected_examples_count = 0

    for group_index in group_indexes:
        group_size = len(grouped_examples[group_index])

        if selected_examples_count >= target_validation_count:
            break

        if examples_count - selected_examples_count - group_size <= 0:
            continue

        selected_groups.add(group_index)
        selected_examples_count += group_size

    return selected_groups
