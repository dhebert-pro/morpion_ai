import random


def split_encoded_examples(
    encoded_examples,
    validation_ratio=0.0,
    seed=0,
    always_train_sources=None,
):
    """Sépare les exemples en apprentissage et validation.

    Les doublons d'un même état restent dans le même bloc. C'est important
    pour les exemples répétés : sinon la validation verrait parfois exactement
    les mêmes positions que l'apprentissage.

    always_train_sources permet de garder certains exemples dans
    l'apprentissage. On l'utilise notamment pour les réflexes tactiques : ces
    positions sont des règles à enseigner au modèle, pas un échantillon à
    réserver pour la validation Monte-Carlo.
    """

    examples = list(encoded_examples)
    always_train_examples = _filter_always_train_examples(
        examples,
        always_train_sources,
    )
    splittable_examples = _filter_splittable_examples(
        examples,
        always_train_sources,
    )

    if len(splittable_examples) <= 1 or validation_ratio <= 0:
        return {
            "training_examples": splittable_examples + always_train_examples,
            "validation_examples": [],
            "always_train_examples_count": len(always_train_examples),
        }

    split_result = _split_splittable_examples(
        splittable_examples,
        validation_ratio,
        seed,
    )

    return {
        "training_examples": split_result["training_examples"] + always_train_examples,
        "validation_examples": split_result["validation_examples"],
        "always_train_examples_count": len(always_train_examples),
    }


def get_effective_validation_error(checkpoint):
    return checkpoint.get(
        "validation_error",
        checkpoint.get("training_error", 0.0),
    )


def _split_splittable_examples(examples, validation_ratio, seed):
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


def _filter_always_train_examples(examples, always_train_sources):
    if not always_train_sources:
        return []

    sources = set(always_train_sources)

    return [example for example in examples if example.get("source") in sources]


def _filter_splittable_examples(examples, always_train_sources):
    if not always_train_sources:
        return examples

    sources = set(always_train_sources)

    return [example for example in examples if example.get("source") not in sources]


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
