from app.games.morpion.adapter import MORPION_ADAPTER
from app.ai.morpion_line_features import encode_line_threats_as_neural_input


def encode_state_key_as_neural_input(
    state_key,
    trained_player,
    opponent_player,
):
    """Encode une clé d'état en vecteur numérique.

    On utilise :
    - un plan pour les cases occupées par le joueur entraîné ;
    - un plan pour les cases occupées par l'adversaire ;
    - des indicateurs de menaces immédiates sur les lignes du morpion.

    Exemple morpion :
    state_key = "OO.XX...."

    Entrées :
    - 9 valeurs indiquant où se trouve O ;
    - 9 valeurs indiquant où se trouve X.
    """

    vector = []

    for symbol in state_key:
        if symbol == trained_player:
            vector.append(1.0)
        else:
            vector.append(0.0)

    for symbol in state_key:
        if symbol == opponent_player:
            vector.append(1.0)
        else:
            vector.append(0.0)

    vector += encode_line_threats_as_neural_input(
        state_key,
        trained_player,
    )
    vector += encode_line_threats_as_neural_input(
        state_key,
        opponent_player,
    )

    return vector


def create_empty_output_vector(output_size):
    return [0.0 for _ in range(output_size)]


def validate_output_index(index, output_size):
    if index < 0 or index >= output_size:
        raise ValueError(
            "Index de coup invalide pour la sortie du réseau : "
            + str(index)
            + " avec output_size="
            + str(output_size)
        )


def encode_move_scores_as_target_vectors(
    move_scores,
    output_size,
    game_adapter=MORPION_ADAPTER,
):
    """Encode les scores des coups en deux vecteurs.

    targets :
    - contient le score connu pour chaque coup ;
    - les coups absents restent à 0.

    legal_moves_mask :
    - vaut 1 pour les coups scorés / légaux ;
    - vaut 0 pour les autres coups.

    Le mask servira plus tard à ne pas apprendre sur les coups illégaux.
    """

    targets = create_empty_output_vector(output_size)
    legal_moves_mask = create_empty_output_vector(output_size)

    for score_data in move_scores:
        move = score_data["move"]
        score = float(score_data["score"])
        output_index = game_adapter.move_to_index(move)

        validate_output_index(output_index, output_size)

        targets[output_index] = score
        legal_moves_mask[output_index] = 1.0

    return targets, legal_moves_mask


def encode_move_score_example(
    example,
    game_adapter=MORPION_ADAPTER,
):
    """Encode un exemple lisible du dataset en exemple numérique.

    Entrée attendue :
    {
        "state_key": "OO.XX....",
        "player": "O",
        "move_scores": [
            {"move": 2, "score": 1.0},
            {"move": 5, "score": 0.35},
        ],
        "best_move": 2,
    }

    Sortie produite :
    {
        "state_key": "OO.XX....",
        "inputs": [...],
        "targets": [...],
        "legal_moves_mask": [...],
        "best_move": 2,
        "best_move_index": 2,
        ...
    }
    """

    state_key = example["state_key"]
    trained_player = example.get("player", game_adapter.trained_player)
    opponent_player = game_adapter.opponent_player
    output_size = game_adapter.output_size

    inputs = encode_state_key_as_neural_input(
        state_key,
        trained_player,
        opponent_player,
    )

    targets, legal_moves_mask = encode_move_scores_as_target_vectors(
        example.get("move_scores", []),
        output_size,
        game_adapter,
    )

    legal_moves = []
    legal_move_indexes = []

    for score_data in example.get("move_scores", []):
        move = score_data["move"]
        output_index = game_adapter.move_to_index(move)

        validate_output_index(output_index, output_size)

        legal_moves.append(move)
        legal_move_indexes.append(output_index)

    best_move = example.get("best_move")

    if best_move is None:
        best_move_index = None
    else:
        best_move_index = game_adapter.move_to_index(best_move)
        validate_output_index(best_move_index, output_size)

    encoded_example = {
        "state_key": state_key,
        "inputs": inputs,
        "targets": targets,
        "legal_moves_mask": legal_moves_mask,
        "legal_moves": legal_moves,
        "legal_move_indexes": legal_move_indexes,
        "best_move": best_move,
        "best_move_index": best_move_index,
    }

    _copy_optional_debug_fields(example, encoded_example)

    return encoded_example


def _copy_optional_debug_fields(source_example, encoded_example):
    optional_keys = [
        "source",
        "probe_name",
        "description",
        "best_moves",
    ]

    for key in optional_keys:
        if key in source_example:
            encoded_example[key] = source_example[key]


def encode_move_score_dataset(
    dataset,
    game_adapter=MORPION_ADAPTER,
):
    """Encode tout le dataset de scoring en dataset numérique."""

    encoded_examples = []

    for example in dataset.get("examples", []):
        encoded_examples.append(
            encode_move_score_example(
                example,
                game_adapter,
            )
        )

    empty_game = game_adapter.create_new_game()
    empty_state_key = game_adapter.encode_game_state(empty_game)

    input_size = len(
        encode_state_key_as_neural_input(
            empty_state_key,
            game_adapter.trained_player,
            game_adapter.opponent_player,
        )
    )

    return {
        "game": dataset.get("game", game_adapter.name),
        "trained_player": dataset.get("trained_player", game_adapter.trained_player),
        "opponent_player": dataset.get("opponent_player", game_adapter.opponent_player),
        "input_size": input_size,
        "output_size": game_adapter.output_size,
        "source_examples_count": dataset.get("examples_count", len(dataset.get("examples", []))),
        "encoded_examples_count": len(encoded_examples),
        "examples": encoded_examples,
    }