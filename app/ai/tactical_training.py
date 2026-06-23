from app.ai.tactical_evaluation import (
    get_default_morpion_tactical_probes,
    create_game_from_board,
    get_expected_moves_from_probe,
)

from app.games.morpion.adapter import MORPION_ADAPTER


def create_move_score_example_from_tactical_probe(
    probe,
    game_adapter=MORPION_ADAPTER,
):
    """Transforme une position tactique en exemple d'apprentissage.

    Le coup attendu reçoit le score 1.0.
    Les autres coups légaux reçoivent le score 0.0.

    Cela permet de forcer le réseau à apprendre des réflexes simples :
    - gagner immédiatement ;
    - bloquer immédiatement.
    """

    game = create_game_from_board(
        probe["board"],
        game_adapter,
    )

    expected_moves = get_expected_moves_from_probe(probe)
    legal_moves = game_adapter.get_legal_moves(game)

    for expected_move in expected_moves:
        if expected_move not in legal_moves:
            raise ValueError(
                "Un coup tactique attendu n'est pas légal : "
                + str(expected_move)
                + " pour "
                + str(probe["name"])
            )

    move_scores = []

    for move in legal_moves:
        if move in expected_moves:
            score = 1.0
        else:
            score = 0.0

        move_scores.append({
            "move": move,
            "score": score,
        })

    return {
        "state_key": game_adapter.encode_game_state(game),
        "player": game_adapter.trained_player,
        "move_scores": move_scores,
        "best_move": expected_moves[0],
        "best_moves": expected_moves,
        "source": "tactical_probe",
        "probe_name": probe["name"],
        "description": probe["description"],
    }


def create_tactical_move_score_dataset(
    probes,
    repeat_count=1,
    game_adapter=MORPION_ADAPTER,
):
    """Crée un mini-dataset à partir de positions tactiques.

    repeat_count permet de répéter les exemples tactiques pour qu'ils pèsent
    davantage pendant l'apprentissage.
    """

    examples = []

    for _ in range(repeat_count):
        for probe in probes:
            example = create_move_score_example_from_tactical_probe(
                probe,
                game_adapter,
            )
            examples.append(example)

    return {
        "game": game_adapter.name,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
        "examples_count": len(examples),
        "source": "tactical_probes",
        "repeat_count": repeat_count,
        "examples": examples,
    }


def create_default_morpion_tactical_dataset(
    repeat_count=1,
    game_adapter=MORPION_ADAPTER,
):
    probes = get_default_morpion_tactical_probes()

    return create_tactical_move_score_dataset(
        probes=probes,
        repeat_count=repeat_count,
        game_adapter=game_adapter,
    )


def merge_move_score_datasets(
    base_dataset,
    extra_dataset,
):
    """Fusionne deux datasets de scoring.

    Les métadonnées principales viennent du dataset de base.
    Les exemples sont simplement concaténés.
    """

    base_examples = base_dataset.get("examples", [])
    extra_examples = extra_dataset.get("examples", [])

    merged_examples = []

    for example in base_examples:
        merged_examples.append(example)

    for example in extra_examples:
        merged_examples.append(example)

    return {
        "game": base_dataset.get("game"),
        "trained_player": base_dataset.get("trained_player"),
        "opponent_player": base_dataset.get("opponent_player"),
        "training_games_count": base_dataset.get("training_games_count"),
        "simulations_per_move": base_dataset.get("simulations_per_move"),
        "max_examples": base_dataset.get("max_examples"),
        "dataset_seed": base_dataset.get("dataset_seed"),
        "available_states_count": base_dataset.get("available_states_count", len(base_examples)),
        "base_examples_count": len(base_examples),
        "extra_examples_count": len(extra_examples),
        "examples_count": len(merged_examples),
        "examples": merged_examples,
    }