from app.ai.neural_strategy import choose_neural_move
from app.games.morpion.adapter import MORPION_ADAPTER


def create_game_from_board(board, game_adapter=MORPION_ADAPTER):
    """Crée un état de jeu à partir d'un plateau.

    Pour le moment, les tests tactiques sont adaptés au morpion.
    Quand on ajoutera d'autres jeux, chacun pourra fournir ses propres
    positions tactiques.
    """

    game = game_adapter.create_new_game()
    game["board"] = board.copy()

    return game


def create_tactical_probe(
    name,
    board,
    expected_move=None,
    description="",
    expected_moves=None,
):
    moves = _normalize_expected_moves(expected_move, expected_moves)

    return {
        "name": name,
        "board": board,
        "expected_move": moves[0],
        "expected_moves": moves,
        "description": description,
    }


def _normalize_expected_moves(expected_move, expected_moves):
    if expected_moves is None:
        if expected_move is None:
            raise ValueError("Un test tactique doit avoir au moins un coup attendu.")

        return [expected_move]

    moves = list(expected_moves)

    if len(moves) == 0:
        raise ValueError("Un test tactique doit avoir au moins un coup attendu.")

    return moves


def get_expected_moves_from_probe(probe):
    if "expected_moves" in probe:
        moves = list(probe["expected_moves"])
    else:
        moves = [probe["expected_move"]]

    if len(moves) == 0:
        raise ValueError("Un test tactique doit avoir au moins un coup attendu.")

    return moves


def get_default_morpion_tactical_probes():
    """Retourne les positions tactiques de base pour le morpion."""

    from app.ai.morpion_tactical_probes import create_default_morpion_tactical_probes

    return create_default_morpion_tactical_probes()


def run_tactical_probe(
    probe,
    model_data,
    game_adapter=MORPION_ADAPTER,
):
    game = create_game_from_board(
        probe["board"],
        game_adapter,
    )

    chosen_move = choose_neural_move(
        game=game,
        model_data=model_data,
        game_adapter=game_adapter,
    )

    expected_moves = get_expected_moves_from_probe(probe)

    return {
        "name": probe["name"],
        "description": probe["description"],
        "expected_move": expected_moves[0],
        "expected_moves": expected_moves,
        "chosen_move": chosen_move,
        "passed": chosen_move in expected_moves,
    }


def run_tactical_evaluation(
    probes,
    model_data,
    game_adapter=MORPION_ADAPTER,
):
    results = []

    for probe in probes:
        result = run_tactical_probe(
            probe,
            model_data,
            game_adapter,
        )
        results.append(result)

    return results


def run_default_morpion_tactical_evaluation(
    model_data,
    game_adapter=MORPION_ADAPTER,
):
    probes = get_default_morpion_tactical_probes()

    return run_tactical_evaluation(
        probes,
        model_data,
        game_adapter,
    )


def summarize_tactical_evaluation(results):
    total_count = len(results)
    passed_count = 0

    for result in results:
        if result["passed"]:
            passed_count += 1

    if total_count == 0:
        success_rate = 0.0
    else:
        success_rate = passed_count / total_count * 100

    return {
        "total_count": total_count,
        "passed_count": passed_count,
        "failed_count": total_count - passed_count,
        "success_rate": success_rate,
    }


def format_tactical_evaluation_report(results):
    summary = summarize_tactical_evaluation(results)

    lines = []

    lines.append("Résumé évaluation tactique")
    lines.append("Tests réussis : " + str(summary["passed_count"]) + " / " + str(summary["total_count"]))
    lines.append("Taux de réussite : " + str(round(summary["success_rate"], 2)) + " %")
    lines.append("")

    for result in results:
        if result["passed"]:
            status = "OK"
        else:
            status = "NON"

        lines.append(
            status
            + " - "
            + result["name"]
            + " | attendu : "
            + _format_expected_moves(result)
            + " | choisi : "
            + str(result["chosen_move"])
        )
        lines.append("     " + result["description"])

    return "\n".join(lines)


def _format_expected_moves(result):
    expected_moves = result.get("expected_moves")

    if expected_moves is None:
        return str(result["expected_move"])

    if len(expected_moves) == 1:
        return str(expected_moves[0])

    return ", ".join(str(move) for move in expected_moves)
