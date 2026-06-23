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


def create_tactical_probe(name, board, expected_move, description):
    return {
        "name": name,
        "board": board,
        "expected_move": expected_move,
        "description": description,
    }


def get_default_morpion_tactical_probes():
    """Retourne les positions tactiques de base pour le morpion.

    Ces positions servent à détecter les erreurs grossières :
    - rater une victoire immédiate ;
    - ne pas bloquer une victoire immédiate adverse.
    """

    return [
        create_tactical_probe(
            name="win_top_row",
            board=[
                "O", "O", None,
                "X", "X", None,
                None, None, None,
            ],
            expected_move=2,
            description="O doit gagner immédiatement sur la ligne du haut.",
        ),
        create_tactical_probe(
            name="win_diagonal",
            board=[
                "O", "X", None,
                "X", "O", None,
                None, None, None,
            ],
            expected_move=8,
            description="O doit gagner immédiatement sur la diagonale.",
        ),
        create_tactical_probe(
            name="block_top_row",
            board=[
                "X", "X", None,
                "O", None, None,
                None, "O", None,
            ],
            expected_move=2,
            description="O doit bloquer la victoire immédiate de X sur la ligne du haut.",
        ),
        create_tactical_probe(
            name="block_diagonal",
            board=[
                "X", "O", None,
                "O", "X", None,
                None, None, None,
            ],
            expected_move=8,
            description="O doit bloquer la victoire immédiate de X sur la diagonale.",
        ),
    ]


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

    expected_move = probe["expected_move"]

    return {
        "name": probe["name"],
        "description": probe["description"],
        "expected_move": expected_move,
        "chosen_move": chosen_move,
        "passed": chosen_move == expected_move,
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
            + str(result["expected_move"])
            + " | choisi : "
            + str(result["chosen_move"])
        )
        lines.append("     " + result["description"])

    return "\n".join(lines)