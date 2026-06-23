from app.games.santorini.dataset import summarize_santorini_dataset


def format_santorini_dataset_report(dataset, examples_limit=3, moves_limit=5):
    summary = summarize_santorini_dataset(dataset)
    lines = []

    lines.append("Dataset Santorini Monte-Carlo")
    lines.append("Jeu : " + str(summary["game"]))
    lines.append("États disponibles : " + str(summary["available_states_count"]))
    lines.append("Exemples créés : " + str(summary["examples_count"]))
    lines.append("Coups scorés : " + str(summary["scored_moves_count"]))
    lines.append("Coups légaux moyens : " + str(summary["average_legal_moves"]))
    lines.append("Score moyen du meilleur coup : " + str(summary["average_best_score"]))
    lines.append("Écart moyen meilleur-pire coup : " + str(summary["average_score_spread"]))
    lines.append("Exemples réellement discriminants : " + str(summary["decisive_examples_count"]))
    lines.append("")
    lines.append("Exemples inspectables :")

    for index, example in enumerate(dataset.get("examples", [])[:examples_limit], start=1):
        lines += _format_example(index, example, moves_limit)

    return "\n".join(lines)


def _format_example(index, example, moves_limit):
    lines = []
    lines.append("Exemple " + str(index))
    lines.append("- coups légaux : " + str(example.get("legal_moves_count", 0)))
    lines.append("- meilleur coup : " + str(example.get("best_action")))
    lines.append("- sortie réseau : " + str(example.get("best_output_index")))
    lines.append("- score : " + str(example.get("best_score")))
    lines.append("- meilleurs coups scorés :")

    for move_score in _top_move_scores(example, moves_limit):
        lines.append(
            "  "
            + str(move_score["action"])
            + " | sortie "
            + str(move_score["output_index"])
            + " | score "
            + str(move_score["score"])
        )

    return lines


def _top_move_scores(example, moves_limit):
    scores = example.get("move_scores", [])
    return sorted(scores, key=lambda item: item["score"], reverse=True)[:moves_limit]
