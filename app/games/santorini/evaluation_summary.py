def summarize_o_results(results):
    total = results.get("X", 0) + results.get("O", 0) + results.get("draw", 0)

    if total == 0:
        efficiency = 0.0
    else:
        efficiency = ((results.get("O", 0) + 0.5 * results.get("draw", 0)) / total) * 100

    return {
        "total_games": total,
        "wins_x": results.get("X", 0),
        "wins_o": results.get("O", 0),
        "draws": results.get("draw", 0),
        "efficiency": efficiency,
    }


def format_o_evaluation_summary(title, summary):
    lines = []
    lines.append(title)
    lines.append("Parties jouées : " + str(summary["total_games"]))
    lines.append("Victoires de X : " + str(summary["wins_x"]))
    lines.append("Victoires de O : " + str(summary["wins_o"]))
    lines.append("Parties non terminées : " + str(summary["draws"]))
    lines.append("Score d'efficacité O : " + str(round(summary["efficiency"], 2)) + " %")
    return "\n".join(lines)


def format_o_evaluation_comparison(neural_summary, baseline_summary):
    delta = neural_summary["efficiency"] - baseline_summary["efficiency"]
    lines = []
    lines.append(format_o_evaluation_summary("Résumé évaluation neuronale Santorini", neural_summary))
    lines.append("")
    lines.append(format_o_evaluation_summary("Référence random O contre random X", baseline_summary))
    lines.append("")
    lines.append("Écart modèle - random : " + _format_delta(delta) + " points")
    return "\n".join(lines)


def _format_delta(value):
    rounded = round(value, 2)
    if rounded > 0:
        return "+" + str(rounded)
    return str(rounded)
