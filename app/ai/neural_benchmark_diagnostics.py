def build_benchmark_diagnostic_lines(benchmark_result):
    """Produit une lecture rapide des courbes du benchmark.

    Le but n'est pas de prouver mathématiquement la cause d'un mauvais modèle.
    On veut surtout distinguer les signaux utiles : surapprentissage probable,
    modèle final moins bon que le meilleur, ou métriques encore instables.
    """

    checkpoints = benchmark_result.get("checkpoints", [])

    if len(checkpoints) < 2:
        return ["Lecture diagnostic : données insuffisantes."]

    lines = ["Lecture diagnostic :"]
    lines.append(_format_training_validation_trend(benchmark_result))
    lines.append(_format_generalization_gap(benchmark_result))
    lines.append(_format_best_model_status(benchmark_result))
    lines.append(_format_evaluation_stability(benchmark_result))

    return lines


def _format_training_validation_trend(benchmark_result):
    training_gain = benchmark_result.get("training_error_improvement", 0.0)
    validation_gain = benchmark_result.get("validation_error_improvement", 0.0)

    if training_gain > 0.0 and validation_gain < 0.0:
        return (
            "- Signal d'overfitting : l'erreur train baisse, "
            "mais l'erreur validation monte."
        )

    if training_gain > 0.0 and validation_gain > 0.0:
        return "- Généralisation correcte : train et validation baissent ensemble."

    if training_gain <= 0.0 and validation_gain > 0.0:
        return "- Validation meilleure, mais apprentissage train peu convaincant."

    return "- Pas de progrès clair sur les erreurs train/validation."


def _format_generalization_gap(benchmark_result):
    final_training_error = benchmark_result.get("final_training_error", 0.0)
    final_validation_error = benchmark_result.get("final_validation_error", final_training_error)
    gap = final_validation_error - final_training_error

    if gap > 0.05:
        return "- Écart validation/train élevé : " + str(round(gap, 6))

    if gap > 0.0:
        return "- Écart validation/train modéré : " + str(round(gap, 6))

    return "- Pas d'écart validation/train inquiétant."


def _format_best_model_status(benchmark_result):
    if benchmark_result.get("final_checkpoint_is_best", True):
        return "- Le dernier palier est aussi le meilleur modèle retenu."

    best_index = benchmark_result.get("best_checkpoint_index", "?")
    return "- Le meilleur modèle était au palier " + str(best_index) + ", pas à la fin."


def _format_evaluation_stability(benchmark_result):
    checkpoints = benchmark_result.get("checkpoints", [])
    efficiencies = [checkpoint["evaluation_efficiency"] for checkpoint in checkpoints]
    tactical_rates = [checkpoint["tactical_success_rate"] for checkpoint in checkpoints]

    efficiency_spread = max(efficiencies) - min(efficiencies)
    tactical_spread = max(tactical_rates) - min(tactical_rates)

    return (
        "- Amplitude efficacité : "
        + str(round(efficiency_spread, 2))
        + " points ; amplitude tactique : "
        + str(round(tactical_spread, 2))
        + " points."
    )
