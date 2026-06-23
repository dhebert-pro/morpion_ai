from app.ai.neural_checkpoint import (
    format_checkpoint_line,
    get_checkpoint_table_header,
    get_best_checkpoint_from_benchmark_result,
)
from app.ai.neural_benchmark_diagnostics import build_benchmark_diagnostic_lines


def format_neural_benchmark_report(benchmark_result):
    lines = []
    best_checkpoint = get_best_checkpoint_from_benchmark_result(benchmark_result)

    lines.append("Benchmark entraînement neuronal")
    lines.append("Jeu : " + str(benchmark_result["game"]))
    lines.append("Départ depuis modèle existant : " + str(benchmark_result["started_from_existing_model"]))
    lines.append("États collectés disponibles : " + str(benchmark_result.get("available_states_count", 0)))
    lines.append("Exemples Monte-Carlo retenus : " + str(benchmark_result.get("base_examples_count", 0)))
    lines.append("Exemples tactiques : " + str(benchmark_result.get("tactical_examples_count", benchmark_result.get("extra_examples_count", 0))))
    reference_count = benchmark_result.get("reference_examples_count", 0)

    if reference_count > 0:
        lines.append("Exemples contre références : " + str(reference_count))

    lines.append("Exemples totaux : " + str(benchmark_result["examples_count"]))
    lines.append("Exemples apprentissage : " + str(benchmark_result.get("training_examples_count", benchmark_result["examples_count"])))
    lines.append("Exemples validation Monte-Carlo : " + str(benchmark_result.get("validation_examples_count", 0)))
    lines.append("Exemples tactiques/référence forcés en apprentissage : " + str(benchmark_result.get("always_train_examples_count", 0)))
    lines.append("Répétitions tactiques : " + str(benchmark_result["tactical_repeat_count"]))
    lines.append("Couche cachée : " + str(benchmark_result["hidden_size"]))
    lines.append(_format_checkpoints_configuration(benchmark_result))
    lines.append("Parties d'évaluation par palier : " + str(benchmark_result["evaluation_games_count"]))
    lines += _format_reference_training_configuration(benchmark_result)
    lines += _format_reference_configuration(benchmark_result)
    lines.append("Graine évaluation stable : " + str(benchmark_result.get("evaluation_seed")))
    lines.append("")
    lines.append(get_checkpoint_table_header())

    for checkpoint in benchmark_result.get("checkpoints", []):
        is_best = checkpoint["checkpoint_index"] == best_checkpoint["checkpoint_index"]
        lines.append(format_checkpoint_line(checkpoint, is_best))

    lines.append("")
    lines += _format_gains(benchmark_result)
    lines.append("")
    lines += _format_best_checkpoint(best_checkpoint)
    lines += _format_best_tactical_failures(best_checkpoint)
    lines += _format_stop_status(benchmark_result)
    lines.append("")
    lines += build_benchmark_diagnostic_lines(benchmark_result)

    return "\n".join(lines)


def _format_reference_training_configuration(benchmark_result):
    games_count = benchmark_result.get("reference_training_games_count", 0)

    if games_count is None or games_count <= 0:
        return []

    names = benchmark_result.get("reference_training_names", [])

    return [
        "Parties de référence pour dataset : " + str(games_count),
        "Exemples max contre références : "
        + str(benchmark_result.get("reference_training_max_examples", 0)),
        "Adversaires du dataset référence : "
        + ", ".join(str(name) for name in names),
    ]


def _format_reference_configuration(benchmark_result):
    games_count = benchmark_result.get("reference_evaluation_games_count", 0)

    if games_count is None or games_count <= 0:
        return []

    names = benchmark_result.get("reference_evaluation_names", [])

    return [
        "Parties référence par palier : " + str(games_count),
        "Adversaires référence : " + ", ".join(str(name) for name in names),
    ]


def _format_checkpoints_configuration(benchmark_result):
    return (
        "Paliers : "
        + str(benchmark_result["checkpoints_count"])
        + " × "
        + str(benchmark_result["epochs_per_checkpoint"])
        + " époques"
    )


def _format_gains(benchmark_result):
    return [
        "Gain erreur dataset : "
        + str(round(benchmark_result.get("training_error_improvement", 0.0), 6)),
        "Gain erreur validation : "
        + str(round(benchmark_result.get("validation_error_improvement", 0.0), 6)),
        "Gain efficacité : "
        + str(round(benchmark_result.get("evaluation_efficiency_improvement", 0.0), 2))
        + " points",
        "Gain tactique : "
        + str(round(benchmark_result.get("tactical_success_rate_improvement", 0.0), 2))
        + " points",
    ]


def _format_best_checkpoint(best_checkpoint):
    lines = [
        "Meilleur palier : "
        + str(best_checkpoint["checkpoint_index"])
        + " ("
        + str(best_checkpoint["total_epochs"])
        + " époques)",
        "Erreur train du meilleur modèle : "
        + str(round(best_checkpoint["training_error"], 6)),
        "Erreur valid du meilleur modèle : "
        + str(round(best_checkpoint.get("validation_error", 0.0), 6)),
        "Efficacité du meilleur modèle : "
        + str(round(best_checkpoint["evaluation_efficiency"], 2))
        + " %",
        "Tactique du meilleur modèle : "
        + str(round(best_checkpoint["tactical_success_rate"], 2))
        + " %",
    ]
    lines += _format_best_reference(best_checkpoint)
    return lines


def _format_best_reference(best_checkpoint):
    efficiency = best_checkpoint.get("reference_worst_efficiency")

    if efficiency is None:
        return []

    name = best_checkpoint.get("reference_worst_name", "?")

    return [
        "Référence du meilleur modèle : "
        + str(round(efficiency, 2))
        + " % au pire contre "
        + str(name)
    ]


def _format_best_tactical_failures(best_checkpoint):
    failed_results = best_checkpoint.get("tactical_failed_results", [])

    if len(failed_results) == 0:
        return ["Échecs tactiques du meilleur modèle : aucun."]

    lines = ["Échecs tactiques du meilleur modèle :"]

    for result in failed_results:
        lines.append(
            "- "
            + result["name"]
            + " | attendu : "
            + _format_expected_moves(result.get("expected_moves", []))
            + " | choisi : "
            + str(result.get("chosen_move"))
        )

    return lines


def _format_expected_moves(expected_moves):
    if len(expected_moves) == 0:
        return "?"

    return ", ".join(str(move) for move in expected_moves)


def _format_stop_status(benchmark_result):
    lines = []

    if benchmark_result.get("final_checkpoint_is_best", True):
        lines.append("Modèle retenu : dernier palier, car c'est aussi le meilleur.")
    else:
        lines.append("Modèle retenu : meilleur palier, pas le dernier.")

    if benchmark_result.get("stopped_early", False):
        lines.append("Arrêt anticipé : aucun meilleur modèle récent.")

    return lines
