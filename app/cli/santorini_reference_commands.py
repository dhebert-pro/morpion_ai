import sys

from app.ai.neural_network import SimpleNeuralNetwork
from app.config import (
    SANTORINI_DATASET_SEED,
    SANTORINI_NEURAL_EVALUATION_GAMES_COUNT,
    SANTORINI_NEURAL_MODEL_FILE,
    SANTORINI_REFERENCE_EVALUATION_OPPONENTS,
)
from app.games.santorini.evaluation_summary import (
    format_o_evaluation_summary,
    summarize_o_results,
)
from app.games.santorini.neural_training import (
    get_santorini_model_data,
    load_santorini_model_package,
)
from app.games.santorini.reference_evaluation import (
    evaluate_santorini_neural_vs_references_paired,
)


def run_evaluate_santorini_reference_command():
    games_count = _read_int_arg(2, SANTORINI_NEURAL_EVALUATION_GAMES_COUNT)
    package = load_santorini_model_package(SANTORINI_NEURAL_MODEL_FILE)
    model_data = get_santorini_model_data(package)

    print("Évaluation Santorini contre adversaires de référence")
    print("Fichier : " + str(SANTORINI_NEURAL_MODEL_FILE))
    print("Parties par adversaire : " + str(games_count))
    print("Comparaison appairée : mêmes placements de départ")
    print()

    if not model_data:
        print("Aucun modèle Santorini valide trouvé.")
        return

    network = SimpleNeuralNetwork.from_dict(model_data)
    evaluations = evaluate_santorini_neural_vs_references_paired(
        network=network,
        opponent_names=SANTORINI_REFERENCE_EVALUATION_OPPONENTS,
        games_count=games_count,
        seed=SANTORINI_DATASET_SEED,
    )

    print(_format_reference_evaluations(evaluations))


def _format_reference_evaluations(evaluations):
    lines = []

    for evaluation in evaluations:
        opponent = evaluation["opponent"]
        neural_summary = summarize_o_results(evaluation["neural_results"])
        baseline_summary = summarize_o_results(evaluation["baseline_results"])
        delta = neural_summary["efficiency"] - baseline_summary["efficiency"]
        lines.append("Adversaire X : " + opponent)
        lines.append(format_o_evaluation_summary("Modèle O", neural_summary))
        lines.append("")
        lines.append(format_o_evaluation_summary("Random O", baseline_summary))
        lines.append("Écart modèle - random : " + _format_delta(delta) + " points")
        lines.append("")

    return "\n".join(lines).rstrip()


def _format_delta(value):
    rounded = round(value, 2)
    if rounded > 0:
        return "+" + str(rounded)
    return str(rounded)


def _read_int_arg(position, default_value):
    if len(sys.argv) <= position:
        return default_value

    try:
        value = int(sys.argv[position])
    except ValueError:
        return default_value

    return max(1, value)
