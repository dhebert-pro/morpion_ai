from app.ai.neural_reference_evaluation import (
    evaluate_neural_model_against_references,
)


def evaluate_reference_opponents_if_needed(
    model_data,
    games_count,
    game_adapter,
    seed,
    reference_names,
):
    if games_count is None or games_count <= 0:
        return []

    return evaluate_neural_model_against_references(
        model_data=model_data,
        games_count=games_count,
        game_adapter=game_adapter,
        seed=seed,
        reference_names=reference_names,
    )


def summarize_reference_evaluations(reference_evaluations):
    if len(reference_evaluations) == 0:
        return {
            "worst_efficiency": None,
            "worst_name": None,
        }

    worst_evaluation = reference_evaluations[0]

    for evaluation in reference_evaluations[1:]:
        if evaluation["efficiency"] < worst_evaluation["efficiency"]:
            worst_evaluation = evaluation

    return {
        "worst_efficiency": worst_evaluation["efficiency"],
        "worst_name": worst_evaluation["reference_name"],
    }


def get_reference_selection_efficiency(checkpoint):
    reference_efficiency = checkpoint.get("reference_worst_efficiency")

    if reference_efficiency is None:
        return checkpoint["evaluation_efficiency"]

    return reference_efficiency


def format_reference_efficiency(checkpoint):
    efficiency = checkpoint.get("reference_worst_efficiency")

    if efficiency is None:
        return "-"

    name = checkpoint.get("reference_worst_name")

    if name is None:
        return str(round(efficiency, 2)) + " %"

    return str(round(efficiency, 2)) + " % " + str(name)
