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
            "worst_efficiency_name": None,
            "worst_survival_rate": None,
            "worst_survival_name": None,
        }

    worst_efficiency = reference_evaluations[0]
    worst_survival = reference_evaluations[0]

    for evaluation in reference_evaluations[1:]:
        if evaluation["efficiency"] < worst_efficiency["efficiency"]:
            worst_efficiency = evaluation

        if evaluation["survival_rate"] < worst_survival["survival_rate"]:
            worst_survival = evaluation

    return {
        "worst_efficiency": worst_efficiency["efficiency"],
        "worst_efficiency_name": worst_efficiency["reference_name"],
        "worst_survival_rate": worst_survival["survival_rate"],
        "worst_survival_name": worst_survival["reference_name"],
    }


def get_reference_selection_survival(checkpoint):
    survival_rate = checkpoint.get("reference_worst_survival_rate")

    if survival_rate is None:
        return 100.0

    return survival_rate


def get_reference_selection_efficiency(checkpoint):
    reference_efficiency = checkpoint.get("reference_worst_efficiency")

    if reference_efficiency is None:
        return checkpoint["evaluation_efficiency"]

    return reference_efficiency


def format_reference_summary(checkpoint):
    efficiency = checkpoint.get("reference_worst_efficiency")

    if efficiency is None:
        return "-"

    survival = checkpoint.get("reference_worst_survival_rate")
    efficiency_name = checkpoint.get("reference_worst_efficiency_name")
    survival_name = checkpoint.get("reference_worst_survival_name")

    if efficiency_name is None:
        efficiency_name = checkpoint.get("reference_worst_name")

    if survival is None:
        return str(round(efficiency, 2)) + " %"

    return (
        str(round(survival, 2))
        + " % surv. "
        + str(survival_name)
        + " / "
        + str(round(efficiency, 2))
        + " % "
        + str(efficiency_name)
    )
