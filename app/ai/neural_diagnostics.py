from app.ai.neural_pipeline import (
    train_neural_model_in_memory,
    format_neural_training_summary,
)

from app.ai.neural_evaluation import (
    evaluate_neural_model,
    summarize_neural_evaluation_results,
    format_neural_evaluation_summary,
)

from app.games.morpion.adapter import MORPION_ADAPTER


def run_neural_diagnostic(
    training_games_count,
    simulations_per_move,
    max_examples,
    hidden_size,
    epochs,
    learning_rate,
    evaluation_games_count,
    show_progress=False,
    seed=0,
    game_adapter=MORPION_ADAPTER,
):
    """Lance un diagnostic complet du moteur neuronal.

    Cette fonction ne sauvegarde rien.
    Elle sert à vérifier rapidement si la chaîne IA complète fonctionne :

    1. création d'un dataset par professeur Monte-Carlo ;
    2. encodage numérique ;
    3. entraînement du réseau ;
    4. évaluation du réseau contre un adversaire aléatoire.
    """

    training_result = train_neural_model_in_memory(
        training_games_count=training_games_count,
        simulations_per_move=simulations_per_move,
        max_examples=max_examples,
        hidden_size=hidden_size,
        epochs=epochs,
        learning_rate=learning_rate,
        show_progress=show_progress,
        seed=seed,
        game_adapter=game_adapter,
    )

    model_data = training_result["model_data"]

    evaluation_results = evaluate_neural_model(
        model_data=model_data,
        games_count=evaluation_games_count,
        game_adapter=game_adapter,
    )

    evaluation_summary = summarize_neural_evaluation_results(
        evaluation_results,
        trained_player=game_adapter.trained_player,
        opponent_player=game_adapter.opponent_player,
    )

    return {
        "game": game_adapter.name,
        "training_summary": training_result["summary"],
        "evaluation_results": evaluation_results,
        "evaluation_summary": evaluation_summary,
        "model_data": model_data,
    }


def format_neural_diagnostic_report(diagnostic_result):
    """Formate le diagnostic neuronal en texte lisible."""

    training_text = format_neural_training_summary(
        diagnostic_result["training_summary"]
    )
    evaluation_text = format_neural_evaluation_summary(
        diagnostic_result["evaluation_summary"]
    )

    lines = []

    lines.append("Diagnostic IA neuronale")
    lines.append("")
    lines.append(training_text)
    lines.append("")
    lines.append(evaluation_text)

    return "\n".join(lines)