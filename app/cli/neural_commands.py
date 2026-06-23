from app.config import (
    NEURAL_MODEL_FILE,
    SHOW_PROGRESS_DURING_TRAINING,
    NEURAL_TRAINING_GAMES_COUNT,
    NEURAL_SIMULATIONS_PER_MOVE,
    NEURAL_MAX_EXAMPLES,
    NEURAL_TACTICAL_REPEAT_COUNT,
    NEURAL_HIDDEN_SIZE,
    NEURAL_EPOCHS,
    NEURAL_LEARNING_RATE,
)
from app.ai.neural_diagnostics import run_neural_diagnostic, format_neural_diagnostic_report
from app.ai.neural_model_service import (
    train_and_save_neural_model,
    train_and_save_neural_model_from_package,
    load_neural_model_package,
)
from app.cli.flags import has_flag
from app.cli.neural_display import (
    print_neural_training_parameters,
    print_neural_training_result,
)
from app.cli.neural_watch_commands import run_neural_training_with_watch_command

def run_neural_demo_command():
    print("Diagnostic rapide du moteur neuronal")
    print("Aucune sauvegarde ne sera effectuée.")
    print()

    diagnostic_result = run_neural_diagnostic(
        training_games_count=60,
        simulations_per_move=3,
        max_examples=20,
        hidden_size=12,
        epochs=120,
        learning_rate=0.2,
        evaluation_games_count=50,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        seed=0,
    )

    print()
    print(format_neural_diagnostic_report(diagnostic_result))


def run_neural_training_command():
    if has_flag("--watch"):
        run_neural_training_with_watch_command(start_from_saved_model=True)
        return

    print("Entraînement du modèle neuronal")
    print("Mode : continuer le modèle existant si disponible")
    print_neural_training_parameters()
    print()

    existing_model_package = load_neural_model_package(NEURAL_MODEL_FILE)
    initial_text = "Modèle existant trouvé :" if existing_model_package else "Aucun modèle existant trouvé."
    print(initial_text, NEURAL_MODEL_FILE if existing_model_package else "")
    print("L'entraînement va continuer depuis ce modèle." if existing_model_package else "L'entraînement démarre de zéro.")
    print()

    model_package = train_and_save_neural_model_from_package(
        file_path=NEURAL_MODEL_FILE,
        existing_model_package=existing_model_package,
        training_games_count=NEURAL_TRAINING_GAMES_COUNT,
        simulations_per_move=NEURAL_SIMULATIONS_PER_MOVE,
        max_examples=NEURAL_MAX_EXAMPLES,
        hidden_size=NEURAL_HIDDEN_SIZE,
        epochs=NEURAL_EPOCHS,
        learning_rate=NEURAL_LEARNING_RATE,
        tactical_repeat_count=NEURAL_TACTICAL_REPEAT_COUNT,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        seed=0,
    )
    print_neural_training_result(model_package)


def run_neural_reset_command():
    if has_flag("--watch"):
        run_neural_training_with_watch_command(start_from_saved_model=False)
        return

    print("Réinitialisation du modèle neuronal")
    print("Mode : repartir de zéro et écraser le modèle sauvegardé")
    print_neural_training_parameters()
    print()

    model_package = train_and_save_neural_model(
        file_path=NEURAL_MODEL_FILE,
        training_games_count=NEURAL_TRAINING_GAMES_COUNT,
        simulations_per_move=NEURAL_SIMULATIONS_PER_MOVE,
        max_examples=NEURAL_MAX_EXAMPLES,
        hidden_size=NEURAL_HIDDEN_SIZE,
        epochs=NEURAL_EPOCHS,
        learning_rate=NEURAL_LEARNING_RATE,
        tactical_repeat_count=NEURAL_TACTICAL_REPEAT_COUNT,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        seed=0,
        initial_model_data=None,
    )
    print_neural_training_result(model_package)
