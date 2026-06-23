import subprocess
import sys

import app.ai.strategies as strategies

from app.config import (
    PROJECT_ROOT,
    MODEL_FILE,
    MOVE_SCORE_DATASET_FILE,
    MOVE_SCORE_DATASET_MAX_EXAMPLES,
    NEURAL_MODEL_FILE,
    TEST_FILE,
    TRAINING_GAMES_COUNT,
    SIMULATIONS_PER_MOVE,
    EVALUATION_GAMES_COUNT,
    SHOW_PROGRESS_DURING_TRAINING,
    NEURAL_TRAINING_GAMES_COUNT,
    NEURAL_SIMULATIONS_PER_MOVE,
    NEURAL_MAX_EXAMPLES,
    NEURAL_TACTICAL_REPEAT_COUNT,
    NEURAL_HIDDEN_SIZE,
    NEURAL_EPOCHS,
    NEURAL_LEARNING_RATE,
    NEURAL_EVALUATION_GAMES_COUNT,
    NEURAL_BENCHMARK_TRAINING_GAMES_COUNT,
    NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE,
    NEURAL_BENCHMARK_MAX_EXAMPLES,
    NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT,
    NEURAL_BENCHMARK_HIDDEN_SIZE,
    NEURAL_BENCHMARK_CHECKPOINTS_COUNT,
    NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT,
    NEURAL_BENCHMARK_LEARNING_RATE,
    NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT,
)

from app.storage.model_storage import load_model, save_model
from app.storage.json_storage import save_json

from app.ai.training import train_model

from app.ai.evaluation import (
    evaluate_model,
    print_evaluation_results,
)

from app.ai.training_dataset import (
    build_move_score_dataset,
    summarize_move_score_dataset,
)

from app.ai.neural_diagnostics import (
    run_neural_diagnostic,
    format_neural_diagnostic_report,
)

from app.ai.neural_model_service import (
    train_and_save_neural_model,
    train_and_save_neural_model_from_package,
    load_neural_model_package,
    get_model_data_from_package,
    evaluate_saved_neural_model_package,
)

from app.ai.neural_evaluation import (
    format_neural_evaluation_summary,
)

from app.ai.neural_benchmark import (
    run_neural_training_benchmark,
    format_neural_benchmark_report,
    create_model_package_from_benchmark_result,
)

from app.games.morpion.engine import play_turn

from app.games.morpion.rules import (
    create_new_game,
    parse_human_input,
)


def has_flag(flag_name):
    return flag_name in sys.argv[2:]


def run_training_command():
    print("Paramètres d'entraînement :")
    print("Parties simulées pour collecter les états :", TRAINING_GAMES_COUNT)
    print("Simulations par coup :", SIMULATIONS_PER_MOVE)
    print("Parties d'évaluation :", EVALUATION_GAMES_COUNT)
    print()

    print("Entraînement de l'IA...")
    strategies.TRAINED_MODEL = train_model(
        training_games_count=TRAINING_GAMES_COUNT,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
    )
    save_model(strategies.TRAINED_MODEL)

    print("Entraînement terminé.")
    print("Nombre d'états appris :", len(strategies.TRAINED_MODEL))
    print("Modèle sauvegardé dans :", MODEL_FILE)
    print()

    print("Évaluation rapide du modèle entraîné...")
    results = evaluate_model(
        strategies.TRAINED_MODEL,
        games_count=EVALUATION_GAMES_COUNT,
    )
    print_evaluation_results(results)


def run_build_dataset_command():
    print("Création du dataset d'apprentissage :")
    print("Parties simulées pour collecter les états :", TRAINING_GAMES_COUNT)
    print("Simulations par coup :", SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples :", MOVE_SCORE_DATASET_MAX_EXAMPLES)
    print()

    dataset = build_move_score_dataset(
        training_games_count=TRAINING_GAMES_COUNT,
        simulations_per_move=SIMULATIONS_PER_MOVE,
        max_examples=MOVE_SCORE_DATASET_MAX_EXAMPLES,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
    )

    save_json(dataset, MOVE_SCORE_DATASET_FILE)

    summary = summarize_move_score_dataset(dataset)

    print("Dataset créé.")
    print("Fichier sauvegardé dans :", MOVE_SCORE_DATASET_FILE)
    print()
    print("Résumé :")
    print("Jeu :", summary["game"])
    print("Exemples :", summary["examples_count"])
    print("Coups scorés :", summary["scored_moves_count"])
    print("Nombre moyen de coups légaux :", summary["average_legal_moves"])
    print("Score moyen du meilleur coup :", summary["average_best_score"])


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


def print_neural_training_parameters():
    print("Parties simulées pour collecter les états :", NEURAL_TRAINING_GAMES_COUNT)
    print("Simulations par coup :", NEURAL_SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples Monte-Carlo :", NEURAL_MAX_EXAMPLES)
    print("Répétitions des exemples tactiques :", NEURAL_TACTICAL_REPEAT_COUNT)
    print("Taille couche cachée :", NEURAL_HIDDEN_SIZE)
    print("Époques :", NEURAL_EPOCHS)
    print("Taux d'apprentissage :", NEURAL_LEARNING_RATE)


def print_neural_training_result(model_package):
    training_summary = model_package["training_summary"]

    print()
    print("Modèle neuronal sauvegardé dans :", NEURAL_MODEL_FILE)
    print("Reprise d'un modèle existant :", training_summary["started_from_existing_model"])
    print("Exemples Monte-Carlo :", training_summary["base_examples_count"])
    print("Exemples tactiques ajoutés :", training_summary["extra_examples_count"])
    print("Exemples totaux :", training_summary["examples_count"])
    print("Erreur initiale :", round(training_summary["initial_error"], 6))
    print("Erreur finale :", round(training_summary["final_error"], 6))
    print("Amélioration erreur :", round(training_summary["error_improvement"], 6))
    print()

    print("Évaluation rapide du modèle neuronal sauvegardé...")
    evaluation = evaluate_saved_neural_model_package(
        model_package,
        games_count=NEURAL_EVALUATION_GAMES_COUNT,
    )

    print(format_neural_evaluation_summary(evaluation["summary"]))


def run_neural_training_command():
    use_watch_mode = has_flag("--watch")

    if use_watch_mode:
        run_neural_training_with_watch_command(
            start_from_saved_model=True,
        )
        return

    print("Entraînement du modèle neuronal")
    print("Mode : continuer le modèle existant si disponible")
    print_neural_training_parameters()
    print()

    existing_model_package = load_neural_model_package(
        NEURAL_MODEL_FILE,
    )

    if existing_model_package:
        print("Modèle existant trouvé :", NEURAL_MODEL_FILE)
        print("L'entraînement va continuer depuis ce modèle.")
    else:
        print("Aucun modèle existant trouvé.")
        print("L'entraînement démarre de zéro.")

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
    use_watch_mode = has_flag("--watch")

    if use_watch_mode:
        run_neural_training_with_watch_command(
            start_from_saved_model=False,
        )
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


def run_neural_training_with_watch_command(start_from_saved_model):
    print("Entraînement surveillé du modèle neuronal")
    print("Mode : affichage de l'amélioration par paliers")
    print("Le modèle final sera sauvegardé.")
    print()

    initial_model_data = None

    if start_from_saved_model:
        existing_model_package = load_neural_model_package(
            NEURAL_MODEL_FILE,
        )
        initial_model_data = get_model_data_from_package(
            existing_model_package,
        )

        if initial_model_data:
            print("Modèle existant trouvé :", NEURAL_MODEL_FILE)
            print("L'entraînement surveillé va continuer depuis ce modèle.")
        else:
            print("Aucun modèle existant trouvé.")
            print("L'entraînement surveillé démarre de zéro.")
    else:
        print("Réinitialisation demandée.")
        print("L'entraînement surveillé repart de zéro.")

    print()

    benchmark_result = run_neural_training_benchmark(
        training_games_count=NEURAL_BENCHMARK_TRAINING_GAMES_COUNT,
        simulations_per_move=NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE,
        max_examples=NEURAL_BENCHMARK_MAX_EXAMPLES,
        tactical_repeat_count=NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT,
        hidden_size=NEURAL_BENCHMARK_HIDDEN_SIZE,
        checkpoints_count=NEURAL_BENCHMARK_CHECKPOINTS_COUNT,
        epochs_per_checkpoint=NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT,
        learning_rate=NEURAL_BENCHMARK_LEARNING_RATE,
        evaluation_games_count=NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        seed=0,
        initial_model_data=initial_model_data,
    )

    model_package = create_model_package_from_benchmark_result(
        benchmark_result,
    )

    save_json(
        model_package,
        NEURAL_MODEL_FILE,
    )

    print()
    print(format_neural_benchmark_report(benchmark_result))
    print()
    print("Modèle neuronal final sauvegardé dans :", NEURAL_MODEL_FILE)


def run_neural_benchmark_command(start_from_saved_model):
    print("Benchmark du modèle neuronal")
    print("Aucune sauvegarde ne sera effectuée.")
    print("Parties simulées pour collecter les états :", NEURAL_BENCHMARK_TRAINING_GAMES_COUNT)
    print("Simulations par coup :", NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE)
    print("Nombre maximal d'exemples Monte-Carlo :", NEURAL_BENCHMARK_MAX_EXAMPLES)
    print("Répétitions tactiques :", NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT)
    print("Taille couche cachée :", NEURAL_BENCHMARK_HIDDEN_SIZE)
    print("Paliers :", NEURAL_BENCHMARK_CHECKPOINTS_COUNT)
    print("Époques par palier :", NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT)
    print("Taux d'apprentissage :", NEURAL_BENCHMARK_LEARNING_RATE)
    print("Parties d'évaluation par palier :", NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT)
    print()

    initial_model_data = None

    if start_from_saved_model:
        existing_model_package = load_neural_model_package(
            NEURAL_MODEL_FILE,
        )
        initial_model_data = get_model_data_from_package(
            existing_model_package,
        )

        if initial_model_data:
            print("Modèle existant trouvé :", NEURAL_MODEL_FILE)
            print("Le benchmark mesure l'amélioration à partir de ce modèle.")
        else:
            print("Aucun modèle existant trouvé.")
            print("Le benchmark partira de zéro.")
    else:
        print("Mode : benchmark depuis zéro.")
        print("Le modèle sauvegardé est ignoré.")

    print()

    benchmark_result = run_neural_training_benchmark(
        training_games_count=NEURAL_BENCHMARK_TRAINING_GAMES_COUNT,
        simulations_per_move=NEURAL_BENCHMARK_SIMULATIONS_PER_MOVE,
        max_examples=NEURAL_BENCHMARK_MAX_EXAMPLES,
        tactical_repeat_count=NEURAL_BENCHMARK_TACTICAL_REPEAT_COUNT,
        hidden_size=NEURAL_BENCHMARK_HIDDEN_SIZE,
        checkpoints_count=NEURAL_BENCHMARK_CHECKPOINTS_COUNT,
        epochs_per_checkpoint=NEURAL_BENCHMARK_EPOCHS_PER_CHECKPOINT,
        learning_rate=NEURAL_BENCHMARK_LEARNING_RATE,
        evaluation_games_count=NEURAL_BENCHMARK_EVALUATION_GAMES_COUNT,
        show_progress=SHOW_PROGRESS_DURING_TRAINING,
        seed=0,
        initial_model_data=initial_model_data,
    )

    print()
    print(format_neural_benchmark_report(benchmark_result))


def run_neural_evaluate_command():
    model_package = load_neural_model_package(
        NEURAL_MODEL_FILE,
    )

    if not model_package:
        print("Aucun modèle neuronal sauvegardé trouvé.")
        print("Lance d'abord : python main.py train-neural")
        return

    print("Évaluation du modèle neuronal sauvegardé")
    print("Fichier :", NEURAL_MODEL_FILE)
    print("Parties d'évaluation :", NEURAL_EVALUATION_GAMES_COUNT)
    print()

    evaluation = evaluate_saved_neural_model_package(
        model_package,
        games_count=NEURAL_EVALUATION_GAMES_COUNT,
    )

    print(format_neural_evaluation_summary(evaluation["summary"]))


def run_evaluate_command():
    model = load_model()

    if len(model) == 0:
        print("Aucun modèle entraîné trouvé.")
        print("Lance d'abord : python main.py train")
        return

    print("Évaluation du modèle...")
    print("Adversaire X : coups aléatoires")
    print("IA O : modèle entraîné")
    print()

    results = evaluate_model(
        model,
        games_count=EVALUATION_GAMES_COUNT,
    )
    print_evaluation_results(results)


def run_test_command():
    if not TEST_FILE.exists():
        print("Fichier de tests introuvable :", TEST_FILE)
        return

    completed_process = subprocess.run(
        [sys.executable, str(TEST_FILE)],
        cwd=PROJECT_ROOT,
    )

    if completed_process.returncode != 0:
        raise SystemExit(completed_process.returncode)


def run_play_command():
    strategies.TRAINED_MODEL = load_model()

    if len(strategies.TRAINED_MODEL) == 0:
        print("Aucun modèle entraîné trouvé.")
        print("L'IA utilisera seulement sa stratégie de secours.")
        print("Pour entraîner l'IA, lance : python main.py train")
        print()
    else:
        print("Modèle chargé.")
        print("Nombre d'états connus :", len(strategies.TRAINED_MODEL))
        print()

    game = create_new_game()

    print("Nouvelle partie créée")
    print("Tu joues X.")
    print("L'IA joue O.")
    print("Les cases sont numérotées de 0 à 8.")
    print("Tape q pour quitter.")

    while True:
        texte = input("Ton coup ? ")
        coup = parse_human_input(texte)

        if coup == "quit":
            print("Partie arrêtée.")
            break

        if coup is None:
            print("Entrée invalide. Tape un nombre entre 0 et 8, ou q pour quitter.")
            continue

        result = play_turn(game, coup)

        for message in result["messages"]:
            print(message)

        if result["finished"]:
            break


def print_help():
    print("Commandes disponibles :")
    print("  python main.py train                   → entraîne l'ancien modèle tabulaire")
    print("  python main.py build-dataset           → crée le dataset d'apprentissage Monte-Carlo")
    print("  python main.py neural-demo             → teste le moteur neuronal en mémoire")
    print("  python main.py train-neural            → continue l'entraînement neuronal si possible")
    print("  python main.py train-neural --watch    → continue et affiche l'amélioration par paliers")
    print("  python main.py reset-neural            → réinitialise et réentraîne le modèle neuronal")
    print("  python main.py reset-neural --watch    → réinitialise avec suivi par paliers")
    print("  python main.py neural-benchmark        → mesure l'amélioration sans sauvegarder")
    print("  python main.py neural-benchmark-reset  → mesure l'amélioration depuis zéro sans sauvegarder")
    print("  python main.py evaluate-neural         → évalue le modèle neuronal sauvegardé")
    print("  python main.py evaluate                → évalue l'ancien modèle tabulaire")
    print("  python main.py play                    → lance une partie avec l'ancien modèle sauvegardé")
    print("  python main.py test                    → lance tous les tests")


def run_cli():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "train":
        run_training_command()
    elif command == "build-dataset":
        run_build_dataset_command()
    elif command == "neural-demo":
        run_neural_demo_command()
    elif command == "train-neural":
        run_neural_training_command()
    elif command == "reset-neural":
        run_neural_reset_command()
    elif command == "neural-benchmark":
        run_neural_benchmark_command(start_from_saved_model=True)
    elif command == "neural-benchmark-reset":
        run_neural_benchmark_command(start_from_saved_model=False)
    elif command == "evaluate-neural":
        run_neural_evaluate_command()
    elif command == "evaluate":
        run_evaluate_command()
    elif command == "play":
        run_play_command()
    elif command == "test":
        run_test_command()
    else:
        print("Commande inconnue :", command)
        print_help()