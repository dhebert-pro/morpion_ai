import subprocess
import sys

import app.ai.strategies as strategies

from app.config import (
    PROJECT_ROOT,
    MODEL_FILE,
    TEST_FILE,
    TRAINING_GAMES_COUNT,
    SIMULATIONS_PER_MOVE,
    EVALUATION_GAMES_COUNT,
    SHOW_PROGRESS_DURING_TRAINING,
)

from app.storage.model_storage import load_model, save_model

from app.ai.training import train_model

from app.ai.evaluation import (
    evaluate_model,
    print_evaluation_results,
)

from app.games.morpion.engine import play_turn

from app.games.morpion.rules import (
    create_new_game,
    parse_human_input,
)


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
        show_progress=SHOW_PROGRESS_DURING_TRAINING
    )
    save_model(strategies.TRAINED_MODEL)

    print("Entraînement terminé.")
    print("Nombre d'états appris :", len(strategies.TRAINED_MODEL))
    print("Modèle sauvegardé dans :", MODEL_FILE)
    print()

    print("Évaluation rapide du modèle entraîné...")
    results = evaluate_model(strategies.TRAINED_MODEL, games_count=EVALUATION_GAMES_COUNT)
    print_evaluation_results(results)


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

    results = evaluate_model(model, games_count=EVALUATION_GAMES_COUNT)
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
    print("  python main.py train      → entraîne l'IA, sauvegarde le modèle et affiche son score")
    print("  python main.py evaluate   → évalue le modèle entraîné")
    print("  python main.py play       → lance une partie avec le modèle sauvegardé")
    print("  python main.py test       → lance tous les tests")


def run_cli():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()

    if command == "train":
        run_training_command()
    elif command == "evaluate":
        run_evaluate_command()
    elif command == "play":
        run_play_command()
    elif command == "test":
        run_test_command()
    else:
        print("Commande inconnue :", command)
        print_help()