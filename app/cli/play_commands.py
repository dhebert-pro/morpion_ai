import app.ai.strategies as strategies

from app.storage.model_storage import load_model
from app.games.morpion.engine import play_turn
from app.games.morpion.rules import create_new_game, parse_human_input


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
