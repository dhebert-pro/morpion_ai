import sys

from app.games.santorini.action_format import parse_human_input
from app.games.santorini.display import format_board, format_legal_inputs, format_status
from app.games.santorini.engine import play_input
from app.games.santorini.rules import create_new_game
from app.games.santorini.simulation import run_random_matches
from app.games.santorini.network_report import format_santorini_network_report


def print_santorini_help():
    print("Santorini de base : 2 joueurs, sans pouvoirs divins.")
    print("Placement : tape une case, par exemple A1.")
    print("Coup : origine-destination:construction, par exemple A1-B2:C2.")
    print("Coup gagnant vers le niveau 3 : origine-destination, sans construction.")
    print("Tape ? pour lister les coups légaux, q pour quitter.")
    print()


def run_play_santorini_command():
    game = create_new_game()
    print_santorini_help()

    while True:
        print(format_board(game))
        print(format_status(game))

        if game["phase"] == "finished":
            break

        text = input("Entrée ? ")
        parsed_input = parse_human_input(text)

        if parsed_input == "quit":
            print("Partie arrêtée.")
            break

        if parsed_input == "help":
            print(format_legal_inputs(game))
            print()
            continue

        if parsed_input is None:
            print("Entrée invalide.")
            print()
            continue

        result = play_input(game, parsed_input)

        for message in result["messages"]:
            print(message)

        print()


def run_simulate_santorini_random_command():
    count = read_match_count()
    result = run_random_matches(count=count, seed=0)

    print("Simulation Santorini random contre random")
    print("Parties :", result["games"])
    print("Victoires X :", result["wins_x"])
    print("Victoires O :", result["wins_o"])
    print("Non terminées :", result["unfinished"])
    print("Tours moyens :", round(result["average_turns"], 2))


def read_match_count():
    if len(sys.argv) < 3:
        return 100

    try:
        count = int(sys.argv[2])
    except ValueError:
        return 100

    return max(1, count)


def run_inspect_santorini_io_command():
    print(format_santorini_network_report())
