from app.games.santorini.action_format import format_action
from app.games.santorini.action_index import ACTION_OUTPUT_SIZE
from app.games.santorini.coordinates import index_to_cell
from app.games.santorini.encoding import (
    SANTORINI_INPUT_PLANES,
    SANTORINI_INPUT_SIZE,
    encode_santorini_state,
    get_input_plane_slice,
)
from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.rules import create_new_game, place_worker


def create_sample_game():
    game = create_new_game()

    for cell in [0, 4, 20, 24]:
        place_worker(game, cell)

    return game


def format_santorini_network_report(game=None, limit=20):
    selected_game = game or create_sample_game()
    vector = encode_santorini_state(selected_game, selected_game["current_player"])
    actions = get_indexed_legal_actions(selected_game)

    lines = []
    lines.append("Inspection entrée/sortie Santorini")
    lines.append("Joueur encodé : " + str(selected_game["current_player"]))
    lines.append("Taille entrée : " + str(len(vector)))
    lines.append("Taille sortie : " + str(ACTION_OUTPUT_SIZE))
    lines.append("")
    lines.append("Plans d'entrée :")

    for plane_name in SANTORINI_INPUT_PLANES:
        start, end = get_input_plane_slice(plane_name)
        lines.append("- " + plane_name + " : indices " + str(start) + " à " + str(end - 1))

    lines.append("")
    lines.append("Coups légaux indexés : " + str(len(actions)))

    for action in actions[:limit]:
        lines.append(_format_indexed_action(action))

    if len(actions) > limit:
        lines.append("... " + str(len(actions) - limit) + " coups non affichés")

    if len(vector) != SANTORINI_INPUT_SIZE:
        lines.append("ATTENTION : taille d'entrée inattendue.")

    return "\n".join(lines)


def _format_indexed_action(action):
    build_text = "aucune"

    if action.get("build") is not None:
        build_text = index_to_cell(action["build"])

    return (
        "- sortie "
        + str(action["output_index"])
        + " : "
        + format_action(action)
        + " | déplacement "
        + index_to_cell(action["from"])
        + "→"
        + index_to_cell(action["to"])
        + " | construction "
        + build_text
    )
