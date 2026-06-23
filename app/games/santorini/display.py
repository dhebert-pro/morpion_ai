from app.games.santorini.coordinates import BOARD_SIZE, index_to_cell, row_col_to_index
from app.games.santorini.rules import get_worker_at


def format_cell(game, index):
    worker = get_worker_at(game, index)

    if worker is not None:
        player, worker_index = worker
        return player + str(worker_index + 1)

    if game["domes"][index]:
        return "D "

    return str(game["heights"][index]) + " "


def format_board(game):
    lines = ["    A  B  C  D  E"]

    for row in range(BOARD_SIZE):
        cells = []
        for col in range(BOARD_SIZE):
            index = row_col_to_index(row, col)
            cells.append(format_cell(game, index))
        lines.append(" " + str(row + 1) + "  " + " ".join(cells))

    return "\n".join(lines)


def format_status(game):
    if game["phase"] == "placement":
        return "Placement du joueur " + game["current_player"] + "."

    if game["phase"] == "finished":
        return "Partie terminée. Vainqueur : " + str(game["winner"])

    return "À jouer : " + game["current_player"] + "."


def format_cell_list(cells):
    return ", ".join(index_to_cell(cell) for cell in cells)


def format_legal_inputs(game, limit=30):
    if game["phase"] == "placement":
        from app.games.santorini.agents import get_free_placement_cells

        cells = get_free_placement_cells(game)
        return "Placements possibles : " + format_cell_list(cells)

    if game["phase"] != "play":
        return "Aucun coup légal : la partie est terminée."

    from app.games.santorini.action_format import format_action_list
    from app.games.santorini.rules import get_legal_actions

    actions = get_legal_actions(game)
    text = "Coups légaux : " + format_action_list(actions, limit=limit)

    if len(actions) > limit:
        text += " ... (" + str(len(actions)) + " coups au total)"

    return text
