from app.games.santorini.coordinates import CELL_COUNT
from app.games.santorini.rules import get_legal_actions, switch_player

SANTORINI_INPUT_PLANES = [
    "heights",
    "domes",
    "own_worker_1",
    "own_worker_2",
    "opponent_worker_1",
    "opponent_worker_2",
    "own_legal_destinations",
    "own_winning_destinations",
    "opponent_winning_destinations",
]

SANTORINI_INPUT_SIZE = CELL_COUNT * len(SANTORINI_INPUT_PLANES)


def encode_santorini_state(game, player=None):
    active_player = player or game["current_player"]
    opponent = switch_player(active_player)

    planes = []
    planes.append(_encode_heights(game))
    planes.append(_encode_domes(game))
    planes.append(_encode_worker_slot(game, active_player, 0))
    planes.append(_encode_worker_slot(game, active_player, 1))
    planes.append(_encode_worker_slot(game, opponent, 0))
    planes.append(_encode_worker_slot(game, opponent, 1))
    planes.append(_encode_destinations(game, active_player, winning_only=False))
    planes.append(_encode_destinations(game, active_player, winning_only=True))
    planes.append(_encode_destinations(game, opponent, winning_only=True))

    vector = []
    for plane in planes:
        vector += plane

    return vector


def encode_santorini_state_key(game):
    parts = []
    parts.append("h=" + "".join(str(height) for height in game["heights"]))
    parts.append("d=" + "".join("1" if dome else "0" for dome in game["domes"]))
    parts.append("x=" + _format_workers(game["workers"]["X"]))
    parts.append("o=" + _format_workers(game["workers"]["O"]))
    parts.append("p=" + str(game["current_player"]))
    parts.append("phase=" + str(game["phase"]))
    return "|".join(parts)


def _format_workers(workers):
    return ",".join("." if cell is None else str(cell) for cell in workers)


def _encode_heights(game):
    return [height / 3.0 for height in game["heights"]]


def _encode_domes(game):
    return [1.0 if dome else 0.0 for dome in game["domes"]]


def _encode_worker_slot(game, player, slot):
    vector = [0.0 for _ in range(CELL_COUNT)]
    workers = game["workers"].get(player, [])

    if slot >= len(workers):
        return vector

    cell = workers[slot]

    if cell is not None:
        vector[cell] = 1.0

    return vector


def _encode_destinations(game, player, winning_only=False):
    vector = [0.0 for _ in range(CELL_COUNT)]

    if game["phase"] != "play" or game["winner"] is not None:
        return vector

    for action in get_legal_actions(game, player):
        destination = action["to"]

        if winning_only and game["heights"][destination] != 3:
            continue

        vector[destination] = 1.0

    return vector


def get_input_plane_slice(plane_name):
    index = SANTORINI_INPUT_PLANES.index(plane_name)
    start = index * CELL_COUNT
    end = start + CELL_COUNT
    return start, end
