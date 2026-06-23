from app.games.santorini.coordinates import (
    BOARD_SIZE,
    DIRECTIONS,
    index_to_row_col,
    is_inside,
    row_col_to_index,
)
from app.games.santorini.rules import is_valid_action

NO_BUILD_DIRECTION_INDEX = 8
ACTION_OUTPUT_SIZE = 2 * len(DIRECTIONS) * (len(DIRECTIONS) + 1)


def get_direction_index(origin, target):
    origin_row, origin_col = index_to_row_col(origin)
    target_row, target_col = index_to_row_col(target)
    delta = (target_col - origin_col, target_row - origin_row)

    if delta in DIRECTIONS:
        return DIRECTIONS.index(delta)

    return None


def neighbor_in_direction(origin, direction_index):
    if direction_index < 0 or direction_index >= len(DIRECTIONS):
        return None

    row, col = index_to_row_col(origin)
    delta_col, delta_row = DIRECTIONS[direction_index]
    next_row = row + delta_row
    next_col = col + delta_col

    if not is_inside(next_row, next_col):
        return None

    return row_col_to_index(next_row, next_col)


def get_worker_slot(game, player, from_cell):
    workers = game["workers"][player]

    for index, worker_cell in enumerate(workers):
        if worker_cell == from_cell:
            return index

    return None


def action_to_output_index(game, action, player=None):
    active_player = player or game["current_player"]
    worker_slot = get_worker_slot(game, active_player, action["from"])
    move_direction = get_direction_index(action["from"], action["to"])

    if worker_slot is None or move_direction is None:
        return None

    if action.get("build") is None:
        build_direction = NO_BUILD_DIRECTION_INDEX
    else:
        build_direction = get_direction_index(action["to"], action["build"])

    if build_direction is None:
        return None

    return encode_output_index(worker_slot, move_direction, build_direction)


def encode_output_index(worker_slot, move_direction, build_direction):
    move_count = len(DIRECTIONS)
    build_count = len(DIRECTIONS) + 1
    return worker_slot * move_count * build_count + move_direction * build_count + build_direction


def decode_output_index(index):
    if index < 0 or index >= ACTION_OUTPUT_SIZE:
        return None

    move_count = len(DIRECTIONS)
    build_count = len(DIRECTIONS) + 1
    worker_slot = index // (move_count * build_count)
    remaining = index % (move_count * build_count)
    move_direction = remaining // build_count
    build_direction = remaining % build_count

    return {
        "worker_slot": worker_slot,
        "move_direction": move_direction,
        "build_direction": build_direction,
    }


def output_index_to_action(game, index, player=None):
    components = decode_output_index(index)

    if components is None:
        return None

    active_player = player or game["current_player"]
    worker_cell = game["workers"][active_player][components["worker_slot"]]

    if worker_cell is None:
        return None

    to_cell = neighbor_in_direction(worker_cell, components["move_direction"])

    if to_cell is None:
        return None

    if components["build_direction"] == NO_BUILD_DIRECTION_INDEX:
        build_cell = None
    else:
        build_cell = neighbor_in_direction(to_cell, components["build_direction"])

    action = {"from": worker_cell, "to": to_cell, "build": build_cell}

    if not is_valid_action(game, action):
        return None

    return action
