from app.games.santorini.coordinates import CELL_COUNT, get_neighbors

PLAYERS = ["X", "O"]
PLACEMENT_ORDER = ["X", "X", "O", "O"]


def create_new_game():
    return {
        "heights": [0 for _ in range(CELL_COUNT)],
        "domes": [False for _ in range(CELL_COUNT)],
        "workers": {"X": [None, None], "O": [None, None]},
        "current_player": "X",
        "phase": "placement",
        "placement_index": 0,
        "winner": None,
    }


def copy_game(game):
    return {
        "heights": game["heights"].copy(),
        "domes": game["domes"].copy(),
        "workers": {
            "X": game["workers"]["X"].copy(),
            "O": game["workers"]["O"].copy(),
        },
        "current_player": game["current_player"],
        "phase": game["phase"],
        "placement_index": game["placement_index"],
        "winner": game["winner"],
    }


def switch_player(player):
    if player == "X":
        return "O"
    return "X"


def get_worker_at(game, cell):
    for player in PLAYERS:
        for worker_index, worker_cell in enumerate(game["workers"][player]):
            if worker_cell == cell:
                return player, worker_index
    return None


def is_cell_free(game, cell):
    if cell is None or cell < 0 or cell >= CELL_COUNT:
        return False
    if game["domes"][cell]:
        return False
    return get_worker_at(game, cell) is None


def get_current_placement_player(game):
    if game["phase"] != "placement":
        return None
    return PLACEMENT_ORDER[game["placement_index"]]


def is_valid_placement(game, cell):
    if game["phase"] != "placement":
        return False
    return is_cell_free(game, cell)


def place_worker(game, cell):
    if not is_valid_placement(game, cell):
        return False

    player = get_current_placement_player(game)
    worker_slot = game["workers"][player].index(None)
    game["workers"][player][worker_slot] = cell
    game["placement_index"] += 1

    if game["placement_index"] >= len(PLACEMENT_ORDER):
        game["phase"] = "play"
        game["current_player"] = "X"
    else:
        game["current_player"] = get_current_placement_player(game)

    return True


def is_valid_move_destination(game, player, from_cell, to_cell):
    if from_cell not in game["workers"][player]:
        return False
    if to_cell not in get_neighbors(from_cell):
        return False
    if not is_cell_free(game, to_cell):
        return False
    return game["heights"][to_cell] <= game["heights"][from_cell] + 1


def can_build_at(game, worker_cell, build_cell):
    if build_cell not in get_neighbors(worker_cell):
        return False
    if build_cell < 0 or build_cell >= CELL_COUNT:
        return False
    if game["domes"][build_cell]:
        return False
    return get_worker_at(game, build_cell) is None


def move_worker_without_build(game, player, from_cell, to_cell):
    worker_index = game["workers"][player].index(from_cell)
    game["workers"][player][worker_index] = to_cell


def is_valid_action(game, action):
    if game["phase"] != "play" or game["winner"] is not None:
        return False

    player = game["current_player"]
    from_cell = action.get("from")
    to_cell = action.get("to")
    build_cell = action.get("build")

    if not is_valid_move_destination(game, player, from_cell, to_cell):
        return False

    if game["heights"][to_cell] == 3:
        return build_cell is None

    copied_game = copy_game(game)
    move_worker_without_build(copied_game, player, from_cell, to_cell)
    return build_cell is not None and can_build_at(copied_game, to_cell, build_cell)


def build_at(game, cell):
    if game["heights"][cell] == 3:
        game["domes"][cell] = True
    else:
        game["heights"][cell] += 1


def apply_action(game, action):
    if not is_valid_action(game, action):
        return False

    player = game["current_player"]
    move_worker_without_build(game, player, action["from"], action["to"])

    if game["heights"][action["to"]] == 3:
        game["winner"] = player
        game["phase"] = "finished"
        return True

    build_at(game, action["build"])
    next_player = switch_player(player)
    game["current_player"] = next_player

    if not has_any_legal_action(game, next_player):
        game["winner"] = player
        game["phase"] = "finished"

    return True


def get_legal_actions(game, player=None):
    if game["phase"] != "play" or game["winner"] is not None:
        return []

    active_player = player or game["current_player"]
    actions = []

    for from_cell in game["workers"][active_player]:
        if from_cell is None:
            continue

        for to_cell in get_neighbors(from_cell):
            if not is_valid_move_destination(game, active_player, from_cell, to_cell):
                continue

            if game["heights"][to_cell] == 3:
                actions.append({"from": from_cell, "to": to_cell, "build": None})
                continue

            copied_game = copy_game(game)
            move_worker_without_build(copied_game, active_player, from_cell, to_cell)

            for build_cell in get_neighbors(to_cell):
                if can_build_at(copied_game, to_cell, build_cell):
                    actions.append({"from": from_cell, "to": to_cell, "build": build_cell})

    return actions


def has_any_legal_action(game, player=None):
    return len(get_legal_actions(game, player)) > 0


def get_game_result(game):
    if game["winner"] is not None:
        return game["winner"]
    return "ongoing"


def get_score_for_o(result):
    if result == "O":
        return 1.0
    if result == "X":
        return 0.0
    raise ValueError("Résultat Santorini inconnu : " + str(result))
