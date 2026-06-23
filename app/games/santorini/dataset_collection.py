import random

from app.games.santorini.agents import choose_random_action, choose_random_placement
from app.games.santorini.encoding import encode_santorini_state_key
from app.games.santorini.rules import apply_action, copy_game, create_new_game, place_worker


TRAINED_PLAYER = "O"


def collect_santorini_training_states(games_count, max_turns=120, seed=0):
    rng = random.Random(seed)
    states = {}

    for _ in range(games_count):
        game = create_random_started_game(rng)
        turn_count = 0

        while game["phase"] == "play" and turn_count < max_turns:
            if game["current_player"] == TRAINED_PLAYER:
                state_key = encode_santorini_state_key(game)
                states[state_key] = copy_game(game)

            action = choose_random_action(game, rng)

            if action is None:
                _finish_blocked_game(game)
                break

            apply_action(game, action)
            turn_count += 1

    return states


def create_random_started_game(rng):
    game = create_new_game()

    while game["phase"] == "placement":
        cell = choose_random_placement(game, rng)

        if cell is None:
            break

        place_worker(game, cell)

    return game


def select_santorini_state_items(states, max_examples=None, seed=0):
    items = list(states.items())

    if max_examples is None or len(items) <= max_examples:
        return sorted(items, key=lambda item: item[0])

    groups = _group_by_total_height(items)
    rng = random.Random(seed)

    for group in groups.values():
        rng.shuffle(group)

    selected = _round_robin(groups, max_examples)
    rng.shuffle(selected)
    return selected


def _finish_blocked_game(game):
    current_player = game["current_player"]
    game["winner"] = "O" if current_player == "X" else "X"
    game["phase"] = "finished"


def _group_by_total_height(items):
    groups = {}

    for item in items:
        game = item[1]
        total_height = sum(game["heights"])
        bucket = min(total_height // 3, 8)

        if bucket not in groups:
            groups[bucket] = []

        groups[bucket].append(item)

    return groups


def _round_robin(groups, max_examples):
    selected = []
    keys = sorted(groups.keys())
    positions = {key: 0 for key in keys}

    while len(selected) < max_examples:
        added = False

        for key in keys:
            position = positions[key]
            group = groups[key]

            if position >= len(group):
                continue

            selected.append(group[position])
            positions[key] = position + 1
            added = True

            if len(selected) >= max_examples:
                break

        if not added:
            break

    return selected
