import random

from app.games.santorini.rules import get_legal_actions, is_cell_free
from app.games.santorini.coordinates import CELL_COUNT


def create_rng(seed=None):
    return random.Random(seed)


def get_free_placement_cells(game):
    return [cell for cell in range(CELL_COUNT) if is_cell_free(game, cell)]


def choose_random_placement(game, rng=None):
    generator = rng or random
    cells = get_free_placement_cells(game)

    if not cells:
        return None

    return generator.choice(cells)


def choose_random_action(game, rng=None):
    generator = rng or random
    actions = get_legal_actions(game)

    if not actions:
        return None

    return generator.choice(actions)
