from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.agents import get_free_placement_cells, choose_random_action
from app.games.santorini.coordinates import cell_to_index
from app.games.santorini.rules import create_new_game, place_worker
from app.games.santorini.simulation import play_random_game, run_random_matches


def c(text):
    return cell_to_index(text)


def setup_game():
    game = create_new_game()
    for text in ["A1", "E1", "A5", "E5"]:
        place_worker(game, c(text))
    return game


def test_free_placement_cells_excludes_occupied_cells():
    game = create_new_game()
    place_worker(game, c("A1"))
    cells = get_free_placement_cells(game)
    assert_equal(c("A1") in cells, False)
    assert_equal(len(cells), 24)


def test_random_agent_returns_legal_action():
    game = setup_game()
    action = choose_random_action(game)
    assert_true(action is not None)


def test_random_game_finishes_or_hits_guard():
    result = play_random_game(seed=0, max_turns=200, keep_log=True)
    assert_true(result["winner"] in ["X", "O", "ongoing"])
    assert_true(result["turns"] <= 200)
    assert_true(len(result["log"]) >= 4)


def test_random_matches_are_counted():
    result = run_random_matches(count=5, seed=0)
    total = result["wins_x"] + result["wins_o"] + result["unfinished"]
    assert_equal(total, 5)
    assert_equal(result["games"], 5)


TESTS = [
    ("Santorini liste les placements libres", test_free_placement_cells_excludes_occupied_cells),
    ("Santorini agent aléatoire choisit un coup", test_random_agent_returns_legal_action),
    ("Santorini simule une partie aléatoire", test_random_game_finishes_or_hits_guard),
    ("Santorini compte des matchs aléatoires", test_random_matches_are_counted),
]
