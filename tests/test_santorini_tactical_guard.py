from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.indexed_actions import get_indexed_legal_actions
from app.games.santorini.neural_player import choose_santorini_neural_action_from_network
from app.games.santorini.rules import apply_action, copy_game, create_new_game
from app.games.santorini.tactical_guard import (
    count_immediate_winning_actions,
    filter_santorini_tactical_actions,
)


class FakeNetwork:
    def __init__(self, preferred_index):
        self.preferred_index = preferred_index

    def predict(self, _inputs):
        values = [0.0 for _ in range(144)]
        values[self.preferred_index] = 100.0
        return values


def _play_game(workers, current_player="O"):
    game = create_new_game()
    game["phase"] = "play"
    game["current_player"] = current_player
    game["workers"] = workers
    return game


def test_guard_keeps_immediate_winning_actions_first():
    game = _play_game({"X": [0, 24], "O": [6, 18]})
    game["heights"][6] = 2
    game["heights"][7] = 3
    actions = get_indexed_legal_actions(game, "O")

    filtered = filter_santorini_tactical_actions(game, actions)

    assert_true(len(filtered) > 0)
    assert_true(all(action["to"] == 7 for action in filtered))


def test_neural_choice_is_pure_by_default_before_prediction_score():
    game = _play_game({"X": [6, 24], "O": [12, 18]})
    game["heights"][6] = 2
    game["heights"][7] = 3
    actions = get_indexed_legal_actions(game, "O")
    bad_action = _find_action_giving_opponent_win(game, actions)

    chosen = choose_santorini_neural_action_from_network(
        game,
        FakeNetwork(bad_action["output_index"]),
    )

    assert_equal(chosen["output_index"], bad_action["output_index"])


def test_neural_choice_can_use_guard_in_explicit_diagnostic_mode():
    game = _play_game({"X": [6, 24], "O": [12, 18]})
    game["heights"][6] = 2
    game["heights"][7] = 3
    actions = get_indexed_legal_actions(game, "O")
    bad_action = _find_action_giving_opponent_win(game, actions)

    chosen = choose_santorini_neural_action_from_network(
        game,
        FakeNetwork(bad_action["output_index"]),
        use_tactical_guard=True,
    )
    copied = copy_game(game)
    apply_action(copied, chosen)

    assert_equal(count_immediate_winning_actions(copied, "X"), 0)


def _find_action_giving_opponent_win(game, actions):
    for action in actions:
        copied = copy_game(game)
        apply_action(copied, action)
        if count_immediate_winning_actions(copied, "X") > 0:
            return action

    raise AssertionError("Aucun mauvais coup trouvé pour le test")


TESTS = [
    ("Santorini garde tactique prend une victoire immédiate", test_guard_keeps_immediate_winning_actions_first),
    ("Santorini réseau pur suit le score avant la garde", test_neural_choice_is_pure_by_default_before_prediction_score),
    ("Santorini garde disponible seulement en mode explicite", test_neural_choice_can_use_guard_in_explicit_diagnostic_mode),
]
