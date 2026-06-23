from tests.test_helpers import assert_equal

from app.games.santorini.action_format import parse_human_input, format_action
from app.games.santorini.coordinates import cell_to_index


def c(text):
    return cell_to_index(text)


def test_parse_placement():
    assert_equal(parse_human_input("A1"), c("A1"))
    assert_equal(parse_human_input(" c3 "), c("C3"))


def test_parse_action_with_build():
    assert_equal(parse_human_input("A1-B2:C2"), {
        "from": c("A1"),
        "to": c("B2"),
        "build": c("C2"),
    })


def test_parse_winning_action_without_build():
    assert_equal(parse_human_input("A1-B2"), {
        "from": c("A1"),
        "to": c("B2"),
        "build": None,
    })


def test_parse_quit_and_invalid():
    assert_equal(parse_human_input("q"), "quit")
    assert_equal(parse_human_input("Z9"), None)
    assert_equal(parse_human_input("A1-B2:Z9"), None)


def test_format_action():
    assert_equal(format_action({"from": c("A1"), "to": c("B2"), "build": c("C2")}), "A1-B2:C2")
    assert_equal(format_action({"from": c("A1"), "to": c("B2"), "build": None}), "A1-B2")


TESTS = [
    ("Santorini lit un placement", test_parse_placement),
    ("Santorini lit un coup avec construction", test_parse_action_with_build),
    ("Santorini lit un coup gagnant sans construction", test_parse_winning_action_without_build),
    ("Santorini lit q et rejette les entrées invalides", test_parse_quit_and_invalid),
    ("Santorini affiche un coup", test_format_action),
]
