from tests.test_helpers import assert_equal

from app.games.santorini.coordinates import cell_to_index, index_to_cell, get_neighbors


def test_cell_to_index():
    assert_equal(cell_to_index("A1"), 0)
    assert_equal(cell_to_index("E1"), 4)
    assert_equal(cell_to_index("A5"), 20)
    assert_equal(cell_to_index("E5"), 24)
    assert_equal(cell_to_index("c3"), 12)
    assert_equal(cell_to_index("Z9"), None)


def test_index_to_cell():
    assert_equal(index_to_cell(0), "A1")
    assert_equal(index_to_cell(12), "C3")
    assert_equal(index_to_cell(24), "E5")


def test_neighbors_corner_and_center():
    assert_equal(get_neighbors(0), [1, 5, 6])
    assert_equal(get_neighbors(12), [6, 7, 8, 11, 13, 16, 17, 18])


TESTS = [
    ("Santorini convertit les coordonnées en index", test_cell_to_index),
    ("Santorini convertit les index en coordonnées", test_index_to_cell),
    ("Santorini calcule les voisins", test_neighbors_corner_and_center),
]
