from tests.test_helpers import assert_equal

from app.utils.progress import build_progress_bar


def test_progress_bar_at_start():
    assert_equal(build_progress_bar(0, 100, width=10), "[----------] 0%")


def test_progress_bar_at_half():
    assert_equal(build_progress_bar(50, 100, width=10), "[#####-----] 50%")


def test_progress_bar_at_end():
    assert_equal(build_progress_bar(100, 100, width=10), "[##########] 100%")


def test_progress_bar_clamps_values():
    assert_equal(build_progress_bar(150, 100, width=10), "[##########] 100%")
    assert_equal(build_progress_bar(-10, 100, width=10), "[----------] 0%")


def test_progress_bar_with_zero_total():
    assert_equal(build_progress_bar(0, 0, width=10), "[----------] 0%")


TESTS = [
    ("Barre de progression au départ", test_progress_bar_at_start),
    ("Barre de progression à moitié", test_progress_bar_at_half),
    ("Barre de progression terminée", test_progress_bar_at_end),
    ("Barre de progression borne les valeurs extrêmes", test_progress_bar_clamps_values),
    ("Barre de progression avec total nul", test_progress_bar_with_zero_total),
]
