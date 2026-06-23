from tests.test_helpers import assert_true

from app.games.santorini.network_report import format_santorini_network_report


def test_santorini_network_report_mentions_input_and_output_sizes():
    report = format_santorini_network_report(limit=3)
    assert_true("Taille entrée : 225" in report)
    assert_true("Taille sortie : 144" in report)
    assert_true("Coups légaux indexés" in report)
    assert_true("sortie" in report)


TESTS = [
    ("Santorini affiche le rapport entrée/sortie", test_santorini_network_report_mentions_input_and_output_sizes),
]
