import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tests.test_helpers import run_test

from tests.test_rules import TESTS as RULES_TESTS
from tests.test_engine import TESTS as ENGINE_TESTS
from tests.test_strategies import TESTS as STRATEGIES_TESTS
from tests.test_training import TESTS as TRAINING_TESTS
from tests.test_evaluation import TESTS as EVALUATION_TESTS
from tests.test_storage import TESTS as STORAGE_TESTS
from tests.test_progress import TESTS as PROGRESS_TESTS


def get_all_tests():
    return (
        RULES_TESTS
        + ENGINE_TESTS
        + STRATEGIES_TESTS
        + TRAINING_TESTS
        + EVALUATION_TESTS
        + STORAGE_TESTS
        + PROGRESS_TESTS
    )


def run_all_tests():
    tests = get_all_tests()
    passed_count = 0

    print("Lancement des tests...")
    print()

    for name, test_function in tests:
        if run_test(name, test_function):
            passed_count += 1

    total_count = len(tests)

    print()
    print("Résumé :", passed_count, "/", total_count, "tests passés.")

    if passed_count != total_count:
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()
