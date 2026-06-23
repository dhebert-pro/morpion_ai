from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from tests.test_rules import TESTS as RULES_TESTS
from tests.test_engine import TESTS as ENGINE_TESTS
from tests.test_strategies import TESTS as STRATEGIES_TESTS
from tests.test_training import TESTS as TRAINING_TESTS
from tests.test_evaluation import TESTS as EVALUATION_TESTS
from tests.test_storage import TESTS as STORAGE_TESTS
from tests.test_progress import TESTS as PROGRESS_TESTS
from tests.test_game_adapter import TESTS as GAME_ADAPTER_TESTS
from tests.test_move_scoring import TESTS as MOVE_SCORING_TESTS
from tests.test_training_dataset import TESTS as TRAINING_DATASET_TESTS
from tests.test_neural_encoding import TESTS as NEURAL_ENCODING_TESTS
from tests.test_neural_pipeline import TESTS as NEURAL_PIPELINE_TESTS
from tests.test_neural_training_session import TESTS as NEURAL_TRAINING_SESSION_TESTS
from tests.test_neural_network import TESTS as NEURAL_NETWORK_TESTS
from tests.test_neural_strategy import TESTS as NEURAL_STRATEGY_TESTS
from tests.test_neural_evaluation import TESTS as NEURAL_EVALUATION_TESTS
from tests.test_neural_diagnostics import TESTS as NEURAL_DIAGNOSTICS_TESTS
from tests.test_neural_model_service import TESTS as NEURAL_MODEL_SERVICE_TESTS
from tests.test_neural_benchmark import TESTS as NEURAL_BENCHMARK_TESTS
from tests.test_tactical_evaluation import TESTS as TACTICAL_EVALUATION_TESTS
from tests.test_tactical_training import TESTS as TACTICAL_TRAINING_TESTS


def get_all_tests():
    return (
        RULES_TESTS
        + ENGINE_TESTS
        + STRATEGIES_TESTS
        + TRAINING_TESTS
        + EVALUATION_TESTS
        + STORAGE_TESTS
        + PROGRESS_TESTS
        + GAME_ADAPTER_TESTS
        + MOVE_SCORING_TESTS
        + TRAINING_DATASET_TESTS
        + NEURAL_ENCODING_TESTS
        + NEURAL_PIPELINE_TESTS
        + NEURAL_TRAINING_SESSION_TESTS
        + NEURAL_NETWORK_TESTS
        + NEURAL_STRATEGY_TESTS
        + NEURAL_EVALUATION_TESTS
        + NEURAL_DIAGNOSTICS_TESTS
        + NEURAL_MODEL_SERVICE_TESTS
        + NEURAL_BENCHMARK_TESTS
        + TACTICAL_EVALUATION_TESTS
        + TACTICAL_TRAINING_TESTS
    )


def run_single_test(test_name, test_function):
    try:
        test_function()
        print("OK  -", test_name)
        return True
    except AssertionError as error:
        print("NON -", test_name)
        print("Erreur d'assertion :", error)
        return False
    except Exception:
        print("NON -", test_name)
        traceback.print_exc()
        return False


def run_all_tests():
    tests = get_all_tests()
    passed_count = 0

    for test_name, test_function in tests:
        if run_single_test(test_name, test_function):
            passed_count += 1

    print()
    print("Résumé :", passed_count, "/", len(tests), "tests passés.")

    if passed_count != len(tests):
        raise SystemExit(1)


if __name__ == "__main__":
    run_all_tests()