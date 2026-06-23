from tests.test_helpers import assert_equal, assert_true

from app.games.santorini.dataset import build_santorini_move_score_dataset
from app.games.santorini.neural_dataset import encode_santorini_move_score_dataset
from app.games.santorini.encoding import SANTORINI_INPUT_SIZE
from app.games.santorini.action_index import ACTION_OUTPUT_SIZE


def make_raw_dataset():
    return build_santorini_move_score_dataset(
        games_count=3,
        simulations_per_move=1,
        max_examples=1,
        seed=0,
    )


def test_santorini_dataset_examples_contain_inputs():
    dataset = make_raw_dataset()
    example = dataset["examples"][0]
    assert_equal(len(example["inputs"]), SANTORINI_INPUT_SIZE)


def test_encode_santorini_dataset_dimensions():
    dataset = make_raw_dataset()
    encoded = encode_santorini_move_score_dataset(dataset)
    assert_equal(encoded["input_size"], SANTORINI_INPUT_SIZE)
    assert_equal(encoded["output_size"], ACTION_OUTPUT_SIZE)
    assert_equal(encoded["encoded_examples_count"], 1)


def test_encode_santorini_example_masks_legal_moves():
    dataset = make_raw_dataset()
    encoded = encode_santorini_move_score_dataset(dataset)
    example = encoded["examples"][0]
    assert_true(sum(example["legal_moves_mask"]) > 0)
    assert_equal(len(example["targets"]), ACTION_OUTPUT_SIZE)
    assert_equal(len(example["legal_moves_mask"]), ACTION_OUTPUT_SIZE)


TESTS = [
    ("Santorini dataset conserve les entrées réseau", test_santorini_dataset_examples_contain_inputs),
    ("Santorini encode le dataset neuronal", test_encode_santorini_dataset_dimensions),
    ("Santorini encode le masque des coups légaux", test_encode_santorini_example_masks_legal_moves),
]
