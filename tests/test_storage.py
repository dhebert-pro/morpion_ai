import os

from tests.test_helpers import assert_equal

from app.storage.model_storage import save_model, load_model


def test_save_and_load_model():
    file_path = "test_trained_model.json"

    model = {
        "....X....": 0,
        "O...X....": 8
    }

    save_model(model, file_path)
    loaded_model = load_model(file_path)

    assert_equal(loaded_model, model)

    if os.path.exists(file_path):
        os.remove(file_path)


def test_load_missing_model_returns_empty_model():
    file_path = "missing_test_model.json"

    if os.path.exists(file_path):
        os.remove(file_path)

    loaded_model = load_model(file_path)

    assert_equal(loaded_model, {})


def test_loaded_model_values_are_numbers():
    file_path = "test_trained_model_numbers.json"

    model = {
        "....X....": 0
    }

    save_model(model, file_path)
    loaded_model = load_model(file_path)

    assert_equal(loaded_model["....X...."], 0)
    assert_equal(type(loaded_model["....X...."]), int)

    if os.path.exists(file_path):
        os.remove(file_path)


TESTS = [
    ("Sauvegarder puis charger un modèle", test_save_and_load_model),
    ("Charger un modèle absent renvoie un modèle vide", test_load_missing_model_returns_empty_model),
    ("Les coups chargés depuis le modèle sont des nombres", test_loaded_model_values_are_numbers),
]
