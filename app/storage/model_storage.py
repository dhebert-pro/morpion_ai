import json
import os

from app.config import MODEL_FILE


def save_model(model, file_path=MODEL_FILE):
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(file_path=MODEL_FILE):
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    model = {}

    for state_key in data:
        model[state_key] = int(data[state_key])

    return model