import json
from pathlib import Path

from app.config import MODEL_FILE


def save_model(model, file_path=MODEL_FILE):
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(model, file, ensure_ascii=False, indent=2)


def load_model(file_path=MODEL_FILE):
    path = Path(file_path)

    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    model = {}

    for state_key in data:
        model[state_key] = int(data[state_key])

    return model