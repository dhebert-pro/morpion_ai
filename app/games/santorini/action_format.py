from app.games.santorini.coordinates import cell_to_index, index_to_cell


def parse_human_input(text):
    value = text.strip().upper().replace(" ", "")

    if value == "Q":
        return "quit"

    if value == "?":
        return "help"

    if "-" not in value:
        return cell_to_index(value)

    if ":" in value:
        move_part, build_text = value.split(":", 1)
        build_cell = cell_to_index(build_text)
    else:
        move_part = value
        build_cell = None

    if move_part.count("-") != 1:
        return None

    from_text, to_text = move_part.split("-")
    from_cell = cell_to_index(from_text)
    to_cell = cell_to_index(to_text)

    if from_cell is None or to_cell is None:
        return None

    if build_cell is None and ":" in value:
        return None

    return {"from": from_cell, "to": to_cell, "build": build_cell}


def format_action(action):
    text = index_to_cell(action["from"]) + "-" + index_to_cell(action["to"])

    if action.get("build") is not None:
        text += ":" + index_to_cell(action["build"])

    return text


def format_action_list(actions, limit=None):
    selected_actions = actions

    if limit is not None:
        selected_actions = actions[:limit]

    return ", ".join(format_action(action) for action in selected_actions)
