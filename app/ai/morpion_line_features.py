from app.games.morpion.rules import WINNING_LINES


def encode_line_threats_as_neural_input(state_key, player):
    if len(state_key) != 9:
        return []

    threats = []

    for line in WINNING_LINES:
        threats.append(_encode_line_threat(state_key, player, line))

    return threats


def _encode_line_threat(state_key, player, line):
    player_count = 0
    empty_count = 0

    for cell in line:
        if state_key[cell] == player:
            player_count += 1
        elif state_key[cell] == ".":
            empty_count += 1

    if player_count == 2 and empty_count == 1:
        return 1.0

    return 0.0
