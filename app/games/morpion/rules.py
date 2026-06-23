WINNING_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6]
]


def create_new_game():
    return {
        "board": [None, None, None, None, None, None, None, None, None],
        "current_player": "X"
    }


def copy_game(game):
    return {
        "board": game["board"].copy(),
        "current_player": game["current_player"]
    }


def encode_game_state(game):
    symbols = []

    for cell in game["board"]:
        if cell is None:
            symbols.append(".")
        else:
            symbols.append(cell)

    return "".join(symbols)


def parse_human_input(text):
    text = text.strip().lower()

    if text == "q":
        return "quit"

    try:
        return int(text)
    except ValueError:
        return None


def switch_player(player):
    if player == "X":
        return "O"

    return "X"


def is_valid_move(game, move):
    if move < 0 or move > 8:
        return False

    if game["board"][move] is not None:
        return False

    return True


def get_legal_moves(game):
    legal_moves = []

    for index in range(9):
        if game["board"][index] is None:
            legal_moves.append(index)

    return legal_moves


def get_winner(game):
    for line in WINNING_LINES:
        a = line[0]
        b = line[1]
        c = line[2]

        if game["board"][a] is not None:
            if game["board"][a] == game["board"][b] == game["board"][c]:
                return game["board"][a]

    return None


def get_game_result(game):
    winner = get_winner(game)

    if winner is not None:
        return winner

    if len(get_legal_moves(game)) == 0:
        return "draw"

    return "ongoing"


def get_score_for_o(result):
    if result == "O":
        return 1.0

    if result == "draw":
        return 0.5

    if result == "X":
        return 0.0

    raise ValueError("Résultat inconnu : " + str(result))