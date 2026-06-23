from app.games.morpion.rules import WINNING_LINES


def create_default_morpion_tactical_probes():
    """Retourne un banc tactique plus riche pour le morpion.

    Le but n'est pas de résoudre tout le jeu parfaitement ici, mais de vérifier
    que le modèle ne réussit pas seulement 4 cas triviaux. Les probes couvrent :
    - gagner immédiatement ;
    - bloquer une victoire immédiate ;
    - choisir un bon coup d'ouverture défensif.
    """

    probes = []
    probes.extend(_create_immediate_win_probes())
    probes.extend(_create_immediate_block_probes())
    probes.extend(_create_opening_probes())

    return probes


def _create_immediate_win_probes():
    probes = []

    for line_index, line in enumerate(WINNING_LINES):
        board = _create_board_with_line_threat(
            line=line,
            player="O",
            filler_player="X",
        )
        expected_move = _get_empty_cell_on_line(board, line)

        probes.append({
            "name": "win_line_" + str(line_index),
            "board": board,
            "expected_move": expected_move,
            "expected_moves": [expected_move],
            "description": "O doit gagner immédiatement sur la ligne " + str(line) + ".",
        })

    return probes


def _create_immediate_block_probes():
    probes = []

    for line_index, line in enumerate(WINNING_LINES):
        board = _create_board_with_line_threat(
            line=line,
            player="X",
            filler_player="O",
        )
        expected_move = _get_empty_cell_on_line(board, line)

        probes.append({
            "name": "block_line_" + str(line_index),
            "board": board,
            "expected_move": expected_move,
            "expected_moves": [expected_move],
            "description": "O doit bloquer la victoire immédiate de X sur la ligne " + str(line) + ".",
        })

    return probes


def _create_opening_probes():
    return [
        {
            "name": "answer_corner_with_center",
            "board": [
                "X", None, None,
                None, None, None,
                None, None, None,
            ],
            "expected_move": 4,
            "expected_moves": [4],
            "description": "Si X commence dans un coin, O doit prendre le centre.",
        },
        {
            "name": "answer_center_with_corner",
            "board": [
                None, None, None,
                None, "X", None,
                None, None, None,
            ],
            "expected_move": 0,
            "expected_moves": [0, 2, 6, 8],
            "description": "Si X prend le centre, un coin est une bonne réponse pour O.",
        },
    ]


def _create_board_with_line_threat(line, player, filler_player):
    board = [None, None, None, None, None, None, None, None, None]

    board[line[0]] = player
    board[line[1]] = player

    filler_cells = _find_safe_filler_cells(
        board=board,
        forbidden_cells=line,
        filler_player=filler_player,
    )

    for cell in filler_cells:
        board[cell] = filler_player

    return board


def _find_safe_filler_cells(board, forbidden_cells, filler_player):
    filler_cells = []
    working_board = board.copy()

    for index in range(9):
        if index in forbidden_cells:
            continue

        candidate_board = working_board.copy()
        candidate_board[index] = filler_player

        if not _has_two_on_open_line(candidate_board, filler_player):
            filler_cells.append(index)
            working_board[index] = filler_player

        if len(filler_cells) == 2:
            return filler_cells

    return filler_cells


def _has_two_on_open_line(board, player):
    for line in WINNING_LINES:
        player_count = 0
        empty_count = 0

        for cell in line:
            if board[cell] == player:
                player_count += 1
            elif board[cell] is None:
                empty_count += 1

        if player_count == 2 and empty_count == 1:
            return True

    return False


def _get_empty_cell_on_line(board, line):
    for cell in line:
        if board[cell] is None:
            return cell

    raise ValueError("La ligne ne contient aucune case vide.")
