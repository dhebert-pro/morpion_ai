BOARD_SIZE = 5
CELL_COUNT = BOARD_SIZE * BOARD_SIZE
COLUMNS = "ABCDE"
DIRECTIONS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def row_col_to_index(row, col):
    return row * BOARD_SIZE + col


def index_to_row_col(index):
    return index // BOARD_SIZE, index % BOARD_SIZE


def is_inside(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def cell_to_index(text):
    if not isinstance(text, str):
        return None

    value = text.strip().upper()
    if len(value) < 2:
        return None

    column = value[0]
    row_text = value[1:]

    if column not in COLUMNS:
        return None

    if not row_text.isdigit():
        return None

    row = int(row_text) - 1
    col = COLUMNS.index(column)

    if not is_inside(row, col):
        return None

    return row_col_to_index(row, col)


def index_to_cell(index):
    row, col = index_to_row_col(index)
    return COLUMNS[col] + str(row + 1)


def get_neighbors(index):
    row, col = index_to_row_col(index)
    neighbors = []

    for delta_col, delta_row in DIRECTIONS:
        next_row = row + delta_row
        next_col = col + delta_col

        if is_inside(next_row, next_col):
            neighbors.append(row_col_to_index(next_row, next_col))

    return neighbors
