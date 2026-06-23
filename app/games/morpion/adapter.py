from app.ai.game_adapter import GameAdapter

from app.games.morpion.rules import (
    create_new_game,
    copy_game,
    encode_game_state,
    get_legal_moves,
    is_valid_move,
    switch_player,
    get_game_result,
    get_winner,
    get_score_for_o,
)


def apply_move(game, move, player):
    game["board"][move] = player


def move_to_index(move):
    return move


def index_to_move(index):
    return index


MORPION_ADAPTER = GameAdapter(
    name="morpion",
    first_player="X",
    trained_player="O",
    opponent_player="X",
    create_new_game=create_new_game,
    copy_game=copy_game,
    encode_game_state=encode_game_state,
    get_legal_moves=get_legal_moves,
    is_valid_move=is_valid_move,
    apply_move=apply_move,
    switch_player=switch_player,
    get_game_result=get_game_result,
    get_winner=get_winner,
    get_score_for_trained_player=get_score_for_o,
    output_size=9,
    move_to_index=move_to_index,
    index_to_move=index_to_move,
)