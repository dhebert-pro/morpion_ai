from app.ai.game_adapter import GameAdapter
from app.games.santorini.action_index import ACTION_OUTPUT_SIZE, decode_output_index
from app.games.santorini.encoding import encode_santorini_state_key
from app.games.santorini.indexed_actions import (
    get_indexed_legal_actions,
    move_to_output_index,
)
from app.games.santorini.rules import (
    apply_action,
    copy_game,
    create_new_game,
    get_game_result,
    is_valid_action,
    switch_player,
)


def apply_move(game, move, player):
    if game["current_player"] != player:
        raise ValueError("Ce n'est pas au joueur " + str(player) + " de jouer.")

    return apply_action(game, move)


def get_winner(game):
    return game.get("winner")


def get_score_for_o(result):
    if result == "O":
        return 1.0
    if result == "X":
        return 0.0
    if result == "ongoing":
        return 0.5
    raise ValueError("Résultat Santorini inconnu : " + str(result))


SANTORINI_ADAPTER = GameAdapter(
    name="santorini",
    first_player="X",
    trained_player="O",
    opponent_player="X",
    create_new_game=create_new_game,
    copy_game=copy_game,
    encode_game_state=encode_santorini_state_key,
    get_legal_moves=get_indexed_legal_actions,
    is_valid_move=is_valid_action,
    apply_move=apply_move,
    switch_player=switch_player,
    get_game_result=get_game_result,
    get_winner=get_winner,
    get_score_for_trained_player=get_score_for_o,
    output_size=ACTION_OUTPUT_SIZE,
    move_to_index=move_to_output_index,
    index_to_move=decode_output_index,
)
