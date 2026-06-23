from app.games.morpion.adapter import MORPION_ADAPTER


def simulate_player_move(game, move, player, game_adapter=MORPION_ADAPTER):
    simulated_game = game_adapter.copy_game(game)
    game_adapter.apply_move(simulated_game, move, player)
    return simulated_game


def count_immediate_winning_moves_for_player(
    game,
    player,
    game_adapter=MORPION_ADAPTER,
):
    count = 0

    for move in game_adapter.get_legal_moves(game):
        simulated_game = simulate_player_move(
            game,
            move,
            player,
            game_adapter,
        )

        if game_adapter.get_game_result(simulated_game) == player:
            count += 1

    return count


def count_fork_moves_for_player(
    game,
    player,
    game_adapter=MORPION_ADAPTER,
    fork_threats_count=2,
):
    count = 0

    for move in game_adapter.get_legal_moves(game):
        simulated_game = simulate_player_move(
            game,
            move,
            player,
            game_adapter,
        )

        if game_adapter.get_game_result(simulated_game) != "ongoing":
            continue

        threats_count = count_immediate_winning_moves_for_player(
            simulated_game,
            player,
            game_adapter,
        )

        if threats_count >= fork_threats_count:
            count += 1

    return count


def move_allows_immediate_opponent_win(
    game,
    move,
    game_adapter=MORPION_ADAPTER,
):
    simulated_game = simulate_player_move(
        game,
        move,
        game_adapter.trained_player,
        game_adapter,
    )

    if game_adapter.get_game_result(simulated_game) != "ongoing":
        return False

    return count_immediate_winning_moves_for_player(
        simulated_game,
        game_adapter.opponent_player,
        game_adapter,
    ) > 0


def count_opponent_fork_replies_after_move(
    game,
    move,
    game_adapter=MORPION_ADAPTER,
    fork_threats_count=2,
):
    simulated_game = simulate_player_move(
        game,
        move,
        game_adapter.trained_player,
        game_adapter,
    )

    if game_adapter.get_game_result(simulated_game) != "ongoing":
        return 0

    fork_replies = 0

    for opponent_move in game_adapter.get_legal_moves(simulated_game):
        reply_game = simulate_player_move(
            simulated_game,
            opponent_move,
            game_adapter.opponent_player,
            game_adapter,
        )

        if game_adapter.get_game_result(reply_game) != "ongoing":
            continue

        immediate_threats = count_immediate_winning_moves_for_player(
            reply_game,
            game_adapter.opponent_player,
            game_adapter,
        )

        if immediate_threats >= fork_threats_count:
            fork_replies += 1

    return fork_replies


def get_responses_avoiding_immediate_loss(
    game,
    game_adapter=MORPION_ADAPTER,
):
    responses = []

    for move in game_adapter.get_legal_moves(game):
        response_game = simulate_player_move(
            game,
            move,
            game_adapter.trained_player,
            game_adapter,
        )
        result = game_adapter.get_game_result(response_game)

        if result == game_adapter.trained_player:
            responses.append(move)
            continue

        opponent_threats = count_immediate_winning_moves_for_player(
            response_game,
            game_adapter.opponent_player,
            game_adapter,
        )

        if opponent_threats == 0:
            responses.append(move)

    return responses


def opponent_reply_forces_fork_after_defense(
    game_after_opponent_reply,
    game_adapter=MORPION_ADAPTER,
):
    immediate_threats = count_immediate_winning_moves_for_player(
        game_after_opponent_reply,
        game_adapter.opponent_player,
        game_adapter,
    )

    if immediate_threats == 0:
        return False

    responses = get_responses_avoiding_immediate_loss(
        game_after_opponent_reply,
        game_adapter,
    )

    if len(responses) == 0:
        return False

    for response in responses:
        response_game = simulate_player_move(
            game_after_opponent_reply,
            response,
            game_adapter.trained_player,
            game_adapter,
        )

        fork_moves = count_fork_moves_for_player(
            response_game,
            game_adapter.opponent_player,
            game_adapter,
        )

        if fork_moves == 0:
            return False

    return True


def count_opponent_forcing_fork_replies_after_move(
    game,
    move,
    game_adapter=MORPION_ADAPTER,
):
    simulated_game = simulate_player_move(
        game,
        move,
        game_adapter.trained_player,
        game_adapter,
    )

    if game_adapter.get_game_result(simulated_game) != "ongoing":
        return 0

    forcing_replies = 0

    for opponent_move in game_adapter.get_legal_moves(simulated_game):
        reply_game = simulate_player_move(
            simulated_game,
            opponent_move,
            game_adapter.opponent_player,
            game_adapter,
        )

        if game_adapter.get_game_result(reply_game) != "ongoing":
            continue

        if opponent_reply_forces_fork_after_defense(
            reply_game,
            game_adapter,
        ):
            forcing_replies += 1

    return forcing_replies


