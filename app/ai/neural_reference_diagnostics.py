import random

from app.ai.neural_strategy import choose_neural_move
from app.ai.reference_opponents import (
    REFERENCE_TACTICAL,
    choose_reference_opponent_move,
)
from app.ai.tactical_guard import (
    count_immediate_winning_moves_for_player,
    count_opponent_fork_replies_after_move,
    count_opponent_forcing_fork_replies_after_move,
)
from app.games.morpion.adapter import MORPION_ADAPTER


def play_traced_neural_against_reference_game(
    model_data,
    reference_name=REFERENCE_TACTICAL,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    game = game_adapter.create_new_game()
    current_player = game_adapter.first_player
    moves = []
    first_danger = None

    while game_adapter.get_game_result(game) == "ongoing":
        before_game = game_adapter.copy_game(game)

        if current_player == game_adapter.trained_player:
            move = choose_neural_move(
                game=game,
                model_data=model_data,
                game_adapter=game_adapter,
            )
        else:
            move = choose_reference_opponent_move(
                game=game,
                player=game_adapter.opponent_player,
                opponent=game_adapter.trained_player,
                reference_name=reference_name,
                game_adapter=game_adapter,
                rng=rng,
            )

        game_adapter.apply_move(game, move, current_player)
        moves.append({
            "player": current_player,
            "move": move,
            "board": game_adapter.encode_game_state(game),
        })

        if current_player == game_adapter.trained_player:
            danger = describe_danger_after_trained_move(
                before_game=before_game,
                move=move,
                game_after_move=game,
                game_adapter=game_adapter,
            )

            if danger is not None and first_danger is None:
                danger["move_number"] = len(moves)
                first_danger = danger

        current_player = game_adapter.switch_player(current_player)

    return {
        "result": game_adapter.get_game_result(game),
        "moves": moves,
        "final_board": game_adapter.encode_game_state(game),
        "first_danger": first_danger,
    }


def describe_danger_after_trained_move(
    before_game,
    move,
    game_after_move,
    game_adapter=MORPION_ADAPTER,
):
    immediate_threats = count_immediate_winning_moves_for_player(
        game_after_move,
        game_adapter.opponent_player,
        game_adapter,
    )
    fork_replies = count_opponent_fork_replies_after_move(
        before_game,
        move,
        game_adapter,
    )
    forcing_fork_replies = count_opponent_forcing_fork_replies_after_move(
        before_game,
        move,
        game_adapter,
    )

    if (
        immediate_threats == 0
        and fork_replies == 0
        and forcing_fork_replies == 0
    ):
        return None

    return {
        "trained_move": move,
        "immediate_threats": immediate_threats,
        "fork_replies": fork_replies,
        "forcing_fork_replies": forcing_fork_replies,
        "board": game_adapter.encode_game_state(game_after_move),
    }


def collect_reference_loss_diagnostics(
    model_data,
    reference_name=REFERENCE_TACTICAL,
    games_count=200,
    max_losses=3,
    seed=0,
    game_adapter=MORPION_ADAPTER,
):
    rng = random.Random(seed)
    losses = []
    losses_count = 0

    for _ in range(games_count):
        trace = play_traced_neural_against_reference_game(
            model_data=model_data,
            reference_name=reference_name,
            game_adapter=game_adapter,
            rng=rng,
        )

        if trace["result"] == game_adapter.opponent_player:
            losses_count += 1

            if len(losses) < max_losses:
                losses.append(trace)

    return {
        "reference_name": reference_name,
        "games_count": games_count,
        "losses_count": losses_count,
        "shown_losses_count": len(losses),
        "losses": losses,
        "trained_player": game_adapter.trained_player,
        "opponent_player": game_adapter.opponent_player,
    }


def format_reference_loss_diagnostics_report(diagnostics):
    lines = []
    lines.append("Diagnostic des défaites contre adversaire de référence")
    lines.append("Adversaire : " + str(diagnostics["reference_name"]))
    lines.append("Parties analysées : " + str(diagnostics["games_count"]))
    lines.append("Défaites trouvées : " + str(diagnostics["losses_count"]))
    lines.append("Défaites détaillées : " + str(diagnostics["shown_losses_count"]))

    if diagnostics["losses_count"] == 0:
        lines.append("Aucune défaite trouvée sur cet échantillon.")
        return "\n".join(lines)

    for index, loss in enumerate(diagnostics["losses"], start=1):
        lines.append("")
        lines.append("Défaite " + str(index))
        lines.append("Coups : " + format_moves_line(loss["moves"]))
        lines.append("Plateau final :")
        lines.extend(format_board(loss["final_board"]))

        if loss["first_danger"] is None:
            lines.append("Premier danger détecté : aucun marqueur simple.")
        else:
            lines.extend(format_danger(loss["first_danger"]))

    return "\n".join(lines)


def format_moves_line(moves):
    parts = []

    for move in moves:
        parts.append(str(move["player"]) + str(move["move"]))

    return " ".join(parts)


def format_board(state_key):
    lines = []

    for start in range(0, len(state_key), 3):
        row = state_key[start:start + 3]
        lines.append("  " + " ".join(row))

    return lines


def format_danger(danger):
    lines = []
    lines.append("Premier danger détecté après un coup de O :")
    lines.append("- numéro du coup : " + str(danger["move_number"]))
    lines.append("- coup de O : " + str(danger["trained_move"]))
    lines.append("- menaces immédiates adverses : " + str(danger["immediate_threats"]))
    lines.append("- réponses adverses créant une fourchette : " + str(danger["fork_replies"]))
    lines.append(
        "- réponses adverses forçant une fourchette après défense : "
        + str(danger.get("forcing_fork_replies", 0))
    )
    lines.append("- plateau après ce coup :")
    lines.extend(format_board(danger["board"]))
    return lines
