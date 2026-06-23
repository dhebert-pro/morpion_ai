import random

from app.games.santorini.action_format import format_action
from app.games.santorini.display import format_board
from app.games.santorini.neural_player import (
    TRAINED_PLAYER,
    create_random_started_santorini_game,
    choose_santorini_neural_action_from_network,
)
from app.games.santorini.reference_agents import choose_reference_action
from app.games.santorini.rules import apply_action, copy_game, get_game_result, switch_player
from app.games.santorini.tactical_risk import describe_santorini_risk_after_action

OPPONENT_PLAYER = "X"


def diagnose_santorini_neural_vs_reference(
    network,
    opponent_name="climber",
    games_count=100,
    seed=0,
    max_turns=160,
    max_details=3,
):
    rng = random.Random(seed + 9_000)
    losses = []

    for index in range(games_count):
        start = create_random_started_santorini_game(rng)
        game_report = _play_and_diagnose_game(
            start,
            network,
            opponent_name,
            random.Random(seed + 10_000 + index),
            max_turns,
        )

        if game_report["result"] == OPPONENT_PLAYER:
            losses.append(game_report)

    return {
        "opponent": opponent_name,
        "games_count": games_count,
        "losses_count": len(losses),
        "details": losses[:max_details],
    }


def format_santorini_reference_diagnostic(report):
    lines = []
    lines.append("Diagnostic Santorini contre adversaire de référence")
    lines.append("Adversaire : " + report["opponent"])
    lines.append("Parties analysées : " + str(report["games_count"]))
    lines.append("Défaites trouvées : " + str(report["losses_count"]))
    lines.append("Défaites détaillées : " + str(len(report["details"])))

    if not report["details"]:
        lines.append("Aucune défaite trouvée sur cet échantillon.")
        return "\n".join(lines)

    for index, detail in enumerate(report["details"], start=1):
        lines.append("")
        lines.append("Défaite " + str(index))
        lines.append("Coups : " + " ".join(detail["moves"]))
        lines.append("Plateau final :")
        lines.append(detail["final_board"])
        lines.extend(_format_first_danger(detail.get("first_danger")))

    return "\n".join(lines)


def _play_and_diagnose_game(start_game, network, opponent_name, rng, max_turns):
    game = copy_game(start_game)
    moves = []
    o_dangers = []
    turn_count = 0

    while get_game_result(game) == "ongoing" and turn_count < max_turns:
        action = _choose_action(game, network, opponent_name, rng)

        if action is None:
            game["winner"] = switch_player(game["current_player"])
            game["phase"] = "finished"
            break

        if game["current_player"] == TRAINED_PLAYER:
            o_dangers.append(_describe_o_action_danger(game, action, len(moves) + 1))

        moves.append(game["current_player"] + format_action(action))
        apply_action(game, action)
        turn_count += 1

    result = get_game_result(game)
    return {
        "result": "draw" if result == "ongoing" else result,
        "moves": moves,
        "final_board": format_board(game),
        "first_danger": _first_meaningful_danger(o_dangers),
    }


def _choose_action(game, network, opponent_name, rng):
    if game["current_player"] == TRAINED_PLAYER:
        return choose_santorini_neural_action_from_network(game, network, rng)

    return choose_reference_action(game, opponent_name, rng)


def _describe_o_action_danger(game, action, move_number):
    risk = describe_santorini_risk_after_action(game, action)
    return {
        "move_number": move_number,
        "action": format_action(action),
        "opponent_immediate_wins": risk["opponent_immediate_wins"],
        "opponent_threat_replies": risk["opponent_threat_replies"],
        "opponent_max_destination_height": risk["opponent_max_destination_height"],
        "opponent_high_climb_actions": risk["opponent_high_climb_actions"],
    }


def _first_meaningful_danger(dangers):
    for danger in dangers:
        if danger["opponent_immediate_wins"] > 0:
            return danger
        if danger["opponent_threat_replies"] > 0:
            return danger
        if danger["opponent_max_destination_height"] >= 2:
            return danger

    return dangers[0] if dangers else None


def _format_first_danger(danger):
    if not danger:
        return ["Premier danger détecté : aucun coup de O analysé."]

    return [
        "Premier danger détecté après un coup de O :",
        "- numéro du coup : " + str(danger["move_number"]),
        "- coup de O : " + danger["action"],
        "- victoires immédiates offertes à X : " + str(danger["opponent_immediate_wins"]),
        "- réponses X créant une menace niveau 3 : " + str(danger["opponent_threat_replies"]),
        "- hauteur max atteignable par X : " + str(danger["opponent_max_destination_height"]),
        "- coups X vers hauteur 2+ : " + str(danger["opponent_high_climb_actions"]),
    ]
