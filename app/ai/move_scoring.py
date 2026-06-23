from app.ai.training import evaluate_player_move
from app.games.morpion.adapter import MORPION_ADAPTER


def score_legal_moves(
    game,
    player,
    simulations_per_move,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    """Attribue un score à chaque coup légal dans un état donné.

    Le score est calculé par simulations Monte-Carlo.
    Il représente l'intérêt du coup pour le joueur entraîné par l'adaptateur.
    """

    move_scores = {}

    for move in game_adapter.get_legal_moves(game):
        move_scores[move] = evaluate_player_move(
            game,
            move,
            player,
            simulations_per_move,
            game_adapter,
            rng=rng,
        )

    return move_scores


def choose_best_scored_move(move_scores):
    if len(move_scores) == 0:
        return None

    best_move = None
    best_score = None

    for move, score in move_scores.items():
        if best_score is None or score > best_score:
            best_move = move
            best_score = score

    return best_move


def create_move_score_example(
    game,
    simulations_per_move,
    game_adapter=MORPION_ADAPTER,
    rng=None,
):
    """Crée un exemple exploitable plus tard par un modèle neuronal."""

    move_scores = score_legal_moves(
        game,
        game_adapter.trained_player,
        simulations_per_move,
        game_adapter,
        rng=rng,
    )
    best_move = choose_best_scored_move(move_scores)

    formatted_scores = []

    for move, score in move_scores.items():
        formatted_scores.append({
            "move": move,
            "score": score,
        })

    return {
        "state_key": game_adapter.encode_game_state(game),
        "player": game_adapter.trained_player,
        "move_scores": formatted_scores,
        "best_move": best_move,
    }
