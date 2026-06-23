from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class GameAdapter:
    """Contrat minimal qu'un jeu doit fournir au moteur d'IA.

    Le moteur d'IA ne doit pas connaître la structure interne du jeu.
    Il utilise uniquement ces fonctions pour créer, copier, encoder,
    jouer et évaluer des états.

    Les champs output_size, move_to_index et index_to_move servent à rendre
    les coups compatibles avec un futur réseau de neurones, qui produit
    forcément un vecteur de sorties indexées.
    """

    name: str
    first_player: str
    trained_player: str
    opponent_player: str

    create_new_game: Callable
    copy_game: Callable
    encode_game_state: Callable
    get_legal_moves: Callable
    is_valid_move: Callable
    apply_move: Callable
    switch_player: Callable
    get_game_result: Callable
    get_winner: Callable
    get_score_for_trained_player: Callable

    output_size: int
    move_to_index: Callable
    index_to_move: Callable