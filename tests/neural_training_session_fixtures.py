def create_small_encoded_dataset():
    return {
        "game": "test_game",
        "trained_player": "B",
        "opponent_player": "A",
        "input_size": 4,
        "output_size": 3,
        "source_examples_count": 2,
        "encoded_examples_count": 2,
        "examples": [
            {
                "state_key": "state_1",
                "inputs": [1.0, 0.0, 0.0, 0.0],
                "targets": [1.0, 0.0, 0.0],
                "legal_moves_mask": [1.0, 1.0, 0.0],
                "legal_moves": [0, 1],
                "legal_move_indexes": [0, 1],
                "best_move": 0,
                "best_move_index": 0,
            },
            {
                "state_key": "state_2",
                "inputs": [0.0, 1.0, 0.0, 0.0],
                "targets": [0.0, 1.0, 0.0],
                "legal_moves_mask": [1.0, 1.0, 0.0],
                "legal_moves": [0, 1],
                "best_move": 1,
                "best_move_index": 1,
            },
        ],
    }
