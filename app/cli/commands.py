import sys

from app.cli.help_command import print_help
from app.cli.training_commands import run_training_command, run_build_dataset_command
from app.cli.neural_commands import (
    run_neural_demo_command,
    run_neural_training_command,
    run_neural_reset_command,
)
from app.cli.neural_watch_commands import run_neural_benchmark_command
from app.cli.evaluation_commands import (
    run_evaluate_command,
    run_neural_evaluate_command,
    run_neural_tactical_evaluate_command,
    run_neural_reference_evaluate_command,
    run_neural_reference_diagnose_command,
)
from app.cli.play_commands import run_play_command
from app.cli.santorini_commands import (
    run_play_santorini_command,
    run_simulate_santorini_random_command,
    run_inspect_santorini_io_command,
    run_build_santorini_dataset_command,
)
from app.cli.test_command import run_test_command


def run_cli():
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()
    commands = {
        "train": run_training_command,
        "build-dataset": run_build_dataset_command,
        "neural-demo": run_neural_demo_command,
        "train-neural": run_neural_training_command,
        "reset-neural": run_neural_reset_command,
        "evaluate-neural": run_neural_evaluate_command,
        "evaluate-neural-tactical": run_neural_tactical_evaluate_command,
        "evaluate-neural-reference": run_neural_reference_evaluate_command,
        "diagnose-neural-reference": run_neural_reference_diagnose_command,
        "evaluate": run_evaluate_command,
        "play": run_play_command,
        "play-santorini": run_play_santorini_command,
        "simulate-santorini-random": run_simulate_santorini_random_command,
        "inspect-santorini-io": run_inspect_santorini_io_command,
        "build-santorini-dataset": run_build_santorini_dataset_command,
        "test": run_test_command,
    }

    if command == "neural-benchmark":
        run_neural_benchmark_command(start_from_saved_model=True)
        return

    if command == "neural-benchmark-reset":
        run_neural_benchmark_command(start_from_saved_model=False)
        return

    selected_command = commands.get(command)

    if selected_command is None:
        print("Commande inconnue :", command)
        print_help()
        return

    selected_command()
