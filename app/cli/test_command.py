import subprocess
import sys

from app.config import PROJECT_ROOT, TEST_FILE


def run_test_command():
    if not TEST_FILE.exists():
        print("Fichier de tests introuvable :", TEST_FILE)
        return

    completed_process = subprocess.run(
        [sys.executable, str(TEST_FILE)],
        cwd=PROJECT_ROOT,
    )

    if completed_process.returncode != 0:
        raise SystemExit(completed_process.returncode)
