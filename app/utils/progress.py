from app.config import PROGRESS_BAR_WIDTH


def build_progress_bar(current, total, width=PROGRESS_BAR_WIDTH):
    if total <= 0:
        return "[" + "-" * width + "] 0%"

    safe_current = current

    if safe_current < 0:
        safe_current = 0

    if safe_current > total:
        safe_current = total

    ratio = safe_current / total
    filled_count = int(width * ratio)
    empty_count = width - filled_count
    percent = round(ratio * 100)

    return "[" + "#" * filled_count + "-" * empty_count + "] " + str(percent) + "%"


def print_progress(prefix, current, total):
    print("\r" + prefix + " " + build_progress_bar(current, total), end="")

    if current >= total:
        print()