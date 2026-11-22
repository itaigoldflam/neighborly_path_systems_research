"""File output helpers enforcing the repo-wide outputs/ directory convention.

Every experiment script's outputs (logs, .txt reports, plots) belong in a dedicated
outputs/ subdirectory next to the script, per CLAUDE.md's Output & File Management
Rules, rather than next to the source file or wherever the process happened to be
launched from.
"""
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

PathLike = Union[str, Path]


def output_dir(experiment_file: PathLike) -> Path:
    """Return <dir-containing-experiment_file>/outputs/, creating it if missing.

    Call as `output_dir(__file__)` from the experiment script so the output location
    is anchored to the script's own location, not the process's current working
    directory (several existing scripts break today if invoked from elsewhere).
    """
    d = Path(experiment_file).resolve().parent / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_report(path: PathLike, lines: Iterable[str]) -> None:
    """Write an iterable of lines to `path`, one per line, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            line = str(line)
            f.write(line if line.endswith("\n") else line + "\n")


def timestamped_filename(stem: str, ext: str = "txt") -> str:
    """Build a filename like '<stem>_20260809_142301.<ext>' for run-to-run uniqueness."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stem}_{ts}.{ext}"
