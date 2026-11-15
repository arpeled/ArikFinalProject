"""Shared rule for picking the results file of an experiment folder.

Each experiment folder (experiments/auto_improvement_runs/iteration_NNN) is
expected to hold one ``pipeline_results_<timestamp>.csv``. Earlier analysis
scripts took the *latest* file, which silently reported later experiments'
results for experiments 1-5 when files had been mis-filed into those folders
(see docs/review/Independent_Verification_Report.docx, finding F5).

The rule here is: use the earliest file (the original run of that folder) and
warn when there is more than one, so a mix-up is visible instead of silent.
"""
from __future__ import annotations

import warnings
from pathlib import Path

RESULTS_GLOB = "pipeline_results_*.csv"


def list_results_files(iter_dir: Path | str) -> list[Path]:
    """All results files of an experiment folder, oldest first (timestamped names)."""
    return sorted(Path(iter_dir).glob(RESULTS_GLOB))


def select_results_file(iter_dir: Path | str) -> Path | None:
    """Return the results file for an experiment folder, or None if there is none."""
    files = list_results_files(iter_dir)
    if not files:
        return None
    if len(files) > 1:
        warnings.warn(
            f"{Path(iter_dir).name}: {len(files)} results files found; using the "
            f"earliest ({files[0].name}). Check that the others belong to this experiment.",
            UserWarning,
            stacklevel=2,
        )
    return files[0]
