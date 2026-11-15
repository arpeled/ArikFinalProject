"""Regression tests for experiment-result extraction (review finding F5).

Folders iteration_001-005 used to hold copies of the results of experiments
122 and 126-130, and the analysis scripts read the *latest* results file, so
experiments 1-5 were reported with the values of later experiments.
"""
import hashlib
import re
import warnings
from pathlib import Path

import pytest
import yaml

from analysis.iteration_files import list_results_files, select_results_file

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "experiments" / "auto_improvement_runs"


def _iteration_dirs():
    return sorted(p for p in RUNS.glob("iteration_*") if re.fullmatch(r"iteration_\d+", p.name))


# --- unit behaviour of the selection rule -----------------------------------

def test_select_returns_none_when_no_results(tmp_path):
    assert select_results_file(tmp_path) is None


def test_select_prefers_earliest_and_warns_when_several(tmp_path):
    (tmp_path / "pipeline_results_20251227-144248.csv").write_text("Label,AUC\n")
    (tmp_path / "pipeline_results_20260207-093214.csv").write_text("Label,AUC\n")
    with pytest.warns(UserWarning, match="2 results files"):
        chosen = select_results_file(tmp_path)
    assert chosen.name == "pipeline_results_20251227-144248.csv"


def test_select_single_file_does_not_warn(tmp_path):
    (tmp_path / "pipeline_results_20251227-144248.csv").write_text("Label,AUC\n")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert select_results_file(tmp_path).name == "pipeline_results_20251227-144248.csv"


# --- repository data invariants ---------------------------------------------

def test_no_results_file_is_a_copy_of_another_experiment():
    """A results file must belong to exactly one experiment folder."""
    by_content = {}
    for d in _iteration_dirs():
        for f in list_results_files(d):
            by_content.setdefault(hashlib.sha256(f.read_bytes()).hexdigest(), []).append(f)
    dupes = [[str(p.relative_to(RUNS)) for p in group]
             for group in by_content.values() if len({p.parent for p in group}) > 1]
    assert dupes == []


def test_each_folder_has_at_most_one_results_file():
    multi = {d.name: len(list_results_files(d)) for d in _iteration_dirs() if len(list_results_files(d)) > 1}
    assert multi == {}


def test_no_config_comes_from_a_later_experiment():
    """metadata.iteration names the experiment that produced the config (usually the
    previous one), so it can never be later than the folder it sits in. Folders 1-5
    used to hold copies of experiment 126's config."""
    wrong = {}
    for d in _iteration_dirs():
        cfg = d / "config.yaml"
        if not cfg.exists():
            continue
        meta = (yaml.safe_load(cfg.read_text()) or {}).get("metadata") or {}
        it = meta.get("iteration")
        if it is not None and int(it) > int(d.name.split("_")[1]):
            wrong[d.name] = it
    assert wrong == {}


@pytest.mark.parametrize("exp,expected", [(1, 0.743), (2, 0.802), (3, 0.785), (4, 0.742), (5, 0.731)])
def test_early_experiments_report_their_original_macro_auc(exp, expected):
    import pandas as pd
    f = select_results_file(RUNS / f"iteration_{exp:03d}")
    assert pd.read_csv(f)["AUC"].mean() == pytest.approx(expected, abs=5e-4)
