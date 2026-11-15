"""Tests for the data behind Appendix figures B.2 and C.1 (review findings F5, C.1 axes)."""
import math
import warnings
from pathlib import Path

import pytest

from analysis.thesis_appendix_c_axes_timeline import collect_axes

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def axes():
    return collect_axes()


def test_axes_use_experiment_numbers(axes):
    assert axes.index.min() == 1
    assert 92 in axes.index and 122 in axes.index


def test_class_weights_and_oversampling_follow_the_configs(axes):
    assert axes.loc[12, "Class weights"] == 1.0          # exp 12: use_class_weights: true
    assert axes.loc[92, "Class weights"] == 0.0
    assert axes.loc[122, "Hernia oversampling"] == 1.0   # exp 122: 10x hernia oversampling
    assert axes.loc[92, "Focal loss"] == 0.0             # exp 92: BCE
    assert axes.loc[12, "Focal loss"] == 1.0


def test_lost_configurations_are_not_filled_in(axes):
    config_axes = [c for c in axes.columns if not c.startswith("Per-class thresholds")]
    for exp in (1, 3):
        assert all(math.isnan(v) for v in axes.loc[exp, config_axes])


def test_figure_b2_progression_reports_original_early_runs(monkeypatch, tmp_path):
    import analysis.thesis_appendix_b_figures as b
    captured = {}
    monkeypatch.setattr(b, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(b.plt, "savefig", lambda *a, **k: None)
    real_df = b.pd.DataFrame

    def spy(data=None, *a, **k):
        df = real_df(data, *a, **k)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "mean_auc" in data[0]:
            captured["df"] = df
        return df

    monkeypatch.setattr(b.pd, "DataFrame", spy)
    monkeypatch.chdir(REPO)   # the script resolves its paths relative to the project root
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)   # no folder may hold several results files
        b.generate_figure_b3_auc_progression()
    df = captured["df"].set_index("iteration")
    assert df.loc[1, "mean_auc"] == pytest.approx(0.743, abs=5e-4)
    assert df.loc[2, "mean_auc"] == pytest.approx(0.802, abs=5e-4)
    assert df.loc[92, "mean_auc"] == pytest.approx(0.8204, abs=5e-5)


THR = "Per-class thresholds\n(applied)"


def test_threshold_axis_reflects_applied_thresholds(axes):
    # experiments 126-131 optimised the hernia threshold only (all others stay at 0.5)
    for exp in range(126, 132):
        assert axes.loc[exp, THR] == 0.0
    # experiment 12 has threshold_optimization in its config, but every class used 0.5
    assert axes.loc[12, THR] == 0.0
    assert axes.loc[92, THR] == 1.0
    assert axes.loc[50, THR] == 1.0


def test_threshold_rule_handles_missing_results_and_values(axes, tmp_path):
    from analysis.thesis_appendix_c_axes_timeline import _thresholds_applied
    for exp in range(113, 117):                     # folders without a results file
        assert math.isnan(axes.loc[exp, THR])
    assert math.isnan(_thresholds_applied(None))
    f = tmp_path / "pipeline_results_x.csv"
    f.write_text("Label,Threshold\nA,0.3\nB,\nC,0.5\n")
    assert math.isnan(_thresholds_applied(f))       # one off-default value plus one missing
    f.write_text("Label,Threshold\nA,0.3\nB,0.2\nC,\n")
    assert _thresholds_applied(f) == 1.0
