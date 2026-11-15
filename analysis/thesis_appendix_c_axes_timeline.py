"""
Thesis Appendix C, Figure C.1: Timeline of experimental axes across experiments
==============================================================================

For every experiment folder (experiments/auto_improvement_runs/iteration_NNN)
this marks which of the main experimental axes was enabled:

    Focal loss                 config: loss.type in {FocalLoss, focal}
    Class weights              config: loss.use_class_weights
    Rare-class augmentation    config: augmentation.rare_class.enabled
    Per-class thresholds       results file: the decision thresholds actually applied
                               differ from the 0.5 default for at least two classes
    Hernia oversampling        config: sampling.hernia_oversample.enabled

The threshold axis is read from the results file rather than the config because
the config field was not always honoured: experiments 2-28 carry
`evaluation.threshold_optimization` in their configs, yet every class was
evaluated at 0.5 (see the note on experiment 12 in core/iteration_baselines.py).
Experiments 126-131 optimised the hernia threshold only, so they are not marked.

The other axes are the values of the saved configuration; at run time the
pipeline's role rules could override some of them (core/config_based_pipeline.py).
Cells with no information (no preserved config, or no results file) are grey;
experiment numbers with no folder are white.

Usage (from the project root):
    python analysis/thesis_appendix_c_axes_timeline.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `analysis.*` imports
from analysis.iteration_files import select_results_file

PROJECT_DIR = Path(__file__).resolve().parents[1]
ITERATIONS_DIR = PROJECT_DIR / "experiments" / "auto_improvement_runs"
OUTPUT_DIR = PROJECT_DIR / "experiments" / "figures" / "thesis_figures" / "appendix_c"

THRESHOLDS_AXIS = "Per-class thresholds\n(applied)"
CONFIG_AXES = [
    ("Focal loss", lambda c: _get(c, "loss.type") in ("FocalLoss", "focal")),
    ("Class weights", lambda c: bool(_get(c, "loss.use_class_weights"))),
    ("Rare-class augmentation\n(configured)", lambda c: bool(_get(c, "augmentation.rare_class.enabled"))),
    ("Hernia oversampling", lambda c: bool(_get(c, "sampling.hernia_oversample.enabled"))),
]
AXES = CONFIG_AXES[:3] + [(THRESHOLDS_AXIS, None)] + CONFIG_AXES[3:]   # display order
DEFAULT_THRESHOLD = 0.5


def _thresholds_applied(results_file: Path | None) -> float:
    """1.0 if per-class thresholds were applied (>= 2 classes off the 0.5 default), 0.0 if not,
    NaN if there is no results file."""
    if results_file is None:
        return np.nan
    thr = pd.read_csv(results_file)["Threshold"].astype(float)
    off_default = int(((thr - DEFAULT_THRESHOLD).abs() > 1e-9).sum())
    if off_default < 2 and off_default + int(thr.isna().sum()) >= 2:
        return np.nan                     # missing values make the decision indeterminate
    return float(off_default >= 2)


def _get(cfg: dict, path: str):
    value = cfg
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def collect_axes(iterations_dir: Path = ITERATIONS_DIR) -> pd.DataFrame:
    """One row per experiment folder; axis columns are 1/0, or NaN where there is no information."""
    rows = []
    for d in sorted(iterations_dir.glob("iteration_*")):
        m = re.fullmatch(r"iteration_(\d+)", d.name)
        if not m:
            continue
        cfg_path = d / "config.yaml"
        row = {"experiment": int(m.group(1))}
        cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else None
        for name, test in CONFIG_AXES:
            row[name] = float(test(cfg)) if cfg else np.nan
        row[THRESHOLDS_AXIS] = _thresholds_applied(select_results_file(d))
        rows.append(row)
    return pd.DataFrame(rows).set_index("experiment")[[name for name, _ in AXES]]


def plot(df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    full = df.reindex(range(1, df.index.max() + 1))  # experiments with no folder become gaps
    data = np.ma.masked_invalid(full.T.to_numpy(dtype=float))
    cmap = ListedColormap(["#E8ECEF", "#2C3E50"])
    cmap.set_bad("#FFFFFF")

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.imshow(data, aspect="auto", cmap=cmap, vmin=0, vmax=1, interpolation="nearest",
              extent=(0.5, full.index.max() + 0.5, len(AXES) - 0.5, -0.5))
    from matplotlib.patches import Rectangle
    for e in df.index:                              # grey cells: folder exists, information missing
        for row, (name, _) in enumerate(AXES):
            if np.isnan(df.loc[e, name]):
                ax.add_patch(Rectangle((e - 0.5, row - 0.5), 1, 1, facecolor="#BDC3C7", lw=0))
    ax.set_yticks(range(len(AXES)))
    ax.set_yticklabels([a for a, _ in AXES], fontsize=8)
    ax.set_xlabel("Experiment Number", fontweight="bold")
    ax.set_xlim(0.5, full.index.max() + 0.5)
    ax.set_title("Timeline of Experimental Axes Across Experiments", fontweight="bold")
    ax.legend(handles=[Patch(color="#2C3E50", label="Enabled"),
                       Patch(color="#E8ECEF", label="Disabled"),
                       Patch(facecolor="#BDC3C7", label="No information (config lost or no results)"),
                       Patch(facecolor="#FFFFFF", edgecolor="#999999", label="No experiment folder")],
              loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8, frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = collect_axes()
    out = OUTPUT_DIR / "appendix_figure_c1_axes_timeline.png"
    plot(df, out)
    print(f"Saved {out}")
    print(f"Experiments: {len(df)}; configuration not preserved: "
          f"{[e for e in df.index if df.loc[e, 'Focal loss'] != df.loc[e, 'Focal loss']]}")
    print("Experiments with each axis enabled:")
    for name, _ in AXES:
        print(f"  {name.replace(chr(10), ' ')}: {int(np.nansum(df[name]))}")


if __name__ == "__main__":
    main()
