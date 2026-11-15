# Unassigned and overwritten files from experiment folders 1–5

During later runs, files from February 2026 were written into the folders of
experiments 1–5 (`iteration_001` … `iteration_005`). Analysis scripts that read
the *latest* results file therefore reported experiments 1–5 with the results
of experiments 126–130 (finding F5 in
`docs/review/Independent_Verification_Report.docx`).

The folders were cleaned on 2026-11-15:

| What | Action |
|---|---|
| Results, thresholds, confusion matrices and baseline comparisons byte-identical to files in `iteration_122` and `iteration_126`–`iteration_130` | Removed from folders 1–5 (the originals remain in those experiments' folders) |
| Two February runs (`20260204-224256`, `20260205-183525`) and `ITERATION_FAILED_001.txt`, found only in `iteration_001` and not attributable to a numbered experiment | Moved to `feb2026_runs_found_in_iteration_001/` |
| `config.yaml` of folders 1–5 (each a copy of experiment 126's config, `metadata.iteration: 126`), and their `iteration_summary.json` and `ai_analysis_00N.txt` (written by the February runs) | Moved to `overwritten_metadata_from_iteration_00N/` |
| Original configs of experiments 2, 4 and 5 | Restored from `experiments/configs/config_iteration_00N.yaml` |
| Original configs of experiments 1 and 3 | Lost; folders 1 and 3 have no `config.yaml` |

Each of folders 1–5 now holds only its original December 2025 results
(`pipeline_results_202512*.csv`), with macro AUC 0.743, 0.802, 0.785, 0.742 and
0.731. All analysis scripts select results files through
`analysis/iteration_files.py`, which uses the earliest file and warns if a
folder holds more than one. `tests/test_iteration_files.py` checks these
invariants.

This folder's name does not match `iteration_*`, so no loader reads it.
