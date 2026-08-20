"""
Aggregates SUTRAN_DA_results test-set pickles across seeds, grouped by
configuration (everything in the run folder name except `_seed_{n}`).

Usage
-----
python aggregate_results.py --log_name BPIC_19 --training_mode default_training
"""

import argparse
import os
import pickle
import re

import pandas as pd

RESULT_FILENAME = "averaged_results_CB.pkl"
DIAGNOSTICS_FILENAME = "consistency_diagnostics.pkl"

# Metric keys to pull out of the pickles if present. Extend this list as you
# add more diagnostics to avg_results_dict_CB / consistency_diagnostics.pkl.
METRIC_KEYS = [
    # --- each task scored the way it is actually meant to be scored ---
    # (these three are the native task metrics; already present in every
    # averaged_results_CB.pkl, so no reruns were needed to start tracking them)
    "MAE TTNE minutes",   # per-STEP next-event-gap error -- what the ttne head
                          # is trained for. Distinct from mae_ttne_sum_* below,
                          # which sums the suffix to estimate TOTAL remaining
                          # time; a head can improve at one and worsen at the other.
    "MAE RRT minutes",
    "DL sim",
    "mean_abs_gap_IB", "mean_abs_gap_CB",
    "mean_signed_gap_IB", "mean_signed_gap_CB",
    "signed_bias_rrt_IB", "signed_bias_rrt_CB",
    "signed_bias_ttne_sum_IB", "signed_bias_ttne_sum_CB",
    "mae_ttne_sum_IB", "mae_ttne_sum_CB",
    "mae_rrt_IB", "mae_rrt_CB",
    "error_correlation_IB",
    # "_raw" = destandardized WITHOUT clamping negative times at 0 (what the
    # first version of the diagnostics did); the unsuffixed keys above are the
    # clamped, paper-consistent primaries. Kept so the two can be compared.
    "mean_abs_gap_CB_raw", "mean_signed_gap_CB_raw",
    "signed_bias_rrt_CB_raw", "signed_bias_ttne_sum_CB_raw",
    "mae_ttne_sum_CB_raw", "mae_rrt_CB_raw",
    "error_correlation_IB_raw",
    # How often clamping actually applies.
    "frac_neg_ttne_step_preds", "frac_neg_rrt_preds",

    # --- outcome head -------------------------------------------------------
    # These have been written into averaged_results_*.pkl by every outcome-log
    # run all along; they were simply never listed here, so `load_metrics` (which
    # uses .get(k, None)) dropped them silently. HANDOFF §8.4 previously recorded
    # the outcome head as "unmeasured" -- it was measured, just not surfaced.
    # No rerun or backfill is needed for these.
    "Multi-Class Accuracy", "Macro-F1", "Weighted-F1",
    "Macro-Precision", "Macro-Recall", "CE",

    # --- axiom 2: outcome vs activity-suffix consistency --------------------
    # Requires re-inference (backfill_diagnostics.py): the implied outcome is
    # derived from the DECODED suffix, which is not persisted.
    "outcome_head_accuracy_IB", "outcome_head_accuracy_CB",
    "outcome_suffix_accuracy_IB", "outcome_suffix_accuracy_CB",
    "outcome_head_vs_suffix_agreement_IB",
    "outcome_disagreement_rate_IB", "outcome_disagreement_rate_CB",
    "outcome_head_macro_f1_IB", "outcome_suffix_macro_f1_IB",
    "frac_suffix_no_determining_act", "num_outcome_instances_evaluated",
]

SEED_PATTERN = re.compile(r"_seed_(\d+)$")


def find_run_dirs(log_name: str, training_mode: str | None):
    """
    Yield (config_key, seed, pickle_path) for every run under `log_name/`
    whose folder name starts with SUTRAN_DA_results. The results pickle is
    located via a recursive search rather than a fixed subpath, since the
    exact nesting (training mode / MTO-technique subfolders) can vary.
    If `training_mode` is given, prefer matches whose path contains it;
    otherwise just use whatever match is found.
    """
    base = log_name
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Log results directory not found: {base}")

    for entry in sorted(os.listdir(base)):
        full_dir = os.path.join(base, entry)
        if not os.path.isdir(full_dir) or not entry.startswith("SUTRAN_DA_results"):
            continue

        match = SEED_PATTERN.search(entry)
        if not match:
            continue  # skip anything that doesn't end in _seed_{n}

        seed = int(match.group(1))
        config_key = entry[: match.start()]  # strip trailing _seed_{n}

        # Recursively search this run's directory tree for the results file.
        candidates = []
        for root, _dirs, files in os.walk(full_dir):
            if RESULT_FILENAME in files:
                candidates.append(os.path.join(root, RESULT_FILENAME))

        if not candidates:
            continue

        pkl_path = candidates[0]
        if training_mode:
            matching = [c for c in candidates if training_mode in c]
            if matching:
                pkl_path = matching[0]
        if len(candidates) > 1:
            print(f"Warning: multiple '{RESULT_FILENAME}' found under {full_dir}, using {pkl_path}")

        yield config_key, seed, pkl_path


def load_metrics(pkl_path: str) -> dict:
    with open(pkl_path, "rb") as f:
        results_dict = pickle.load(f)
    metrics = {k: results_dict.get(k, None) for k in METRIC_KEYS}

    # Merge in the separately-stored consistency diagnostics, if present
    # (same directory as the main results pickle).
    diagnostics_path = os.path.join(os.path.dirname(pkl_path), DIAGNOSTICS_FILENAME)
    if os.path.isfile(diagnostics_path):
        with open(diagnostics_path, "rb") as f:
            diagnostics = pickle.load(f)
        for key in METRIC_KEYS:
            if key in diagnostics:
                metrics[key] = diagnostics[key]

    return metrics


def aggregate(log_name: str, training_mode: str | None = None,
              seeds: list[int] | None = None) -> pd.DataFrame:
    """Collect every completed run under `log_name/` into a per-seed table and a
    mean/std-over-seeds summary.

    Parameters
    ----------
    seeds : list of int, optional
        Restrict to these seeds. Needed whenever one config name exists on more
        than one hardware generation: on BPIC_17_DR the baseline and the
        `_ltn_{0.5,1.0}_detach_ttne` configs were run both locally (seeds 3, 17,
        Quadro P1000) and on the cluster (seeds 101, 102, A30 MIG). Grouping is
        by config with seed treated as a repeat, so without a filter those rows
        silently average across GPU architectures -- and the effects under study
        are 1-2% (HANDOFF §9.3). Pass the block belonging to one machine.
    """
    rows = []
    for config_key, seed, pkl_path in find_run_dirs(log_name, training_mode):
        if seeds is not None and seed not in seeds:
            continue
        metrics = load_metrics(pkl_path)
        metrics["config"] = config_key
        metrics["seed"] = seed
        rows.append(metrics)

    if not rows:
        seed_note = f" with seed in {seeds}" if seeds is not None else ""
        raise RuntimeError(
            f"No '{RESULT_FILENAME}' found anywhere under "
            f"'{log_name}/SUTRAN_DA_results*_seed_*/'{seed_note}. "
            "Check that at least one run has completed and that the log_name matches "
            "the folder actually produced by TRAIN_EVAL_EQUAL_WEIGHTING.py."
        )

    per_seed_df = pd.DataFrame(rows)

    # Group by config, aggregate mean/std over seeds for each metric.
    grouped = per_seed_df.groupby("config")[METRIC_KEYS].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    return per_seed_df.sort_values(["config", "seed"]), grouped


def format_summary_table(grouped: pd.DataFrame) -> str:
    """Build a compact mean±std markdown table, one row per config."""
    display_rows = []
    for _, row in grouped.iterrows():
        display_row = {"config": row["config"], "n_seeds": int(row[f"{METRIC_KEYS[0]}_count"])}
        for key in METRIC_KEYS:
            mean_val = row.get(f"{key}_mean")
            std_val = row.get(f"{key}_std")
            if pd.isna(mean_val):
                display_row[key] = "-"
            elif pd.isna(std_val):
                display_row[key] = f"{mean_val:.4f}"
            else:
                display_row[key] = f"{mean_val:.4f} ± {std_val:.4f}"
        display_rows.append(display_row)

    display_df = pd.DataFrame(display_rows)
    return display_df.to_markdown(index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate SuTraN+ LTN experiment results across seeds.")
    parser.add_argument("--log_name", required=True, help="Event log identifier / results base folder (e.g. BPIC_19).")
    parser.add_argument(
        "--training_mode",
        default=None,
        choices=["default_training", "CaLenDiR_training", None],
        help=(
            "Optional hint used only to disambiguate if a run directory somehow "
            "contains more than one averaged_results_CB.pkl. Not required for "
            "the normal single-technique-per-run layout."
        ),
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help=(
            "Comma-separated seeds to restrict to, e.g. '101,102'. Use this when a "
            "config name exists on more than one machine: BPIC_17_DR's baseline and "
            "its _ltn_{0.5,1.0}_detach_ttne configs were run both locally (3,17) and "
            "on the cluster (101,102), and without a filter the summary averages "
            "across GPU architectures."
        ),
    )
    parser.add_argument("--out_csv", default=None, help="Optional path to save the raw per-seed table as CSV.")
    parser.add_argument("--out_summary_csv", default=None, help="Optional path to save the aggregated mean/std table as CSV.")
    args = parser.parse_args()

    seed_filter = None
    if args.seeds:
        seed_filter = [int(s) for s in args.seeds.split(",") if s.strip()]
        print(f"Restricting to seeds {seed_filter}.")

    per_seed_df, grouped_df = aggregate(args.log_name, args.training_mode, seed_filter)

    # Warn when a config spans several seeds that were never meant to be pooled.
    # Cheap, and it makes the hazard visible instead of relying on the reader
    # remembering which seeds came from which machine.
    if seed_filter is None:
        MACHINE_BLOCKS = {"local (P1000)": {3, 5, 17, 23, 31, 47}, "cluster (MIG)": {101, 102}}
        # A hardcoded seed->machine map goes stale the moment a new seed is used,
        # and a check that silently stops checking is worse than no check: it
        # gives a green light that is not looking at anything. So announce any
        # seed we cannot classify, rather than passing it over.
        unclassified = set(per_seed_df["seed"]) - set().union(*MACHINE_BLOCKS.values())
        if unclassified:
            print(f"!! WARNING: seed(s) {sorted(unclassified)} are not listed in "
                  f"MACHINE_BLOCKS, so cross-machine pooling CANNOT be detected for "
                  f"them. Add them to MACHINE_BLOCKS in aggregate_results.py.")
        for cfg, grp in per_seed_df.groupby("config"):
            present = set(grp["seed"])
            blocks = [name for name, block in MACHINE_BLOCKS.items() if present & block]
            if len(blocks) > 1:
                print(f"!! WARNING: config '{cfg}' pools seeds from {blocks} "
                      f"({sorted(present)}). Different GPU architectures; pass "
                      f"--seeds to separate them.")

    print("\n=== Per-seed results ===")
    print(per_seed_df.to_string(index=False))

    print("\n=== Aggregated (mean ± std over seeds) ===")
    print(format_summary_table(grouped_df))

    if args.out_csv:
        per_seed_df.to_csv(args.out_csv, index=False)
        print(f"\nSaved per-seed table to {args.out_csv}")

    if args.out_summary_csv:
        grouped_df.to_csv(args.out_summary_csv, index=False)
        print(f"Saved aggregated table to {args.out_summary_csv}")