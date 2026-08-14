"""
Backfills the consistency diagnostics for ALREADY-COMPLETED runs, so that
newly-added metrics (currently: mae_ttne_sum_* and mae_rrt_*) are available
for results that were produced before those metrics existed.

NO RETRAINING happens here. For each run directory this script:
  1. reads that run's `backup_results.csv`,
  2. re-derives which epoch was selected as best (same `select_best_epoch`
     logic the original training used),
  3. reloads that epoch's saved checkpoint,
  4. re-runs test-set inference to regenerate `consistency_diagnostics.pkl`
     with the full, current metric set.

Because inference is deterministic in eval mode, everything else about the
run is reproduced exactly -- the script verifies this by comparing the
recomputed RRT MAE against the value already stored in `averaged_results_CB.pkl`
and warning loudly if they disagree (which would mean the reloaded model is
not the one the original numbers came from).

Cost: roughly 15-20 minutes of GPU time per run (test-set inference uses
autoregressive decoding over the full ~250k-instance test set), so budget
~8 hours for all 29 runs. Runs sequentially and can be safely interrupted --
already-backfilled runs are skipped on the next invocation unless you pass
--force.

Usage
-----
    # everything not yet backfilled (the normal case)
    python backfill_diagnostics.py

    # preview what would run, without touching the GPU
    python backfill_diagnostics.py --dry_run

    # only a subset (substring match on the run-folder name)
    python backfill_diagnostics.py --filter detach_ttne

    # redo runs that already have the new metrics
    python backfill_diagnostics.py --force
"""

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch

from TRAIN_EVAL_FUNCTIONALITY import log_configs
from Utils.callback_selection import get_target_metrics_dict, select_best_epoch

# Metrics whose presence means a run has already been backfilled.
REQUIRED_NEW_KEYS = [
    "mae_ttne_sum_IB", "mae_ttne_sum_CB", "mae_rrt_IB", "mae_rrt_CB",
    # clamped-vs-raw comparison + how often clamping applies
    "mae_ttne_sum_CB_raw", "frac_neg_ttne_step_preds", "frac_neg_rrt_preds",
]

# Must match the architecture hardcoded in TRAIN_EVAL_EQUAL_WEIGHTING.train_eval.
D_MODEL = 32
NUM_PREFIX_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
NUM_HEADS = 8
DROPOUT = 0.2
LAYERNORM_EMBEDS = True
REMAINING_RUNTIME_HEAD = True
TEST_BATCH_SIZE = 2048

# Tolerance for the "did we reload the right model" integrity check, as a
# relative difference on RRT MAE (minutes). Inference is deterministic, so
# this should be ~0; anything above this is a real mismatch worth stopping for.
MAE_MATCH_RTOL = 1e-4


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def find_run_dirs(log_name, name_filter=None):
    """Yields (run_name, backup_path, results_path) for every completed run."""
    for entry in sorted(os.listdir(log_name)):
        full_dir = os.path.join(log_name, entry)
        if not os.path.isdir(full_dir) or not entry.startswith("SUTRAN_DA_results"):
            continue
        if name_filter and name_filter not in entry:
            continue
        # Locate the training-mode / technique subfolder holding backup_results.csv.
        for root, _dirs, files in os.walk(full_dir):
            if "backup_results.csv" in files:
                results_path = os.path.join(root, "TEST_SET_RESULTS")
                if os.path.isdir(results_path):
                    yield entry, root, results_path
                break


def already_backfilled(results_path):
    diag_path = os.path.join(results_path, "consistency_diagnostics.pkl")
    if not os.path.isfile(diag_path):
        return False
    diagnostics = load_pickle(diag_path)
    return all(k in diagnostics for k in REQUIRED_NEW_KEYS)


def backfill_run(run_name, backup_path, results_path, log_name, device):
    """Reloads the best checkpoint for one run and regenerates its diagnostics."""
    from SuTraN.SuTraN import SuTraN
    from SuTraN.inference_procedure import inference_loop
    from TRAIN_EVAL_FUNCTIONALITY.TRAIN_EVAL_EQUAL_WEIGHTING import load_checkpoint

    data_path = log_name
    outcome_bool = log_configs.outcome_bools_dict[log_name]
    out_mask = log_configs.out_masks_dict[log_name]
    out_type = log_configs.out_types_dict[log_name]
    num_outclasses = log_configs.num_outclasses_dict[log_name]

    bin_outbool = (out_type == "binary_outcome")
    multic_outbool = (out_type == "multiclass_outcome")

    # --- Metadata / standardization stats (same as train_eval) ---
    cardinality_dict = load_pickle(os.path.join(data_path, log_name + "_cardin_dict.pkl"))
    num_activities = cardinality_dict["concept:name"] + 2
    cardinality_list_prefix = load_pickle(os.path.join(data_path, log_name + "_cardin_list_prefix.pkl"))
    num_cols_dict = load_pickle(os.path.join(data_path, log_name + "_num_cols_dict.pkl"))
    cat_cols_dict = load_pickle(os.path.join(data_path, log_name + "_cat_cols_dict.pkl"))
    train_means_dict = load_pickle(os.path.join(data_path, log_name + "_train_means_dict.pkl"))
    train_std_dict = load_pickle(os.path.join(data_path, log_name + "_train_std_dict.pkl"))

    mean_std_ttne = [train_means_dict["timeLabel_df"][0], train_std_dict["timeLabel_df"][0]]
    mean_std_tsp = [train_means_dict["suffix_df"][1], train_std_dict["suffix_df"][1]]
    mean_std_tss = [train_means_dict["suffix_df"][0], train_std_dict["suffix_df"][0]]
    mean_std_rrt = [train_means_dict["timeLabel_df"][1], train_std_dict["timeLabel_df"][1]]

    num_numericals_pref = len(num_cols_dict["prefix_df"])
    num_categoricals_pref = len(cat_cols_dict["prefix_df"])

    # --- Test set (never subsetted, so this matches the original run exactly) ---
    test_dataset = torch.load(os.path.join(data_path, "test_tensordataset.pt"))
    og_caseint_test = torch.load(os.path.join(data_path, "og_caseint_test.pt"))
    if outcome_bool and out_mask:
        instance_mask_out_test = torch.load(os.path.join(data_path, "instance_mask_out_test.pt"))
    else:
        instance_mask_out_test = None
        out_mask = False

    # --- Re-derive the best epoch exactly as the original run did ---
    df = pd.read_csv(os.path.join(backup_path, "backup_results.csv"))
    task_list = ["activity_suffix", "timestamp_suffix"]
    if REMAINING_RUNTIME_HEAD:
        task_list.append("remaining_runtime")
    if bin_outbool:
        task_list.append("binary_outcome")
    if multic_outbool:
        task_list.append("multiclass_outcome")
    best_epoch, _ = select_best_epoch(df, get_target_metrics_dict(task_list))
    best_epoch = int(best_epoch)

    checkpoint_path = os.path.join(backup_path, f"model_epoch_{best_epoch}.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"best-epoch checkpoint missing for {run_name}: {checkpoint_path}"
        )
    print(f"    best epoch = {best_epoch}")

    model = SuTraN(
        num_activities=num_activities,
        d_model=D_MODEL,
        cardinality_categoricals_pref=cardinality_list_prefix,
        num_numericals_pref=num_numericals_pref,
        num_prefix_encoder_layers=NUM_PREFIX_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=4 * D_MODEL,
        dropout=DROPOUT,
        remaining_runtime_head=REMAINING_RUNTIME_HEAD,
        layernorm_embeds=LAYERNORM_EMBEDS,
        outcome_bool=outcome_bool,
        out_type=out_type,
        num_outclasses=num_outclasses,
    )
    model.to(device)
    model, _, _, _ = load_checkpoint(model, path_to_checkpoint=checkpoint_path,
                                     train_or_eval="eval", lr=0.002)
    model.to(device)
    model.eval()

    # --- Re-run test-set inference (regenerates consistency_diagnostics.pkl) ---
    _, inf_results_CB, _ = inference_loop(
        model=model,
        inference_dataset=test_dataset,
        remaining_runtime_head=REMAINING_RUNTIME_HEAD,
        outcome_bool=outcome_bool,
        out_mask=out_mask,
        out_type=out_type,
        num_outclasses=num_outclasses,
        num_categoricals_pref=num_categoricals_pref,
        mean_std_ttne=mean_std_ttne,
        mean_std_tsp=mean_std_tsp,
        mean_std_tss=mean_std_tss,
        mean_std_rrt=mean_std_rrt,
        og_caseint=og_caseint_test,
        instance_mask_out=instance_mask_out_test,
        results_path=results_path,
        val_batch_size=TEST_BATCH_SIZE,
    )

    # --- Integrity check: recomputed RRT MAE must match what's already stored ---
    recomputed_rrt_mae = inf_results_CB[4]  # CB MAE RRT in minutes
    stored_path = os.path.join(results_path, "averaged_results_CB.pkl")
    if os.path.isfile(stored_path):
        stored_rrt_mae = load_pickle(stored_path).get("MAE RRT minutes")
        if stored_rrt_mae is not None:
            rel_diff = abs(recomputed_rrt_mae - stored_rrt_mae) / max(abs(stored_rrt_mae), 1e-9)
            if rel_diff > MAE_MATCH_RTOL:
                raise RuntimeError(
                    f"INTEGRITY CHECK FAILED for {run_name}: recomputed RRT MAE "
                    f"{recomputed_rrt_mae:.4f} != stored {stored_rrt_mae:.4f} "
                    f"(rel diff {rel_diff:.2e}). The reloaded checkpoint does not "
                    f"reproduce the original results -- diagnostics NOT trustworthy. "
                    f"Investigate before using this run."
                )
            print(f"    integrity OK (RRT MAE {recomputed_rrt_mae:.2f}, rel diff {rel_diff:.2e})")

    diagnostics = load_pickle(os.path.join(results_path, "consistency_diagnostics.pkl"))
    missing = [k for k in REQUIRED_NEW_KEYS if k not in diagnostics]
    if missing:
        raise RuntimeError(
            f"{run_name}: diagnostics regenerated but still missing {missing}. "
            f"Is SuTraN/inference_procedure.py up to date?"
        )
    print(f"    mae_ttne_sum_CB = {diagnostics['mae_ttne_sum_CB'] / 60:,.0f} min | "
          f"mae_rrt_CB = {diagnostics['mae_rrt_CB'] / 60:,.0f} min")
    # Surface how much the clamping choice actually changed things, so a material
    # difference is visible immediately rather than only after re-aggregating.
    bias_clamped = diagnostics["signed_bias_ttne_sum_CB"] / 60
    bias_raw = diagnostics["signed_bias_ttne_sum_CB_raw"] / 60
    print(f"    Sigma-ttne bias: {bias_clamped:,.0f} min clamped vs "
          f"{bias_raw:,.0f} min raw  (negative preds: "
          f"{diagnostics['frac_neg_ttne_step_preds']:.2%} of ttne steps, "
          f"{diagnostics['frac_neg_rrt_preds']:.2%} of rrt)")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log_name", default="BPIC_19",
                        help="Event log / results base folder (default: BPIC_19).")
    parser.add_argument("--filter", default=None,
                        help="Only process runs whose folder name contains this substring.")
    parser.add_argument("--force", action="store_true",
                        help="Reprocess runs that already have the new metrics.")
    parser.add_argument("--dry_run", action="store_true",
                        help="List what would be processed, then exit without using the GPU.")
    args = parser.parse_args()

    if not os.path.isdir(args.log_name):
        sys.exit(f"Results folder not found: {args.log_name}")

    runs = list(find_run_dirs(args.log_name, args.filter))
    if not runs:
        sys.exit(f"No completed runs found under {args.log_name}/ "
                 f"(filter={args.filter!r}).")

    todo, skipped = [], []
    for run_name, backup_path, results_path in runs:
        if not args.force and already_backfilled(results_path):
            skipped.append(run_name)
        else:
            todo.append((run_name, backup_path, results_path))

    print(f"Found {len(runs)} run(s): {len(todo)} to process, {len(skipped)} already done.")
    if skipped:
        print(f"  (skipping, already backfilled: {len(skipped)} -- use --force to redo)")
    if args.dry_run:
        print("\n--dry_run: the following would be processed:")
        for run_name, _, _ in todo:
            print(f"  {run_name}")
        print(f"\nEstimated GPU time: ~{len(todo) * 17 / 60:.1f} hours "
              f"({len(todo)} runs x ~17 min).")
        return

    if not todo:
        print("Nothing to do.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"Estimated total: ~{len(todo) * 17 / 60:.1f} hours. Safe to interrupt "
          f"between runs -- completed runs are skipped on the next invocation.\n")

    failures = []
    for i, (run_name, backup_path, results_path) in enumerate(todo, start=1):
        print(f"[{i}/{len(todo)}] {run_name}")
        try:
            backfill_run(run_name, backup_path, results_path, args.log_name, device)
            print("    done.\n")
        except Exception as exc:  # keep going; report everything at the end
            print(f"    !!! FAILED: {exc}\n")
            failures.append((run_name, str(exc)))

    print("=" * 70)
    print(f"Backfill complete: {len(todo) - len(failures)} succeeded, {len(failures)} failed.")
    for run_name, msg in failures:
        print(f"  FAILED  {run_name}: {msg}")
    if failures:
        sys.exit(1)
    print("\nRe-run aggregate_results.py / ltn_presentation_notebook.ipynb to pick up "
          "the new metrics.")


if __name__ == "__main__":
    main()
