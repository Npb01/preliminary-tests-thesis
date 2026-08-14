"""
Runs the 12 new training configurations for the revised LTN preliminary
study (lambda sweep + corrected diagnostics -- see the "LTN Cross-Task
Consistency" experiment plan).

Configs "baseline" (seed=3) and "LTN free, lambda=0.1" (seed=3) already
completed successfully in a prior run and are NOT rerun here -- their
results live in BPIC_19/SUTRAN_DA_results_subset_0.15_seed_3/ and
BPIC_19/SUTRAN_DA_results_subset_0.15_ltn_0.1_seed_3/ respectively, and
aggregate_results.py picks them up automatically alongside these new runs.

The detach-mode and loss-balanced-control configs from the original
6-config design were affected by a bug in run_mto_experiment.py that
silently dropped --detach_mode/--balance_losses/--scale_ttne/--scale_rrt
before reaching train_eval() -- they are rerun here now that it's fixed.
The balanced+LTN config never successfully ran and is included fresh.

Each config is launched as a subprocess via run_mto_experiment.py, with
its own log file saved alongside so nothing printed to stdout is lost.
"""

import subprocess
import sys
from pathlib import Path

LOG_NAME = "BPIC_19"
SEED_A = 3   # matches the seed used for the completed baseline / lambda=0.1 runs
SEED_B = 17  # second seed for the headline lambda-sweep configs

NUM_EPOCHS = 40
PATIENCE = 8

# Fraction of the validation set (case-level) used for per-epoch validation
# in these new runs. Cuts ~155s/epoch validation time roughly in half while
# keeping enough batches (~10+) to avoid the noisy-checkpoint-selection bug
# that ruled out the old subset_fraction=0.15 used previously for this.
# NOTE: the reused baseline/lambda=0.1 seed=3 runs were trained with full
# (unsubsetted) validation -- this only affects which epoch's checkpoint
# gets selected, not final test-set comparability across configs.
VAL_SUBSET_FRACTION = 0.5

# From BPIC_19/SUTRAN_DA_results_subset_0.15_seed_3/CaLenDiR_training/
# Default_Equal_Weighting/backup_results.csv, row for the *actual*
# checkpoint-selected best epoch (30, per select_best_epoch()) -- NOT the
# last epoch (39), which is close but not identical (RRT differs ~0.6%).
SCALE_TTNE = 0.4014
SCALE_RRT = 0.3516

LOG_DIR = Path("run_logs")
LOG_DIR.mkdir(exist_ok=True)

# (name, seed, extra CLI args)
CONFIGS = [
    # --- Lambda sweep, both heads free (detach_mode=none) ---
    ("baseline_seed17",        SEED_B, ["--lambda_ltn", "0.0"]),
    ("ltn_lambda0.01_seed3",   SEED_A, ["--lambda_ltn", "0.01"]),
    ("ltn_lambda0.01_seed17",  SEED_B, ["--lambda_ltn", "0.01"]),
    ("ltn_lambda0.05_seed3",   SEED_A, ["--lambda_ltn", "0.05"]),
    ("ltn_lambda0.05_seed17",  SEED_B, ["--lambda_ltn", "0.05"]),
    ("ltn_lambda0.1_seed17",   SEED_B, ["--lambda_ltn", "0.1"]),
    ("ltn_lambda0.5_seed3",    SEED_A, ["--lambda_ltn", "0.5"]),
    ("ltn_lambda0.5_seed17",   SEED_B, ["--lambda_ltn", "0.5"]),

    # --- Detach-mode diagnostics (cheating-risk isolation), lambda=0.1 ---
    ("detach_ttne_rerun",      SEED_A, ["--lambda_ltn", "0.1", "--detach_mode", "ttne"]),
    ("detach_rrt_rerun",       SEED_A, ["--lambda_ltn", "0.1", "--detach_mode", "rrt"]),

    # --- Loss-balanced control / balanced+LTN ---
    ("balanced_only_rerun",    SEED_A, ["--lambda_ltn", "0.0", "--balance_losses",
                                         "--scale_ttne", str(SCALE_TTNE), "--scale_rrt", str(SCALE_RRT)]),
    ("balanced_ltn_lambda0.1", SEED_A, ["--lambda_ltn", "0.1", "--balance_losses",
                                         "--scale_ttne", str(SCALE_TTNE), "--scale_rrt", str(SCALE_RRT)]),
]

if SCALE_TTNE == 1.0 or SCALE_RRT == 1.0:
    print("WARNING: SCALE_TTNE/SCALE_RRT are still at placeholder value 1.0.")
    print("The two balance_losses configs will run with no real rescaling until you fix this.\n")

for name, seed, extra_args in CONFIGS:
    cmd = [
        sys.executable, "-u", "-m", "TRAIN_EVAL_FUNCTIONALITY.run_mto_experiment",
        "--log_name", LOG_NAME,
        "--MTO_technique", "equal_weighting",
        "--seed", str(seed),
        "--subset_fraction", "0.15",
        "--val_subset_fraction", str(VAL_SUBSET_FRACTION),
        "--num_epochs", str(NUM_EPOCHS),
        "--patience", str(PATIENCE),
    ] + extra_args
    log_path = LOG_DIR / f"{name}_seed{seed}.log"
    print(f"\n=== Running {name} (seed={seed}) ===")
    print("Command:", " ".join(cmd))
    print(f"Logging to: {log_path}")

    with open(log_path, "w", encoding="utf-8") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if result.returncode != 0:
        print(f"!!! {name} FAILED (exit code {result.returncode}). "
              f"Check {log_path} for details. Stopping here.")
        sys.exit(result.returncode)
    else:
        print(f"{name} completed successfully.")

print(f"\nAll {len(CONFIGS)} new configs completed. Run aggregate_results.py / the notebook to view results.")
